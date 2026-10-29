"""Load the student's brick files.

Pipeline for one brick: read (utf-8) -> paste diagnostics -> compile -> definition checks (ast)
-> exec into a FRESH module registered as sys.modules["bricks.bNN_name"].
Every problem becomes a BrickProblem with a friendly English explanation, the line and a fix.
"""
import ast
import bisect
import hashlib
import re
import sys
import traceback
import types
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import LABS, MARKER, NAME_TO_LAB, ROOT, TORCH_INSTALL_CMD

# Characters that word processors, slides and chats put in place of plain ASCII.
CURLY_QUOTES = {
    "\u201c": "“", "\u201d": "”", "\u2018": "‘", "\u2019": "’",
    "\u201e": "„", "\u201a": "‚", "\u00ab": "«", "\u00bb": "»", "\u2032": "′", "\u2033": "″",
}
TYPOGRAPHIC = {
    "\u2013": ("–", "en dash", "-"),
    "\u2014": ("—", "em dash", "-"),
    "\u2212": ("−", "typographic minus sign", "-"),
    "\u2026": ("…", "typographic ellipsis", "..."),
}
INVISIBLE_NAMES = {
    "\u00a0": "non-breaking space (NBSP)",
    "\u202f": "narrow non-breaking space",
    "\u2009": "thin space",
    "\u200a": "hair space",
    "\u2007": "figure space",
    "\u3000": "ideographic space",
    "\u200b": "zero-width space",
    "\u200c": "invisible zero-width non-joiner",
    "\u200d": "invisible zero-width joiner",
    "\u2060": "invisible connector (word joiner)",
    "\ufeff": "invisible leading mark (BOM)",
    "\u00ad": "invisible hyphen (soft hyphen)",
    "\u2028": "Unicode line separator",
    "\u2029": "Unicode paragraph separator",
    "\u200e": "invisible text direction mark",
    "\u200f": "invisible text direction mark",
}

# Registry of the sources loaded so far: compiled filename -> BrickSource (used for tracebacks).
_SOURCES = {}


class BrickProblem(Exception):
    """A problem in a brick file, explained in English for the student."""

    def __init__(self, title, message, hint=None, *, kind="error", lab=None, path=None,
                 line=None, col=None, code=None, python_message=None):
        super().__init__(message)
        self.title = title
        self.message = message
        self.hint = hint
        self.kind = kind
        self.lab = lab
        self.path = path
        self.line = line
        self.col = col
        self.code = code
        self.python_message = python_message

    def as_dict(self):
        return {
            "title": self.title, "message": self.message, "hint": self.hint, "kind": self.kind,
            "lab": self.lab, "path": self.path, "line": self.line, "col": self.col,
            "code": self.code, "python_message": self.python_message,
        }


@dataclass
class BrickSource:
    lab: object                 # LabInfo
    path: Path
    display_path: str           # e.g. "bricks/b01_tokenizer.py"
    text: str
    lines: list
    marker_line: int            # 1-based line of the marker, 0 when the marker is missing
    code: str                   # text below the marker (whole file without marker)
    warnings: list = field(default_factory=list)

    def line(self, number):
        if number and 1 <= number <= len(self.lines):
            return self.lines[number - 1]
        return None

    def problem(self, title, message, hint=None, **kwargs):
        line = kwargs.pop("line", None)
        if line is not None and "code" not in kwargs:
            kwargs["code"] = self.line(line)
        return BrickProblem(title, message, hint, lab=self.lab.number, path=self.display_path,
                            line=line, **kwargs)


@dataclass
class LoadedBrick:
    source: BrickSource
    module: types.ModuleType
    digest: str                 # hash of the syntax tree: comments and blank lines do not count


def tree_digest(tree):
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:16]


def display_path(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


# --------------------------------------------------------------------------- reading
def read_brick(lab, directory):
    """Read bricks/bNN_name.py (or reference/bNN_name.py in explore mode)."""
    path = Path(directory) / f"{lab.brick}.py"
    shown = display_path(path)
    if not path.exists():
        raise BrickProblem(
            "File not found",
            f"The file {shown} was not found.",
            "It is included in the workshop package. Create a file with this name in the "
            "bricks/ folder or click \"Complete this step\".",
            lab=lab.number, path=shown)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise BrickProblem(
            "The file is not UTF-8",
            f"The file {shown} was saved with a different character encoding and accented characters cannot be read.",
            "In VS Code, click \"UTF-8\" (or the encoding name) in the bottom bar, choose "
            "\"Save with Encoding\" and then \"UTF-8\".",
            lab=lab.number, path=shown)
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = text.split("\n")
    marker_line = 0
    for i, line in enumerate(lines, start=1):
        if line.strip() == MARKER:
            marker_line = i
            break
    code = "\n".join(lines[marker_line:]) if marker_line else text
    return BrickSource(lab, path, shown, text, lines, marker_line, code)


# --------------------------------------------------------------------------- paste diagnostics
def _skip_string(text, i, quote, triple):
    """Return the index just after the string literal that starts at text[i]."""
    n = len(text)
    i += 3 if triple else 1
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if triple:
            if text.startswith(quote * 3, i):
                return i + 3
        elif ch == quote:
            return i + 1
        elif ch == "\n":
            return i  # unterminated single-line string: compile() will report it
        i += 1
    return n


def code_char_indices(text):
    """Yield the index of every character that is outside string literals and comments."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "#":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch in "'\"":
            i = _skip_string(text, i, ch, text.startswith(ch * 3, i))
            continue
        yield i
        i += 1


def _classify(ch):
    if ch in CURLY_QUOTES:
        return "quote"
    if ch in TYPOGRAPHIC:
        return "typographic"
    if ch in INVISIBLE_NAMES:
        return "invisible"
    if ch in "\t\n\x0c ":
        return None
    category = unicodedata.category(ch)
    if category in ("Cc", "Cf", "Co", "Cs", "Zl", "Zp") or category == "Zs":
        return "invisible"
    return None


def _has_code(lines):
    return any(line.strip() and not line.strip().startswith("#") for line in lines)


def has_student_code(text):
    """True when a brick file has code (not only the header comments)."""
    return _has_code(text.split("\n"))


def diagnose_paste(src):
    """Friendly checks that run BEFORE compile(): empty file, curly quotes, invisible characters."""
    lab = src.lab
    below = src.lines[src.marker_line:] if src.marker_line else src.lines
    if src.marker_line and not _has_code(below):
        above = src.lines[: src.marker_line - 1]
        if _has_code(above):
            raise src.problem(
                "The code is above the marker line",
                f"There is no code below the line {MARKER} in {src.display_path} yet: "
                "the code was pasted above it.",
                f"Cut the code (Ctrl+X) and paste it below the line {MARKER}.", kind="empty")
        raise src.problem(
            "No code yet",
            f"There is no code below the line {MARKER} in {src.display_path} yet.",
            "In the guide, click Copy; in VS Code, open the file and paste below the marker line. "
            "If you just pasted it, wait a second (VS Code saves automatically) or press Ctrl+S "
            "and click Verify again.", kind="empty")
    if not src.marker_line and not _has_code(src.lines):
        raise src.problem("No code yet", f"The file {src.display_path} has no code.",
                          "Paste the code from the guide into the file and click Verify again.", kind="empty")
    if not src.marker_line and src.display_path.startswith("bricks/"):
        src.warnings.append(f"The line {MARKER} was deleted from {src.display_path}. That's fine: the entire "
                            "file was read.")

    found = {"quote": [], "invisible": [], "typographic": []}
    for index in code_char_indices(src.text):
        kind = _classify(src.text[index])
        if kind:
            found[kind].append(index)
    if not any(found.values()):
        return

    line_starts = [0]
    for i, ch in enumerate(src.text):
        if ch == "\n":
            line_starts.append(i + 1)

    def where(index):
        line = bisect.bisect_right(line_starts, index)
        return line, index - line_starts[line - 1] + 1

    def more(count):
        return f" There are {count} such characters in the file." if count > 1 else ""

    if found["quote"]:
        first = found["quote"][0]
        line, col = where(first)
        chars = sorted({src.text[i] for i in found["quote"]})
        shown = " ".join(CURLY_QUOTES[c] for c in chars)
        raise src.problem(
            "Curly quotes in the code",
            f"Line {line} has curly quotes ({shown}). They appear when code passes through slides, "
            f"Word, PDF or some chats, and Python does not accept them.{more(len(found['quote']))}",
            "Copy the code using the guide's Copy button and paste it again, or replace curly quotes with "
            "straight quotes (\" or ').",
            line=line, col=col, kind="paste",
            python_message=f"SyntaxError: invalid character '{src.text[first]}' (U+{ord(src.text[first]):04X})")
    if found["invisible"]:
        first = found["invisible"][0]
        line, col = where(first)
        ch = src.text[first]
        name = INVISIBLE_NAMES.get(ch) or unicodedata.name(ch, "invisible character").lower()
        raise src.problem(
            "Invisible character in the code",
            f"Line {line}, column {col}, has an invisible character: {name} (U+{ord(ch):04X}). It comes from "
            "documents, emails or chats; it looks like a normal space, but Python does not accept it."
            f"{more(len(found['invisible']))}",
            "Delete the spaces on this line and type them again, or copy the code using the guide's Copy button.",
            line=line, col=col, kind="paste",
            python_message=f"SyntaxError: invalid non-printable character U+{ord(ch):04X}")
    first = found["typographic"][0]
    line, col = where(first)
    symbol, name, ascii_version = TYPOGRAPHIC[src.text[first]]
    raise src.problem(
        "Typographic character in the code",
        f"Line {line} has the character '{symbol}' ({name}), which text editors insert instead of "
        f"'{ascii_version}'. Python does not accept it.{more(len(found['typographic']))}",
        f"Replace '{symbol}' with '{ascii_version}', or copy the code using the guide's Copy button.",
        line=line, col=col, kind="paste",
        python_message=f"SyntaxError: invalid character '{symbol}' (U+{ord(src.text[first]):04X})")


# --------------------------------------------------------------------------- compile
def explain_syntax_error(err, src):
    """Turn a SyntaxError/IndentationError into a BrickProblem with an English explanation."""
    msg = err.msg or str(err)
    line = err.lineno or None
    col = err.offset or None
    raw = f"{type(err).__name__}: {msg}"
    common = {"line": line, "col": col, "python_message": raw, "kind": "syntax"}
    code_line = src.line(line) or ""
    copy_hint = "Copy again using the guide's Copy button and replace what you pasted."

    if any(ch in CURLY_QUOTES for ch in code_line):
        return src.problem(
            "Curly quotes in the code",
            f"Line {line} has curly quotes. They appear when code passes through slides, Word, PDF or "
            "some chats, and Python does not accept them.",
            "Replace curly quotes with straight quotes (\" or '), or copy the code using the guide's Copy button.",
            **common)
    if isinstance(err, TabError):
        return src.problem(
            "Mixed tabs and spaces",
            f"Line {line} mixes tabs and spaces in its indentation (the spaces at the start of the line).",
            "Use only spaces. In VS Code: Ctrl+Shift+P → \"Convert Indentation to Spaces\". Or use the guide's "
            "Copy button.", **common)
    if isinstance(err, IndentationError):
        match = re.match(r"expected an indented block after (.+?) on line (\d+)", msg)
        if match:
            start = int(match.group(2))
            if line is not None and line <= start:
                return src.problem(
                    "The end of the code was left out",
                    f"The block starting on line {start} ends without any content: it looks like the selection "
                    "stopped in the middle of the code.", copy_hint, **common)
            return src.problem(
                "Indentation problem",
                f"After line {start}, Python expected an indented block (with leading spaces), but "
                f"line {line} has no indentation. In Python, leading spaces are part of the code.",
                "This usually happens when the selection loses its leading spaces. Copy again using the "
                "Copy button, or indent the line with Tab.", **common)
        if "unexpected indent" in msg:
            return src.problem(
                "Unexpected indentation",
                f"Line {line} has extra leading spaces (unexpected indentation).",
                "The selection may have started in the middle of a line, or the code was pasted already indented. "
                "Select the lines and use Shift+Tab to remove indentation, or use the Copy button.",
                **common)
        if "unindent does not match" in msg:
            return src.problem(
                "Misaligned indentation",
                f"The indentation on line {line} does not match the lines above it.",
                "Align the line with the block it belongs to (same number of spaces), or use the "
                "Copy button.", **common)
        return src.problem("Indentation problem", f"The indentation on line {line} is incorrect.",
                           copy_hint, **common)
    match = re.match(r"'(.)' was never closed", msg)
    if match:
        return src.problem(
            "The end of the code was left out",
            f"On line {line}, a '{match.group(1)}' was opened and never closed: it looks like the selection did not "
            "include the entire block.", copy_hint, **common)
    if msg.startswith("unterminated"):
        return src.problem(
            "Unterminated quoted text",
            f"On line {line}, quoted text starts but does not end. Either the selection was incomplete, "
            "or a quote is missing.", copy_hint, **common)
    if "unexpected EOF" in msg or "incomplete input" in msg:
        return src.problem("The end of the code was left out",
                           "The file ends in the middle of a statement.", copy_hint, **common)
    if "invalid non-printable character" in msg:
        return src.problem("Invisible character in the code",
                           f"Line {line} has an invisible character that Python does not accept.",
                           "Delete the spaces on this line and type them again, or use the Copy button.",
                           **common)
    hints = {
        "expected ':'": "A colon (:) is missing at the end of the line.",
        "Perhaps you forgot a comma": "There may be a missing comma between two items.",
        "unmatched": "A parenthesis, bracket or brace is closed without having been opened.",
        "outside function": "The line is outside the function: check the indentation.",
    }
    hint = next((h for key, h in hints.items() if key in msg), None)
    return src.problem(
        "Syntax error",
        f"Python could not understand line {line}.",
        (hint + " " if hint else "") + "Compare the line with the guide's code or copy again using the Copy button.",
        **common)


def compile_brick(src):
    """Return (code object, ast tree). The filename keeps tracebacks pointing at the brick file."""
    filename = str(src.path)
    try:
        tree = compile(src.text, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        code = compile(tree, filename, "exec", dont_inherit=True)
    except SyntaxError as err:
        raise explain_syntax_error(err, src) from None
    except ValueError as err:  # e.g. "source code string cannot contain null bytes"
        raise src.problem("File with invalid characters", f"Python could not read the file: {err}.",
                          "Copy the code using the guide's Copy button and paste it again.", kind="syntax")
    return code, tree


# --------------------------------------------------------------------------- definitions (ast)
def check_definitions(src, tree):
    """Wrong-lab code is an error when the brick's own names are missing; duplicates are warnings."""
    lab = src.lab
    defs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((node.name, node.lineno, "function"))
        elif isinstance(node, ast.ClassDef):
            defs.append((node.name, node.lineno, "class"))
    names = {name for name, _, _ in defs}

    counts = Counter(name for name, _, _ in defs)
    for name, count in counts.items():
        if count > 1:
            where = [str(lineno) for n, lineno, _ in defs if n == name]
            kind = next(k for n, _, k in defs if n == name)
            src.warnings.append(
                f"The {kind} {name} appears {count} times in {src.display_path} (lines {' and '.join(where)}). "
                "It looks like the code was pasted twice; Python only uses the last copy. Delete the duplicate.")

    foreign = [(name, lineno, NAME_TO_LAB[name]) for name, lineno, _ in defs
               if name in NAME_TO_LAB and NAME_TO_LAB[name] != lab.number]
    if not foreign:
        return
    other = LABS[foreign[0][2]]
    missing_own = [name for name in lab.names if name not in names]
    if missing_own:
        raise src.problem(
            "Code in the wrong file",
            f"This code appears to belong to Lab {other.number}; paste it into {other.rel_path}.",
            f"The file {src.display_path} belongs to Lab {lab.number} and must define "
            f"{' and '.join(lab.names)}. Cut the code from here, paste it into the correct file and copy the code "
            f"for Lab {lab.number} using the guide's Copy button.",
            line=foreign[0][1], kind="wrong-lab")
    labels = ", ".join(sorted({name for name, _, _ in foreign}))
    src.warnings.append(
        f"{src.display_path} also has code from Lab {other.number} ({labels}). It belongs in "
        f"{other.rel_path}.")


# --------------------------------------------------------------------------- exec
def ensure_bricks_package():
    """Import the `bricks` package so `from bricks.b03_embeddings import ...` resolves."""
    if "bricks" in sys.modules:
        return sys.modules["bricks"]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import bricks  # noqa: F401
    except ImportError:
        package = types.ModuleType("bricks")
        package.__path__ = [str(ROOT / "bricks")]
        sys.modules["bricks"] = package
    return sys.modules["bricks"]


def exec_brick(src, code):
    """Execute the brick into a fresh module object registered under its bricks.* name."""
    package = ensure_bricks_package()
    module = types.ModuleType(src.lab.module_name)
    module.__file__ = str(src.path)
    module.__package__ = "bricks"
    sys.modules[src.lab.module_name] = module
    setattr(package, src.lab.brick, module)
    _SOURCES[str(src.path)] = src
    try:
        exec(code, module.__dict__)
    except KeyboardInterrupt:
        raise
    except ImportError as exc:
        raise explain_import_error(exc, src) from None
    except BaseException as exc:  # noqa: BLE001 - any student error must become a message
        info = describe_exception(exc)
        raise BrickProblem(
            "Error loading the file",
            f"The code in {src.display_path} raised an error when executed: {info['type']}: {info['message']}",
            info.get("hint"), kind="exec", lab=src.lab.number, path=info.get("path") or src.display_path,
            line=info.get("line"), code=info.get("code"),
            python_message=f"{info['type']}: {info['message']}") from None
    return module


def explain_import_error(exc, src):
    name = getattr(exc, "name", None) or ""
    raw = f"{type(exc).__name__}: {exc}"
    info = describe_exception(exc)
    where = {"line": info.get("line"), "code": info.get("code"), "python_message": raw,
             "lab": src.lab.number, "path": info.get("path") or src.display_path, "kind": "import"}
    if name == "torch" or name.startswith("torch."):
        return BrickProblem(
            "PyTorch not found",
            "PyTorch is not installed in this Python — return to Lab 0.",
            f"With the virtual environment activated, install: {TORCH_INSTALL_CMD} and restart the workshop "
            "(python run.py).", **where)
    match = re.search(r"cannot import name '([^']+)' from '([^']+)'", str(exc))
    if match and match.group(2).startswith("bricks."):
        missing, module_name = match.groups()
        brick = module_name.split(".", 1)[1]
        owner = next((lab for lab in LABS if lab.brick == brick), None)
        hint = (f"Check Lab {owner.number}: the file {owner.rel_path} must define {missing}."
                if owner else "Check the name in the import.")
        return BrickProblem(
            "Import not found",
            f"{src.display_path} imports {missing} from bricks/{brick}.py, but {missing} was not found there.",
            hint, **where)
    if name.startswith("bricks"):
        return BrickProblem("Import not found", f"The module {name} does not exist.",
                            "Check the file name in the import (for example, bricks.b03_embeddings).", **where)
    return BrickProblem(
        "Module not found",
        f"The code imports {name or 'a module'}, which is not installed in this Python.",
        "In this workshop, the code only needs Python and PyTorch. Compare the import with the guide's code.",
        **where)


# --------------------------------------------------------------------------- exceptions from student code
EXCEPTION_HINTS = {
    "NameError": "A name was used without being defined. Is a line missing (for example, an import) "
                 "or is there a typo?",
    "AttributeError": "Check the spelling of the name after the dot (case matters).",
    "TypeError": "Check the number, order and types of the arguments.",
    "KeyError": "The dictionary does not have this key.",
    "IndexError": "An index exceeded the size of the list or tensor.",
    "RuntimeError": "If the error mentions tensors, check the shapes on each line.",
    "ValueError": "If the error mentions tensors, check the shapes on each line.",
    "ZeroDivisionError": "A division by zero occurred.",
    "RecursionError": "A function is calling itself endlessly.",
    "NotImplementedError": "Is the forward method missing from the class?",
}


def describe_exception(exc):
    """Exception type, message and the deepest line inside a brick file (from the traceback)."""
    info = {"type": type(exc).__name__, "message": str(exc), "hint": EXCEPTION_HINTS.get(type(exc).__name__)}
    if isinstance(exc, KeyError) and exc.args:
        info["message"] = repr(exc.args[0])
    location = None
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        src = _SOURCES.get(frame.f_code.co_filename)
        if src is not None:
            location = (src, lineno)
    if location:
        src, lineno = location
        info.update(path=src.display_path, line=lineno, code=src.line(lineno), lab=src.lab.number)
    return info


# --------------------------------------------------------------------------- one brick, end to end
def load_brick(lab, directory):
    """Read, diagnose, compile, check definitions and execute one brick. Raises BrickProblem."""
    src = read_brick(lab, directory)
    diagnose_paste(src)
    code, tree = compile_brick(src)
    check_definitions(src, tree)
    module = exec_brick(src, code)
    return LoadedBrick(src, module, tree_digest(tree))


def snapshot_modules():
    names = [lab.module_name for lab in LABS]
    return {name: sys.modules.get(name) for name in names}


def restore_modules(snapshot):
    package = sys.modules.get("bricks")
    for name, module in snapshot.items():
        short = name.split(".", 1)[1]
        if module is None:
            sys.modules.pop(name, None)
            if package is not None and hasattr(package, short):
                delattr(package, short)
        else:
            sys.modules[name] = module
            if package is not None:
                setattr(package, short, module)


def install_modules(modules):
    """Make the given {lab: module} the ones that later `from bricks.x import y` statements see."""
    package = ensure_bricks_package()
    for lab, module in modules.items():
        sys.modules[LABS[lab].module_name] = module
        setattr(package, LABS[lab].brick, module)
