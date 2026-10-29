"""Read workshop/guide/<language>/labNN.md and fill in {{variables}} from the corpus."""
import re

from .catalog import (BLOCK_SIZE, DEFAULT_GUIDE_LANGUAGE, GUIDE_DIR, GUIDE_LANGUAGES,
                      LABS, N_EMBD, REFERENCE_DIR)
from .textfmt import fmt_float, fmt_int

VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
CODE_DIRECTIVE = re.compile(r"^@code\s+(\S+)\s*$", re.MULTILINE)
_warned = set()


def template_variables(corpus):
    """Values shown in the guides, formatted for pt-BR. All come from the corpus or the engine."""
    v = corpus.vocab_size
    return {
        "corpus_name": corpus.name,
        "corpus_chars": fmt_int(corpus.n_chars),
        "vocab_size": fmt_int(v),
        "ln_vocab": fmt_float(corpus.ln_vocab, 2),
        "train_tokens": fmt_int(corpus.n_train),
        "val_tokens": fmt_int(corpus.n_val),
        "n_embd": fmt_int(N_EMBD),
        "params_embedding": fmt_int(v * N_EMBD),
        "params_bigram": fmt_int(v * N_EMBD + N_EMBD * v + v),
        "bigram_floor": fmt_float(corpus.bigram_floor, 2),
        "block_size_demo": fmt_int(BLOCK_SIZE),
        "batch_size_demo": fmt_int(4),
    }


def substitute(text, variables, source="guia", log=print):
    """Replace {{name}}; unknown names stay visible and are reported once in the console."""
    def replace(match):
        name = match.group(1)
        if name in variables:
            return variables[name]
        key = (source, name)
        if key not in _warned:
            _warned.add(key)
            log(f"[warning] unknown variable {{{{{name}}}}} in {source}")
        return match.group(0)
    return VARIABLE.sub(replace, text)


def reference_code(brick):
    """The reference code of a brick (exactly what the guide's code card shows)."""
    if not re.fullmatch(r"b\d\d_[a-z0-9_]+", brick or ""):
        return None
    path = REFERENCE_DIR / f"{brick}.py"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def fallback_markdown(n, language=DEFAULT_GUIDE_LANGUAGE):
    lab = LABS[n]
    return (f"# Lab {n} — {lab.title}\n\n"
            f"> This guide's text has not been installed yet (workshop/guide/{language}/lab{n:02d}.md).\n\n"
            f"### Copy and paste\n@code {lab.brick}\n\n### Verify and observe\n@verify\n")


def guide_payload(n, corpus, log=print, language=DEFAULT_GUIDE_LANGUAGE):
    if language not in GUIDE_LANGUAGES:
        raise ValueError(f"Unsupported guide language: {language!r}. Choose pt, en, or fr.")
    lab = LABS[n]
    path = GUIDE_DIR / language / f"lab{n:02d}.md"
    fallback = not path.exists()
    if fallback:
        log(f"[warning] guide not found: {path}")
    text = fallback_markdown(n, language) if fallback else path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")
    markdown = substitute(text, template_variables(corpus), source=f"{language}/{path.name}", log=log)
    codes = {}
    for brick in CODE_DIRECTIVE.findall(markdown):
        code = reference_code(brick)
        if code is not None:
            codes[brick] = {"code": code, "path": f"bricks/{brick}.py"}
    return {"lab": n, "language": language, "title": lab.title, "markdown": markdown, "codes": codes, "fallback": fallback,
            "brick": lab.brick, "path": lab.rel_path}
