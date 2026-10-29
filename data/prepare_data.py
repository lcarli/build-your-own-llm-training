"""Prepare the training text of the workshop: writes data/corpus.txt and data/corpus.json.

Examples:
    python data/prepare_data.py                         # Portuguese: Dom Casmurro (default)
    python data/prepare_data.py --lang fr               # French: Maria Chapdelaine
    python data/prepare_data.py --lang en               # English: Anne of Green Gables
    python data/prepare_data.py --corpus verne          # one specific text from the catalog
    python data/prepare_data.py --corpus hemon verne    # several texts, joined into one corpus
    python data/prepare_data.py --list                  # show the catalog
    python data/prepare_data.py --file my_text.txt      # your own UTF-8 file
    python data/prepare_data.py --url https://example.com/text.txt
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MIN_CHARS = 2000
GUTENBERG_TXT = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"

LANGUAGES = {"pt": "Portuguese", "fr": "French", "en": "English"}

# Public-domain texts (authors who died more than 80 years ago). Gutenberg novels are
# unwrapped (one paragraph per line); Tiny Shakespeare keeps its verse and dialogue lines.
CATALOG = {
    "machado": {
        "lang": "pt", "title": "Dom Casmurro", "author": "Machado de Assis", "year": "1899",
        "gutenberg": 55752,
        "about": "Brazilian novel. Uses the spelling of the 1899 edition (for example, \"elle\" and \"cançado\").",
    },
    "hemon": {
        "lang": "fr", "title": "Maria Chapdelaine", "author": "Louis Hémon", "year": "1913",
        "gutenberg": 13525,
        "about": "A novel set in rural Quebec, a classic of Quebec literature.",
    },
    "verne": {
        "lang": "fr", "title": "Vingt mille lieues sous les mers", "author": "Jules Verne", "year": "1869-1870",
        "gutenberg": 5097,
        "about": "Classic French adventure. A longer text: useful for combining with another French book.",
    },
    "montgomery": {
        "lang": "en", "title": "Anne of Green Gables", "author": "L. M. Montgomery", "year": "1908",
        "gutenberg": 45,
        "about": "An easy-to-read Canadian classic set on Prince Edward Island.",
    },
    "shakespeare": {
        "lang": "en", "title": "Tiny Shakespeare", "author": "William Shakespeare", "year": None,
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "unwrap": False,
        "about": "Excerpts from plays compiled by Andrej Karpathy (char-rnn). The nanoGPT text: "
                 "allows comparisons with published results.",
    },
}
DEFAULT_BY_LANG = {"pt": "machado", "fr": "hemon", "en": "montgomery"}

# "*** START OF THE PROJECT GUTENBERG EBOOK DOM CASMURRO ***" (also THIS / E-BOOK variants)
START_MARKER = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK\b.*$",
                          re.IGNORECASE | re.MULTILINE)
END_MARKER = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK\b.*$",
                        re.IGNORECASE | re.MULTILINE)

# Production notes that some Gutenberg files keep right after the START marker.
CREDIT_HINTS = re.compile(r"produced by|this text was adapted|thank you to|transcriber|e-?text prepared|"
                          r"distributed proofread|https?://|www\.", re.IGNORECASE)
ILLUSTRATION = re.compile(r"^\[(?:illustration|ilustração|gravure)[^\]]*\]$", re.IGNORECASE)

# Typographic spaces (French puts them before ; : ! ? and inside « ») become plain spaces;
# invisible characters (zero-width, soft hyphen, BOM) are removed. Both would be confusing tokens.
SPACE_LIKE = {ord(c): " " for c in "\u00a0\u202f\u2007\u2009\u200a\u3000"}
INVISIBLE = {ord(c): None for c in "\u200b\u200c\u200d\u2060\ufeff\u00ad"}


def decode_bytes(raw):
    """UTF-8 first (with or without BOM); Latin-1 as a fallback for old files."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def format_int(n):
    """pt-BR thousands separator: 1115394 -> '1.115.394'."""
    return f"{n:,}".replace(",", ".")


def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_spaces(text):
    return text.translate(SPACE_LIKE).translate(INVISIBLE)


def strip_gutenberg(text):
    """Keep only the book: the text between the START and END markers of Project Gutenberg."""
    start = START_MARKER.search(text)
    if not start:
        return text, False
    end = END_MARKER.search(text, start.end())
    body = text[start.end(): end.start() if end else len(text)]
    return body.strip("\n").strip() + "\n", True


def drop_leading_credits(text, max_paragraphs=6):
    """Remove the first paragraphs when they are production credits, not part of the book."""
    paragraphs = re.split(r"\n[ \t]*\n", text.strip("\n"))
    removed = 0
    while removed < max_paragraphs and len(paragraphs) > 1 and CREDIT_HINTS.search(paragraphs[0]):
        paragraphs.pop(0)
        removed += 1
    return "\n\n".join(paragraphs) + "\n", removed


def unwrap_paragraphs(text):
    """Join the hard-wrapped lines of each paragraph: one paragraph per line, a blank line between them."""
    paragraphs = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        paragraph = re.sub(r" {2,}", " ", " ".join(lines))
        if ILLUSTRATION.match(paragraph):
            continue
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs) + "\n"


def clean_text(text, unwrap):
    """Same cleaning for every source: newlines, typographic spaces, trailing spaces, blank lines."""
    text = normalize_spaces(normalize_newlines(text))
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    if unwrap:
        return unwrap_paragraphs(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"


def source_url(key):
    spec = CATALOG[key]
    return spec.get("url") or GUTENBERG_TXT.format(n=spec["gutenberg"])


def describe(key):
    """One sentence for data/corpus.json: what the text is and where it comes from."""
    spec = CATALOG[key]
    year = f" ({spec['year']})" if spec.get("year") else ""
    if "gutenberg" in spec:
        origin = (f"Project Gutenberg, eBook #{spec['gutenberg']}, public domain in the USA; "
                  "Project Gutenberg header and license removed.")
    else:
        origin = "Public domain."
    return f"{spec['title']}, by {spec['author']}{year}, in {LANGUAGES[spec['lang']]}. {spec['about']} {origin}"


def download(url):
    print(f"Downloading {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "build-your-own-llm/1.0 (workshop)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_catalog_text(key, unwrap=None):
    """Download one catalog text and clean it. unwrap=None uses the catalog default."""
    spec = CATALOG[key]
    text = normalize_newlines(decode_bytes(download(source_url(key))))
    if "gutenberg" in spec:
        text, found = strip_gutenberg(text)
        if found:
            text, _ = drop_leading_credits(text)
        else:
            print("Warning: Project Gutenberg START/END markers were not found; the entire text was used.")
    return clean_text(text, spec.get("unwrap", True) if unwrap is None else unwrap)


def write_corpus(text, name, source, notes, lang=None):
    text = normalize_newlines(text)
    if len(text) < MIN_CHARS:
        print(f"Error: the text has only {format_int(len(text))} characters. The workshop requires at least "
              f"{format_int(MIN_CHARS)}.")
        return False
    corpus_path = DATA_DIR / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    meta = {"name": name, "source": source, "notes": notes}
    if lang:
        meta["lang"] = lang
    with open(DATA_DIR / "corpus.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Done: data/corpus.txt with {format_int(len(text))} characters ({len(set(text))} unique) — {name}.")
    print("If the workshop is running, stop it (Ctrl+C) and run python run.py again.")
    return True


def print_catalog():
    print("Available texts (public domain):\n")
    print(f"  {'id':<14}{'language':<11}title — author")
    for key, spec in CATALOG.items():
        mark = " *" if DEFAULT_BY_LANG.get(spec["lang"]) == key else ""
        year = f" ({spec['year']})" if spec.get("year") else ""
        print(f"  {key + mark:<14}{LANGUAGES[spec['lang']]:<11}{spec['title']} — {spec['author']}{year}")
        print(f"  {'':<25}{spec['about']}")
    print("\n* = default text for the language. Use --lang pt|fr|en or --corpus <id> [<id> ...].")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare the workshop training text (data/corpus.txt).",
        epilog="Without options, use the default Portuguese text (Dom Casmurro). Show all texts with --list.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--lang", choices=sorted(LANGUAGES),
                        help="language: pt (Dom Casmurro), fr (Maria Chapdelaine), or en (Anne of Green Gables)")
    source.add_argument("--corpus", nargs="+", choices=list(CATALOG), metavar="ID",
                        help="one or more catalog texts (see --list); multiple texts become one corpus")
    source.add_argument("--file", type=Path, help="use your own text file (UTF-8)")
    source.add_argument("--url", help="download a text from the internet (.txt)")
    source.add_argument("--list", action="store_true", help="show available texts and exit")
    parser.add_argument("--name", help="corpus name displayed in the workshop")
    lines = parser.add_mutually_exclusive_group()
    lines.add_argument("--unwrap", dest="unwrap", action="store_true", default=None,
                       help="join lines within each paragraph (default for catalog novels)")
    lines.add_argument("--keep-lines", dest="unwrap", action="store_false", default=None,
                       help="keep original line breaks (default for Tiny Shakespeare, --file, and --url)")
    parser.set_defaults(unwrap=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print_catalog()
        return 0
    try:
        if args.file or args.url:
            if args.file and not args.file.exists():
                print(f"Error: file not found: {args.file}.")
                return 1
            raw = args.file.read_bytes() if args.file else download(args.url)
            text, stripped = strip_gutenberg(normalize_newlines(decode_bytes(raw)))
            if stripped:
                text, _ = drop_leading_credits(text)
            text = clean_text(text, bool(args.unwrap))
            if args.file:
                name = args.name or args.file.stem
                notes, source = "File provided by the instructor.", str(args.file.name)
            else:
                name = args.name or (args.url.rstrip("/").rsplit("/", 1)[-1] or "corpus")
                notes, source = "Text downloaded by the instructor.", args.url
            if stripped:
                notes += " Project Gutenberg header removed."
            ok = write_corpus(text, name, source, notes)
        else:
            keys = args.corpus or [DEFAULT_BY_LANG[args.lang or "pt"]]
            keys = list(dict.fromkeys(keys))  # ignore repeated ids
            parts = [fetch_catalog_text(key, args.unwrap).strip("\n") for key in keys]
            langs = {CATALOG[key]["lang"] for key in keys}
            ok = write_corpus("\n\n".join(parts) + "\n",
                              args.name or " + ".join(CATALOG[key]["title"] for key in keys),
                              " ; ".join(source_url(key) for key in keys),
                              " ".join(describe(key) for key in keys),
                              langs.pop() if len(langs) == 1 else "misto")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Error downloading or reading the text: {exc}")
        print("No internet? Use a local file: python data/prepare_data.py --file path/to/text.txt")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
