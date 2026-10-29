"""Corpus loading and the statistics computed directly from the text (no student code involved)."""
import json
import math
from collections import Counter

from .catalog import TRAIN_FRACTION
from .textfmt import fmt_int

MIN_CHARS = 2000
DEMO_NAME = "demonstration text"


class CorpusError(Exception):
    """The corpus file exists but cannot be used (message in English)."""


def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


class Corpus:
    def __init__(self, text, name, source="", notes="", demo=False, path=None):
        text = normalize_newlines(text)
        if text.startswith("\ufeff"):
            text = text[1:]
        self.text = text
        self.name = name
        self.source = source
        self.notes = notes
        self.demo = demo
        self.path = path
        self.n_chars = len(text)
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.n_train = int(self.n_chars * TRAIN_FRACTION)
        self.n_val = self.n_chars - self.n_train
        self._counts = None
        self._floor = None

    @property
    def ln_vocab(self):
        return math.log(self.vocab_size)

    @property
    def counts(self):
        """Occurrences of each character in the whole corpus."""
        if self._counts is None:
            self._counts = Counter(self.text)
        return self._counts

    @property
    def bigram_floor(self):
        """Conditional entropy H(next | current) of the training split, in nats.

        It is the exact minimum of the average cross-entropy over the training pairs that any
        model looking only at the current character can reach, computed by counting pairs.
        """
        if self._floor is None:
            train = self.text[: self.n_train]
            pairs = Counter(zip(train, train[1:]))
            firsts = Counter(train[:-1])
            total = len(train) - 1
            h = 0.0
            for (a, _b), c in pairs.items():
                h += c * math.log(firsts[a] / c)
            self._floor = h / total if total > 0 else 0.0
        return self._floor

    def default_text(self):
        """A short phrase taken from the corpus itself (default input of the panels)."""
        line = ""
        for candidate in self.text.split("\n"):
            candidate = candidate.strip()
            if len(candidate) >= 3:
                line = candidate
                break
        if not line:
            line = self.text[:24].strip() or self.text[:24]
        if len(line) > 32:
            cut = line[:32].rsplit(" ", 1)[0]
            line = cut if len(cut) >= 8 else line[:32]
        return line

    def start_char(self):
        """First character of every gallery sample: a newline when the corpus has one."""
        for ch in ("\n", " "):
            if ch in self.counts:
                return ch
        return self.vocab[0]

    def summary(self):
        return {
            "name": self.name,
            "source": self.source,
            "notes": self.notes,
            "demo": self.demo,
            "chars": self.n_chars,
            "vocab_size": self.vocab_size,
            "train_tokens": self.n_train,
            "val_tokens": self.n_val,
            "ln_vocab": self.ln_vocab,
            "bigram_floor": self.bigram_floor,
            "default_text": self.default_text(),
            "file": self.path.name if self.path else None,
        }


def _read_text_file(path):
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_corpus(data_dir):
    """data/corpus.txt when present (see data/prepare_data.py), otherwise the small demo text."""
    corpus_path = data_dir / "corpus.txt"
    if corpus_path.exists():
        text = _read_text_file(corpus_path)
        meta = {}
        meta_path = data_dir / "corpus.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                meta = {}
        corpus = Corpus(text, meta.get("name") or "corpus.txt", meta.get("source", ""),
                        meta.get("notes", ""), demo=False, path=corpus_path)
        if corpus.n_chars < MIN_CHARS:
            raise CorpusError(
                f"The file data/corpus.txt has only {fmt_int(corpus.n_chars)} characters. "
                f"Use text with at least {fmt_int(MIN_CHARS)} characters "
                "(python data/prepare_data.py) or delete the file to use the demonstration text.")
        return corpus
    sample_path = data_dir / "sample_pt.txt"
    if not sample_path.exists():
        raise CorpusError("Neither data/corpus.txt nor data/sample_pt.txt was found. "
                          "Run: python data/prepare_data.py")
    return Corpus(_read_text_file(sample_path), DEMO_NAME, "data/sample_pt.txt",
                  "Short original text, used only when data/corpus.txt does not exist.",
                  demo=True, path=sample_path)
