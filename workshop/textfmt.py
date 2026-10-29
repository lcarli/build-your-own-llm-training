"""pt-BR number formatting and display helpers for characters."""
import unicodedata


def fmt_int(n):
    """1115394 -> '1.115.394'."""
    return f"{int(n):,}".replace(",", ".")


def fmt_float(x, decimals=2):
    """4.1744 -> '4,17' (thousands with '.', decimals with ',')."""
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_bytes(n):
    """Human-readable size in pt-BR: 1115394 -> '1,1 MB'."""
    if n >= 1_000_000:
        return f"{fmt_float(n / 1_000_000, 1)} MB"
    if n >= 1_000:
        return f"{fmt_float(n / 1_000, 1)} KB"
    return f"{fmt_int(n)} bytes"


_VISIBLE = {" ": "␣", "\n": "↵", "\t": "⇥", "\r": "⏎"}


def visible_char(ch):
    """Make whitespace and control characters visible in tables and charts."""
    if ch in _VISIBLE:
        return _VISIBLE[ch]
    if len(ch) == 1 and unicodedata.category(ch)[0] in ("C", "Z"):
        return f"U+{ord(ch):04X}"
    return ch


def char_category(ch):
    """Coarse character category in English (used to color the PCA scatter)."""
    if ch in " \n\t\r":
        return "space"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        base = unicodedata.normalize("NFD", ch.lower())[0]
        if base in "aeiou":
            return "vowel"
        return "consonant"
    return "punctuation"
