class CharTokenizer:
    """Character-level tokenizer: each unique character becomes one token."""

    def __init__(self, text):
        self.vocab = sorted(set(text))                            # unique characters, sorted
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}    # character -> id
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}    # id -> character

    def encode(self, s):
        return [self.stoi[ch] for ch in s]                        # text -> list of ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)                 # list of ids -> text
