import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Turn each token id into a learnable vector of n_embd numbers."""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.table = nn.Embedding(vocab_size, n_embd)   # one row of n_embd numbers per token

    def forward(self, idx):
        return self.table(idx)                          # (B, T) ids -> (B, T, n_embd) vectors
