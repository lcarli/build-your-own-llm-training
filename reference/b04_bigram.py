import torch.nn as nn

from bricks.b03_embeddings import TokenEmbedding


class BigramLM(nn.Module):
    """Predict the next token looking only at the current token."""

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, n_embd)   # ids -> vectors (Lab 3)
        self.lm_head = nn.Linear(n_embd, vocab_size)      # vector -> one score per possible next token

    def forward(self, idx):
        x = self.embed(idx)          # (B, T, n_embd)
        logits = self.lm_head(x)     # (B, T, vocab_size): raw scores, not probabilities yet
        return logits
