import torch
import torch.nn.functional as F


def next_token_probs(logits, temperature=1.0):
    """Turn the scores of the last position into probabilities."""
    last = logits[:, -1, :]                           # (B, vocab_size): only the last position matters
    return F.softmax(last / temperature, dim=-1)      # divide by the temperature, then softmax (each row sums to 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, generator=None):
    """Append max_new_tokens new tokens to idx, one at a time."""
    for _ in range(max_new_tokens):
        logits = model(idx)                                                     # (B, T, vocab_size)
        probs = next_token_probs(logits, temperature)                           # (B, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1): draw one token
        idx = torch.cat([idx, next_id], dim=1)                                  # (B, T + 1)
    return idx
