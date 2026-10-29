import torch.nn.functional as F


def next_token_loss(logits, targets):
    """Average surprise of the model at the real next tokens (cross-entropy)."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))   # flatten batch and time
