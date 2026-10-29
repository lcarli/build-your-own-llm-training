import torch

from bricks.b06_loss import next_token_loss


def make_optimizer(model, learning_rate):
    """AdamW nudges every parameter a little at each step."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train_step(model, optimizer, xb, yb):
    """One learning step: predict, measure the error, compute gradients, update the weights."""
    logits = model(xb)                       # 1. predict: (B, T, vocab_size)
    loss = next_token_loss(logits, yb)       # 2. measure the error (Lab 6)
    optimizer.zero_grad(set_to_none=True)    # 3. clear the gradients of the previous step
    loss.backward()                          # 4. compute how each weight affects the loss
    optimizer.step()                         # 5. nudge every weight in the direction that lowers the loss
    return loss.item()
