import torch


def split_train_val(ids, train_fraction=0.9):
    """Split the token ids into a training part and a validation part."""
    data = torch.tensor(ids, dtype=torch.long)       # all ids as one long tensor
    n = int(len(data) * train_fraction)              # first 90% for training
    return data[:n], data[n:]                        # (train, val)


def get_batch(data, block_size, batch_size, generator=None):
    """Pick random windows: x is the context, y is the same window shifted by one."""
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])          # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])  # (B, T)
    return x, y
