# Lab 2 — Training examples: one text, thousands of questions

*Objective: turn the corpus into "context → next token" pairs and separate training and validation.*

### Why
The model learns a single task: **predicting the next token**. The text itself already provides the correct answers. In a window of 8 characters, the model practices 8 predictions: given the 1st character, predict the 2nd; given the first 2, predict the 3rd; and so on, until predicting the character immediately after the window.

The text is also divided into two parts:
- **training** (90%): the model learns from it;
- **validation** (10%): the model never trains on it; it measures whether the model has learned something that applies to new text.

To train, we randomly select several text segments at once. This group of segments is a **batch**.

### Predict
A window `x` has 8 tokens. How many training examples (context → target) does it contain?
@answer **8**. The target `y` is the same window shifted one position forward: each position in `x` has its answer at the same position in `y`.

### Copy and paste
@code b02_batches
- `torch.tensor` turns the list of IDs into a **tensor**, PyTorch's structure for numbers.
- `split_train_val` splits the tensor into two parts: 90% and 10%.
- `get_batch` randomly selects `batch_size` starting positions and extracts two windows from each: `x`, the context, and `y`, the same window one position ahead.
- `torch.stack` stacks the windows into a table of shape `(B, T)`: B segments with T tokens each.

### Verify and observe
@verify
In the panel:
- training has **{{train_tokens}}** tokens and validation has **{{val_tokens}}**;
- a batch with {{batch_size_demo}} segments of {{block_size_demo}} tokens: compare `x` and `y` and find the one-position shift;
- the {{block_size_demo}} "context → target" pairs from the first row.

### Experiment
- Click **New batch** to randomly select other segments.
- Change the window size (*block size*) and count the pairs again.

### Think about it
Why set aside 10% of the text that the model never sees during training?
@answer Because a model can **memorize** the training text. The validation loss shows whether it has learned patterns that work on new text. The two curves appear side by side in Lab 7.
