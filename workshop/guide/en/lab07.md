# Lab 7 — Training: the model learns

*Objective: repeat the cycle that adjusts parameters to reduce loss thousands of times.*

### Why
So far, the model's {{params_bigram}} parameters are random. Training means gradually adjusting them so that loss decreases. Each **training step** does five things:

1. **Predict:** the model calculates logits for a batch.
2. **Measure the error:** calculate loss (Lab 6).
3. **Clear:** zero out the gradients from the previous step.
4. **Calculate gradients** (`backward`): for each parameter, find which direction to move it to reduce loss. PyTorch performs this calculation automatically.
5. **Update:** the optimizer (AdamW) moves each parameter a little in that direction.

The size of that "little" is the **learning rate** (the size of the learning steps).

### Predict
Does a learning rate 10 times larger make the model learn 10 times faster?
@answer **Not always.** Large steps can overshoot and cause loss to oscillate; steps that are too small make learning slow. Test it in the panel.

### Copy and paste
@code b07_training
- `make_optimizer` creates AdamW, which will adjust all the model's parameters.
- `train_step` is one complete step: predict, measure the error, clear, calculate gradients, and update.
- `loss.item()` returns loss as an ordinary number for plotting the graph.

### Verify and observe
@verify
After verification, click **Start** in the panel and follow:
- the **training** and **validation** loss curves, updated every 250 steps;
- the dashed line: the **bigram limit**, {{bigram_floor}}. This is the lowest possible training loss for a model that looks only at the current character, calculated by counting character pairs in the training text. Our model may stay a little above it;
- the **evolution gallery**: the same text prompt, with the same seed, at various points during training;
- the {{vocab_size}} × {{vocab_size}} table gaining structure.

### Experiment
- Train with learning rates of 0,1, 0,01, and 0,001 and compare the curves. Between training runs, click **Reset model**: otherwise, **Start** keeps training the same model. Previous curves remain on the graph as thin lines.
- After training, return to Labs 3, 4, 5, and 6: the panels now show the trained model. Have the vectors become organized? Do the probabilities make sense? Has loss decreased?

### Think about it
Loss has stopped near the bigram limit. What is missing for the model to write better?
@answer **Context.** When looking at just one character, no parameter adjustment can go beyond this limit. The next step is to let each token look at the previous ones: **self-attention**, starting in Lab 8.
