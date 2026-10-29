# Lab 6 — Loss: measuring the error

*Objective: calculate a single number that tells us how wrong the model is.*

### Why
To learn, the model needs to know how wrong it was. The measure used by LLMs is **cross-entropy** (a measure of prediction error):

1. At each position, look at the probability the model assigned to the **actual** token that came next.
2. Calculate −log of that probability: this is the model's "surprise".
3. Take the average across all positions.

If the model assigned probability 1 to the correct token, the surprise is 0. If it assigned an almost zero probability, the surprise is enormous. This average is the **loss**: the lower, the better.

### Predict
What is the loss of a model that assigns the same probability to all {{vocab_size}} tokens?
@answer **ln({{vocab_size}}) ≈ {{ln_vocab}}.** Each token receives 1/{{vocab_size}}, and −log(1/{{vocab_size}}) = ln({{vocab_size}}). This is the loss of a model that "knows nothing".

### Copy and paste
@code b06_loss
- `logits.shape` returns the three dimensions: B segments, T positions, and V possible tokens.
- `.view(B * T, V)` combines all positions into a single list of predictions; `targets.view(B * T)` does the same with the correct answers.
- `F.cross_entropy` applies softmax, calculates −log of the correct token's probability, and takes the average, all at once.

### Verify and observe
@verify
In the panel:
- the model's surprise at each position of the text you typed (stronger colors indicate greater surprise) and the probability it assigned to the actual character;
- the average loss, compared with {{ln_vocab}}, the loss of a model that distributes probability equally;
- without training, loss is usually a little above {{ln_vocab}}: random scores favor some tokens for no reason.

### Experiment
- Compare ordinary text with a random sequence of letters. Before training, the two have similar loss.
- Return here after Lab 7 and repeat the comparison.

### Think about it
If the model is always correct, with probability 1, what is the loss?
@answer **0**, because −log(1) = 0. In practice, loss never reaches 0 with real text: the next character is never completely predictable.
