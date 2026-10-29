# Lab 3 — Embeddings: each token becomes a vector

*Objective: give each token a list of numbers that the model can adjust.*

### Why
An ID is just a label: ID 8 is not "greater than" or "similar to" ID 7. The model needs numbers it can calculate with and that can change during training.

The solution is a table: each token gets a row with **{{n_embd}} numbers**, its **embedding** (a vector). Initially, the numbers are random. During training, they are adjusted to help with predictions.

These numbers are **parameters**: the values the model learns. This table has {{vocab_size}} × {{n_embd}} = **{{params_embedding}}** parameters.

### Predict
Before training, will similar letters, such as "a" and "e", have vectors close to each other?
@answer **No.** The vectors start out random. Any organization that emerges later will come from training. Return to this panel after Lab 7.

### Copy and paste
@code b03_embeddings
- `nn.Module` is the foundation of every neural network component in PyTorch; `super().__init__()` prepares that foundation.
- `nn.Embedding(vocab_size, n_embd)` creates the table: one row per token, `n_embd` numbers per row.
- `forward` is what happens when the component is used: each ID becomes its row in the table, and the shape changes from `(B, T)` to `(B, T, n_embd)`.

### Verify and observe
@verify
In the panel:
- the heatmap shows the vectors for the characters in the text you typed: one row per character, {{n_embd}} columns;
- repeated characters have identical rows because they use the same row of the table;
- the scatter plot projects the {{vocab_size}} vectors into 2 dimensions (PCA). It is only a shadow of the {{n_embd}} numbers, but it helps you see which ones are close to each other.

### Experiment
- Type a word with repeated letters, such as "arara", and find the matching rows.
- Observe the projection: are the points scattered randomly?

### Think about it
GPT-2 has a vocabulary of 50.257 tokens. With vectors of 128 numbers, how many parameters would the embedding table alone have?
@answer 50.257 × 128 = **6.432.896** parameters. This is one reason this workshop uses characters: with {{vocab_size}} tokens and vectors of {{n_embd}} numbers, our table has {{params_embedding}}.
