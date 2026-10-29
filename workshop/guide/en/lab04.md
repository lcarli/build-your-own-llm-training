# Lab 4 — Logits: the first prediction

*Objective: assemble the first complete model, which gives a score to every possible next token.*

### Why
Each token already has a vector. Now we need to turn that vector into **a score for each of the {{vocab_size}} tokens** that could come next. These scores are called **logits**: the higher the score, the more the model favors that token.

A linear layer (`nn.Linear`) does this: {{n_embd}} numbers go in, {{vocab_size}} scores come out.

Combining the embedding (Lab 3) and the linear layer gives us a **bigram** model: it predicts the next token by looking **only at the current token**.

### Predict
Can a logit be negative?
@answer **Yes.** A logit is a score, not a probability. In Lab 5, softmax turns the scores into probabilities.

### Copy and paste
@code b04_bigram
- `from bricks.b03_embeddings import TokenEmbedding` reuses the component from Lab 3: LEGO in action.
- `self.lm_head` is the linear layer that turns each vector into {{vocab_size}} scores.
- In `forward`, IDs become vectors and vectors become logits: `(B, T)` → `(B, T, {{n_embd}})` → `(B, T, {{vocab_size}})`.

### Verify and observe
@verify
In the panel:
- the sequence of shapes: `(1, T)` → `(1, T, {{n_embd}})` → `(1, T, {{vocab_size}})`;
- the scores for the character that would follow the last character you typed;
- the {{vocab_size}} × {{vocab_size}} table: each row is a current character, each column is a possible next character. Without training, the pattern is random.

The entire model has **{{params_bigram}}** parameters.

### Experiment
- Change the last character of the text and watch the scores change.
- Find the row for the character you typed in the table.

### Think about it
What can this model **not** know about the text?
@answer Everything that came **before** the current character. After "qu" or a standalone "u", it gives exactly the same scores. Solving this is the role of attention, starting in Lab 8.
