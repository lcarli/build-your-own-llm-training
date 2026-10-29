# Lab 1 — Tokenizer: text becomes numbers

*Objective: turn text into a list of numbers (IDs) and back into text.*

### Why
A language model does not read letters: it only performs calculations with numbers. So the first step is to cut text into pieces, called **tokens**, and give each one a number (an **ID**). The set of all known tokens is the **vocabulary**.

Here, each character is a token. Models such as GPT use larger pieces, such as parts of words, but the idea is the same.

### Predict
How many different characters are there in "o rato roeu a roupa do rei de roma"?
@answer **11**: space, a, d, e, i, m, o, p, r, t, u.

### Copy and paste
@code b01_tokenizer
- `sorted(set(text))` builds the vocabulary: each character exactly once, in order.
- `stoi` (*string to index*) and `itos` (*index to string*) are the two translation dictionaries: character → ID and ID → character.
- `encode` turns text into IDs; `decode` takes the reverse path.

### Verify and observe
@verify
In the panel:
- with the base text "o rato roeu a roupa do rei de roma", the vocabulary has 11 characters, "roma" becomes `[8, 6, 5, 1]`, and `decode` returns "roma";
- with the **{{corpus_name}}** corpus, the vocabulary has **{{vocab_size}}** characters. This is the tokenizer used throughout the rest of the workshop.

### Experiment
- Encode "Roma", with a capital R. You get `KeyError: 'R'`, which is expected: the tokenizer only knows characters it has seen in the base text. Many real LLMs, such as GPT-2, start from bytes and therefore never encounter an unknown character.
- Replace the base text with a sentence of your own and watch the vocabulary change.
- In the corpus table, look for the most and least frequent characters.

### Think about it
At each step, how many options will the model choose from for the next token?
@answer **{{vocab_size}}**, the vocabulary size. In Lab 4, this number becomes the last dimension of the model's output.
