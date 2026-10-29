# Lab 0 — Preparation: the workshop workflow

*Objective: check that your computer is ready and practice the cycle used in every lab: read, copy, paste, and verify.*

### Why
In this workshop, you will build a language model piece by piece, like LEGO. Every lab follows the same cycle:

1. Read the explanation.
2. Click **Copy** on the code block.
3. In VS Code, open the indicated file and paste **below** the line `# ---- PASTE BELOW ----`. The file is saved automatically.
4. Return to this page and click **Verify**.
5. The visualization on the right is unlocked, always showing real numbers from **your** model.

Everything runs on your computer: no cloud, no GPU, and no paid account.

### Predict
How many operations with decimal numbers (multiplications and additions) does your computer perform per second? A thousand? A million? A billion?
@answer The verification measures this on your computer by multiplying tables of numbers. The result appears in **GFLOP/s**: billions of operations per second. Compare it with the person next to you.

### Copy and paste
This first piece of code is just a warm-up: a function that returns a greeting.
@code b00_warmup
- `def` creates a function; `name` is the value it receives.
- The text between triple quotes describes what the function does.
- `f"..."` builds a string with the value of `name` inside.

### Verify and observe
@verify
The verification checks:
- the Python version (3.10 or newer);
- whether PyTorch is installed and how many processor cores it will use;
- the training text: **{{corpus_name}}**, with {{corpus_chars}} characters;
- whether the `bricks/` folder is writable;
- your `greet` function;
- your computer's matrix multiplication speed.

### Experiment
Make deliberate mistakes to get to know the help messages:
1. In `bricks/b00_warmup.py`, delete the spaces at the beginning of the second line of code (the one starting with `"""`) and click **Verify**. The message points to the line and explains the indentation problem.
2. Undo (Ctrl+Z). Now replace the straight quotes `"` on the `return` line with curly quotes `“` and `”`, like those inserted by Word and PowerPoint. Verify again.
3. Fix the file, or click **Complete this step** to restore the reference code.

> Always copy code using the **Copy** button in this guide. Slides, PDFs, emails, and chats replace quotes and spaces with characters that Python does not accept.

### Think about it
Why does this workshop run everything on your computer instead of using a cloud service?
@answer So that every number on the screen comes from **your** model, at no cost and without depending on the internet. And to show that an LLM's mechanism fits on a laptop: in large models, what changes is mainly the scale (more data, more parameters, and much more computation).
