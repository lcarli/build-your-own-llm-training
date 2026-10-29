"""Functional checks of each lab. Every check runs the STUDENT's code and is a named step.

A step function receives a CheckContext and returns an English message (or raises StepFailed).
Outputs that later labs need (tokenizer, data, benchmark...) go into ctx.out.
"""
import math
import platform
import sys
import time

from .catalog import BRICKS_DIR, LABS, N_EMBD, TORCH_INSTALL_CMD
from .loader import describe_exception
from .textfmt import fmt_bytes, fmt_float, fmt_int, visible_char
from .torch_support import torch_info

TEST_PHRASE = "o rato roeu a roupa do rei de roma"
CHECK_SEED = 1337


class StepFailed(Exception):
    def __init__(self, message, hint=None, status="fail"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.status = status


class StepWarning(Exception):
    """The step passed, but the student should read the message."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class CheckContext:
    """What a lab check can see: fresh modules, outputs staged by this verification, the session."""

    def __init__(self, session, modules, staged):
        self.session = session
        self.torch = session.torch
        self.modules = modules
        self.staged = staged
        self.out = {}
        self.brick_problem = None   # set when the brick itself could not be loaded (Lab 0 only)

    def need_torch(self):
        if self.torch is None:
            raise StepFailed("PyTorch is not installed in this Python — return to Lab 0.",
                             f"Install with: {TORCH_INSTALL_CMD}")
        return self.torch

    def get(self, lab, name, kind):
        module = self.modules.get(lab)
        obj = getattr(module, name, None) if module is not None else None
        if obj is None:
            path = LABS[lab].rel_path
            raise StepFailed(
                f"The {kind} {name} was not found in {path}.",
                "Check that the name is spelled exactly as in the guide (case matters) and "
                "that the code is below the marker line.")
        if kind == "class" and not isinstance(obj, type):
            raise StepFailed(f"{name} should be a class (class {name}...).")
        if kind == "function" and not callable(obj):
            raise StepFailed(f"{name} should be a function (def {name}(...)).")
        return obj

    def _staged_or_session(self, lab, key, attr):
        value = self.staged.get(lab, {}).get(key)
        return value if value is not None else getattr(self.session, attr)

    @property
    def tokenizer(self):
        return self._staged_or_session(1, "tokenizer", "tokenizer")

    @property
    def ids(self):
        return self._staged_or_session(1, "ids", "ids")

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)

    @property
    def train_data(self):
        return self._staged_or_session(2, "train", "train_data")

    def batch(self, block_size, batch_size, seed=CHECK_SEED):
        """A real batch of the training split, made with the student's get_batch (Lab 2)."""
        torch = self.need_torch()
        get_batch = self.get(2, "get_batch", "function")
        generator = torch.Generator().manual_seed(seed)
        return get_batch(self.train_data, block_size, batch_size, generator=generator)

    def fresh_model(self, seed=CHECK_SEED):
        torch = self.need_torch()
        model_class = self.get(4, "BigramLM", "class")
        torch.manual_seed(seed)
        return model_class(self.vocab_size, N_EMBD)


def shape_of(t):
    return tuple(int(s) for s in t.shape)


def show_shape(shape):
    shape = tuple(shape)
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def is_tensor_like(obj):
    return callable(getattr(obj, "dim", None)) and hasattr(obj, "shape")


# --------------------------------------------------------------------------- Lab 0
def l0_python(ctx):
    version = platform.python_version()
    if sys.version_info < (3, 10):
        raise StepFailed(f"Python {version} is too old.", "Install Python 3.10 or newer (python.org).")
    return f"Python {version}"


def l0_torch(ctx):
    info = torch_info()
    ctx.out["torch"] = info
    if not info["available"]:
        hint = f"With the virtual environment activated, install: {TORCH_INSTALL_CMD} and restart the workshop (python run.py)."
        if info.get("hint"):
            hint += " " + info["hint"]
        raise StepFailed(f"PyTorch is not installed in this Python ({info['python']}). Details: {info['error']}",
                         hint)
    threads = info["threads"]
    return f"PyTorch {info['version']} · {threads} CPU {'thread' if threads == 1 else 'threads'}"


def l0_corpus(ctx):
    corpus = ctx.session.corpus
    size = len(corpus.text.encode("utf-8"))
    message = (f"{corpus.name}: {fmt_int(corpus.n_chars)} characters ({fmt_bytes(size)}), "
               f"{corpus.vocab_size} distinct")
    if corpus.demo:
        raise StepWarning(message + ". This is the (small) demonstration text. For the workshop: "
                                    "python data/prepare_data.py")
    return message


def l0_writable(ctx):
    if ctx.session.explore:
        return "Explore mode: the files in bricks/ are not used."
    probe = BRICKS_DIR / ".write_test.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.read_text(encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise StepFailed(f"Could not write to the bricks/ folder: {exc}",
                         "Copy the workshop folder somewhere you can write files (for example, "
                         "Documents) and run python run.py again.")
    return "bricks/ is writable"


def l0_greet(ctx):
    if ctx.brick_problem is not None:
        raise StepFailed(f"{LABS[0].rel_path} cannot be used yet: {ctx.brick_problem.title.lower()} "
                         "(see the explanation above).")
    greet = ctx.get(0, "greet", "function")
    result = greet("Ana")
    if not isinstance(result, str):
        raise StepFailed(f"greet(\"Ana\") returned {type(result).__name__}, but should return text (str).",
                         "Check the return line.")
    if "Ana" not in result:
        raise StepFailed(f"greet(\"Ana\") returned \"{result}\", which does not contain \"Ana\".",
                         "Use the value of name in the text: f\"Hello, {name}! Your lab is ready.\"")
    ctx.out["greet"] = result
    return f"greet(\"Ana\") → \"{result}\""


def l0_bench(ctx):
    torch = ctx.torch
    if torch is None:
        raise StepFailed("Requires PyTorch.", status="skipped")
    n = 256
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    c = a @ b  # warm-up
    reps = 0
    start = time.perf_counter()
    while True:
        c = a @ b
        reps += 1
        elapsed = time.perf_counter() - start
        if elapsed >= 0.5:
            break
    del c
    gflops = 2 * n ** 3 * reps / elapsed / 1e9
    ctx.out["bench"] = {"gflops": gflops, "size": n, "reps": reps, "seconds": elapsed}
    return (f"≈ {fmt_float(gflops, 1)} GFLOP/s ({fmt_int(reps)} matrix multiplications of {n} × {n} "
            f"in {fmt_float(elapsed, 2)} s)")


# --------------------------------------------------------------------------- Lab 1
def l1_class(ctx):
    ctx.get(1, "CharTokenizer", "class")
    return "class CharTokenizer found"


def l1_vocab(ctx):
    tokenizer = ctx.get(1, "CharTokenizer", "class")(TEST_PHRASE)
    vocab = list(tokenizer.vocab)
    expected = sorted(set(TEST_PHRASE))
    ctx.out["test_tokenizer"] = tokenizer
    shown = ", ".join(repr(c) for c in vocab[:20])
    if vocab != expected:
        raise StepFailed(
            f"The vocabulary of \"{TEST_PHRASE}\" should have {len(expected)} sorted characters: "
            f"[{', '.join(repr(c) for c in expected)}]. Yours has {len(vocab)}: [{shown}].",
            "Build the vocabulary with sorted(set(text)): each character once, in order.")
    return f"{len(vocab)} characters: [{shown}]"


def l1_encode(ctx):
    ids = list(ctx.out["test_tokenizer"].encode("roma"))
    if ids != [8, 6, 5, 1]:
        raise StepFailed(f"encode(\"roma\") returned {ids}; expected [8, 6, 5, 1].",
                         "encode should replace each character with its ID, using the stoi dictionary.")
    return "encode(\"roma\") → [8, 6, 5, 1]"


def l1_roundtrip(ctx):
    tokenizer = ctx.out["test_tokenizer"]
    for text in ("roma", TEST_PHRASE):
        back = tokenizer.decode(tokenizer.encode(text))
        if back != text:
            raise StepFailed(f"decode(encode(\"{text}\")) returned {back!r}.",
                             "decode should replace each ID with its character (itos) and combine them with \"\".join(...).")
    return "decode(encode(text)) returns the original text"


def l1_corpus(ctx):
    corpus = ctx.session.corpus
    tokenizer = ctx.get(1, "CharTokenizer", "class")(corpus.text)
    vocab = list(tokenizer.vocab)
    if vocab != corpus.vocab:
        raise StepFailed(f"For the corpus, the vocabulary should have {corpus.vocab_size} characters, but yours has {len(vocab)}.")
    ids = list(tokenizer.encode(corpus.text))
    if len(ids) != corpus.n_chars:
        raise StepFailed(f"encode(corpus) returned {fmt_int(len(ids))} IDs for {fmt_int(corpus.n_chars)} characters.")
    head = corpus.text[:500]
    if tokenizer.decode(ids[:500]) != head:
        raise StepFailed("decode(encode(corpus)) did not return the original text.")
    ctx.out["tokenizer"] = tokenizer
    ctx.out["ids"] = ids
    return f"{corpus.name}: V = {corpus.vocab_size} distinct characters · {fmt_int(len(ids))} tokens"


# --------------------------------------------------------------------------- Lab 2
def l2_functions(ctx):
    ctx.get(2, "split_train_val", "function")
    ctx.get(2, "get_batch", "function")
    return "split_train_val and get_batch found"


def l2_split(ctx):
    torch = ctx.need_torch()
    ids = ctx.ids
    train, val = ctx.get(2, "split_train_val", "function")(ids)
    n = len(ids)
    n_train, n_val = len(train), len(val)
    if n_train + n_val != n:
        raise StepFailed(f"training ({fmt_int(n_train)}) + validation ({fmt_int(n_val)}) should total {fmt_int(n)} tokens.")
    if n_train != int(n * 0.9):
        raise StepFailed(f"Training should have int(0,9 × {fmt_int(n)}) = {fmt_int(int(n * 0.9))} tokens; it has "
                         f"{fmt_int(n_train)}.", "Use n = int(len(data) * train_fraction).")
    if train.dtype != torch.long:
        raise StepFailed("IDs must be integers.", "Use torch.tensor(ids, dtype=torch.long).")
    ctx.out["train"], ctx.out["val"] = train, val
    return f"training {fmt_int(n_train)} + validation {fmt_int(n_val)} = {fmt_int(n)} tokens"


def _l2_batch(ctx, seed=CHECK_SEED):
    torch = ctx.need_torch()
    get_batch = ctx.get(2, "get_batch", "function")
    generator = torch.Generator().manual_seed(seed)
    return get_batch(ctx.out["train"], 8, 4, generator=generator)


def l2_shapes(ctx):
    x, y = _l2_batch(ctx)
    if not (is_tensor_like(x) and is_tensor_like(y)):
        raise StepFailed("get_batch should return two tensors: x and y.")
    if shape_of(x) != (4, 8) or shape_of(y) != (4, 8):
        raise StepFailed(f"With block_size=8 and batch_size=4, x and y should have shape (4, 8); got "
                         f"{show_shape(shape_of(x))} and {show_shape(shape_of(y))}.",
                         "Use torch.stack to stack the windows.")
    ctx.out["xy"] = (x.tolist(), y.tolist())
    return "x: (4, 8) · y: (4, 8)"


def l2_shift(ctx):
    xl, yl = ctx.out["xy"]
    for row, (xr, yr) in enumerate(zip(xl, yl)):
        if yr[:-1] != xr[1:]:
            raise StepFailed(f"In batch row {row}, y[:, :-1] should equal x[:, 1:] (y is x shifted "
                             "one position).", "y starts one position after x: data[i + 1 : i + block_size + 1].")
    return "y[:, :-1] == x[:, 1:] in every row"


def l2_range(ctx):
    xl, yl = ctx.out["xy"]
    vocab_size = ctx.vocab_size
    values = [v for rows in (xl, yl) for row in rows for v in row]
    if min(values) < 0 or max(values) >= vocab_size:
        raise StepFailed(f"There are IDs outside the range [0, {vocab_size}).")
    return f"all IDs between 0 and {vocab_size - 1}"


def l2_seed(ctx):
    x2, _y2 = _l2_batch(ctx)
    if x2.tolist() != ctx.out["xy"][0]:
        raise StepFailed("With the same generator (same seed), get_batch returned different batches.",
                         "Pass generator=generator to torch.randint.")
    return "same seed → same batch"


# --------------------------------------------------------------------------- Lab 3
def l3_class(ctx):
    ctx.get(3, "TokenEmbedding", "class")
    return "class TokenEmbedding found"


def l3_module(ctx):
    torch = ctx.need_torch()
    torch.manual_seed(CHECK_SEED)
    embedding = ctx.get(3, "TokenEmbedding", "class")(ctx.vocab_size, N_EMBD)
    if not isinstance(embedding, torch.nn.Module):
        raise StepFailed("TokenEmbedding should inherit from nn.Module.", "Write class TokenEmbedding(nn.Module):")
    ctx.out["embedding"] = embedding
    return f"TokenEmbedding({ctx.vocab_size}, {N_EMBD}) is an nn.Module"


def l3_shape(ctx):
    x, _y = ctx.batch(8, 4)
    out = ctx.out["embedding"](x)
    expected = (4, 8, N_EMBD)
    if shape_of(out) != expected:
        raise StepFailed(f"The output should have shape {show_shape(expected)}; got {show_shape(shape_of(out))}.")
    return f"{show_shape(shape_of(x))} → {show_shape(shape_of(out))}"


def l3_params(ctx):
    count = sum(p.numel() for p in ctx.out["embedding"].parameters())
    expected = ctx.vocab_size * N_EMBD
    if count != expected:
        raise StepFailed(f"Expected {ctx.vocab_size} × {N_EMBD} = {fmt_int(expected)} parameters; got {fmt_int(count)}.")
    return f"{ctx.vocab_size} × {N_EMBD} = {fmt_int(count)} parameters"


# --------------------------------------------------------------------------- Lab 4
def l4_class(ctx):
    ctx.get(4, "BigramLM", "class")
    return "class BigramLM found"


def l4_shape(ctx):
    model = ctx.fresh_model()
    x, _y = ctx.batch(8, 4)
    logits = model(x)
    expected = (4, 8, ctx.vocab_size)
    if shape_of(logits) != expected:
        raise StepFailed(f"The logits should have shape {show_shape(expected)}; got {show_shape(shape_of(logits))}.")
    ctx.out["model"] = model
    return f"{show_shape(shape_of(x))} → {show_shape(expected)}"


def l4_params(ctx):
    v = ctx.vocab_size
    count = sum(p.numel() for p in ctx.out["model"].parameters())
    expected = v * N_EMBD + N_EMBD * v + v
    if count != expected:
        raise StepFailed(f"Expected {v}×{N_EMBD} + {N_EMBD}×{v} + {v} = {fmt_int(expected)} parameters; got "
                         f"{fmt_int(count)}.")
    return f"{v}×{N_EMBD} + {N_EMBD}×{v} + {v} = {fmt_int(count)} parameters"


# --------------------------------------------------------------------------- Lab 5
def l5_functions(ctx):
    ctx.get(5, "next_token_probs", "function")
    ctx.get(5, "generate", "function")
    return "next_token_probs and generate found"


def _row_max(probs):
    return [max(row) for row in probs.tolist()]


def l5_shape(ctx):
    torch = ctx.need_torch()
    generator = torch.Generator().manual_seed(CHECK_SEED)
    logits = torch.randn(2, 3, ctx.vocab_size, generator=generator)
    probs = ctx.get(5, "next_token_probs", "function")(logits, 1.0)
    expected = (2, ctx.vocab_size)
    if shape_of(probs) != expected:
        raise StepFailed(f"For logits {show_shape(shape_of(logits))}, the probabilities should have shape "
                         f"{show_shape(expected)}; got {show_shape(shape_of(probs))}.",
                         "Use only the last position: logits[:, -1, :].")
    ctx.out["logits"], ctx.out["probs"] = logits, probs
    return f"logits {show_shape(shape_of(logits))} → probabilities {show_shape(expected)}"


def l5_valid(ctx):
    for row in ctx.out["probs"].tolist():
        if min(row) < 0:
            raise StepFailed("There are negative probabilities.", "Use F.softmax(..., dim=-1).")
        if abs(sum(row) - 1.0) > 1e-5:
            raise StepFailed(f"A row sums to {fmt_float(sum(row), 6)}, but should sum to 1.", "Use dim=-1 in softmax.")
    return "all ≥ 0 and each row sums to 1"


def l5_temperature(ctx):
    next_token_probs = ctx.get(5, "next_token_probs", "function")
    logits = ctx.out["logits"]
    low = _row_max(next_token_probs(logits, 0.5))
    mid = _row_max(ctx.out["probs"])
    high = _row_max(next_token_probs(logits, 2.0))
    if not all(lo > hi for lo, hi in zip(low, high)):
        raise StepFailed("With a lower temperature, the largest probability should increase.",
                         "Divide the logits by the temperature before softmax: last / temperature.")
    return (f"largest probability: {fmt_float(low[0], 3)} (t = 0,5) > {fmt_float(mid[0], 3)} (t = 1) > "
            f"{fmt_float(high[0], 3)} (t = 2)")


def _l5_generate(ctx, seed=42, new_tokens=20):
    torch = ctx.need_torch()
    model = ctx.out.get("model")
    if model is None:
        model = ctx.out["model"] = ctx.fresh_model()
    start = ctx.ids[:5]
    idx = torch.tensor([start], dtype=torch.long)
    out = ctx.get(5, "generate", "function")(model, idx, new_tokens, temperature=1.0,
                                          generator=torch.Generator().manual_seed(seed))
    return start, out


def l5_generate_shape(ctx):
    start, out = _l5_generate(ctx)
    if shape_of(out) != (1, 25):
        raise StepFailed(f"generate with 5 IDs + 20 new ones should return shape (1, 25); got {show_shape(shape_of(out))}.",
                         "Append one token per iteration with torch.cat([idx, next_id], dim=1).")
    ids = out.tolist()[0]
    if ids[:5] != list(start):
        raise StepFailed("generate should keep the initial IDs at the start of the sequence.")
    ctx.out["generated"] = ids
    return "(1, 5) → (1, 25): the 5 initial IDs were preserved"


def l5_generate_range(ctx):
    v = ctx.vocab_size
    if any(not 0 <= i < v for i in ctx.out["generated"]):
        raise StepFailed(f"generate produced IDs outside the range [0, {v}).")
    return f"all IDs between 0 and {v - 1}"


def l5_generate_seed(ctx):
    _start, again = _l5_generate(ctx)
    if again.tolist()[0] != ctx.out["generated"]:
        raise StepFailed("With the same seed, generate should produce the same text.",
                         "Pass generator=generator to torch.multinomial.")
    text = ctx.tokenizer.decode(ctx.out["generated"][5:]).replace("\n", visible_char("\n"))
    return f"same seed → same text: \"{text}\""


# --------------------------------------------------------------------------- Lab 6
def _l6_targets(ctx):
    torch = ctx.torch
    v = ctx.vocab_size
    rows = [[(r * 3 + t) % v for t in range(3)] for r in range(2)]
    return rows, torch.tensor(rows, dtype=torch.long)


def l6_function(ctx):
    ctx.get(6, "next_token_loss", "function")
    return "next_token_loss found"


def l6_scalar(ctx):
    torch = ctx.need_torch()
    _rows, targets = _l6_targets(ctx)
    loss = ctx.get(6, "next_token_loss", "function")(torch.zeros(2, 3, ctx.vocab_size), targets)
    if not is_tensor_like(loss):
        raise StepFailed(f"next_token_loss should return a tensor; returned {type(loss).__name__}.",
                         "Return the result of F.cross_entropy directly.")
    if loss.dim() != 0:
        raise StepFailed(f"The loss should be a single number (0-d tensor); got shape {show_shape(shape_of(loss))}.",
                         "Use the mean (reduction='mean', the default for F.cross_entropy).")
    ctx.out["zero_loss"] = float(loss)
    return "loss.dim() == 0: the mean over all positions"


def l6_uniform(ctx):
    v = ctx.vocab_size
    value = ctx.out["zero_loss"]
    if not abs(value - math.log(v)) < 1e-4:
        raise StepFailed(f"With all-zero logits, the loss should be ln {v} = {fmt_float(math.log(v), 4)}; got "
                         f"{fmt_float(value, 4)}.", "Check .view(B * T, V) and F.cross_entropy.")
    return f"zero logits: loss = {fmt_float(value, 4)} = ln {v}"


def l6_confident(ctx):
    torch = ctx.torch
    v = ctx.vocab_size
    rows, targets = _l6_targets(ctx)
    logits = [[[20.0 if c == rows[r][t] else 0.0 for c in range(v)] for t in range(3)] for r in range(2)]
    loss = float(ctx.get(6, "next_token_loss", "function")(torch.tensor(logits), targets))
    if not loss < 0.01:
        raise StepFailed(f"With a huge score for the correct token, the loss should be near 0; got "
                         f"{fmt_float(loss, 4)}.", "The targets should be the IDs of the next tokens.")
    return f"huge score for the correct token: loss = {fmt_float(loss, 6)}"


# --------------------------------------------------------------------------- Lab 7
def l7_functions(ctx):
    ctx.get(7, "make_optimizer", "function")
    ctx.get(7, "train_step", "function")
    return "make_optimizer and train_step found"


def l7_optimizer(ctx):
    model = ctx.fresh_model()
    optimizer = ctx.get(7, "make_optimizer", "function")(model, 0.01)
    if not (callable(getattr(optimizer, "step", None)) and callable(getattr(optimizer, "zero_grad", None))):
        raise StepFailed("make_optimizer should return an optimizer (with step and zero_grad).",
                         "Return torch.optim.AdamW(model.parameters(), lr=learning_rate).")
    ctx.out["model"], ctx.out["optimizer"] = model, optimizer
    return f"{type(optimizer).__name__} with lr = 0,01"


def l7_float(ctx):
    xb, yb = ctx.batch(8, 32)
    ctx.out["batch"] = (xb, yb)
    first = ctx.get(7, "train_step", "function")(ctx.out["model"], ctx.out["optimizer"], xb, yb)
    if is_tensor_like(first):
        raise StepFailed("train_step returned a tensor; it should return a plain number.",
                         "End with return loss.item().")
    if not isinstance(first, float):
        raise StepFailed(f"train_step should return a float; returned {type(first).__name__}.",
                         "End with return loss.item().")
    ctx.out["losses"] = [first]
    return f"first loss = {fmt_float(first, 4)}"


def l7_decreases(ctx):
    train_step = ctx.get(7, "train_step", "function")
    xb, yb = ctx.out["batch"]
    losses = ctx.out["losses"]
    while len(losses) < 30:
        losses.append(float(train_step(ctx.out["model"], ctx.out["optimizer"], xb, yb)))
    if not losses[-1] < losses[0]:
        raise StepFailed(f"After 30 steps on the same batch, the loss did not decrease ({fmt_float(losses[0], 3)} → "
                         f"{fmt_float(losses[-1], 3)}).",
                         "Check the order: zero_grad, backward and step — and that the optimizer received model.parameters().")
    return f"30 steps on the same batch: {fmt_float(losses[0], 3)} → {fmt_float(losses[-1], 3)}"


LAB_STEPS = {
    0: [("Python 3.10 or newer", l0_python), ("PyTorch installed", l0_torch),
        ("Training text (corpus)", l0_corpus), ("bricks/ folder is writable", l0_writable),
        ("Your greet function", l0_greet), ("Computer speed", l0_bench)],
    1: [("CharTokenizer class", l1_class), ("Test phrase vocabulary", l1_vocab),
        ("encode(\"roma\")", l1_encode), ("decode(encode(text)) == text", l1_roundtrip),
        ("Corpus tokenizer", l1_corpus)],
    2: [("Lab 2 functions", l2_functions), ("Training/validation split", l2_split),
        ("Shapes of x and y", l2_shapes), ("y is x shifted one position", l2_shift),
        ("IDs within the vocabulary", l2_range), ("Same seed, same batch", l2_seed)],
    3: [("TokenEmbedding class", l3_class), ("TokenEmbedding is an nn.Module", l3_module),
        ("Output shape", l3_shape), ("Number of parameters", l3_params)],
    4: [("BigramLM class", l4_class), ("Logits shape", l4_shape), ("Number of parameters", l4_params)],
    5: [("Lab 5 functions", l5_functions), ("Probability shape", l5_shape),
        ("Valid probabilities", l5_valid), ("Temperature", l5_temperature),
        ("generate: shape and start", l5_generate_shape), ("generate: valid IDs", l5_generate_range),
        ("generate: same seed, same text", l5_generate_seed)],
    6: [("next_token_loss function", l6_function), ("Loss is a single number", l6_scalar),
        ("Zero logits give ln V", l6_uniform), ("Confident correct prediction gives loss near 0", l6_confident)],
    7: [("Lab 7 functions", l7_functions), ("make_optimizer returns an optimizer", l7_optimizer),
        ("train_step returns a number", l7_float), ("Loss decreases", l7_decreases)],
}

# In Lab 0 every step runs even when an earlier one fails (it is an environment checklist).
INDEPENDENT_LABS = {0}


def run_lab_checks(lab_number, ctx):
    """Run the named steps of one lab. Returns (group dict, outputs dict)."""
    steps = []
    ok = True
    independent = lab_number in INDEPENDENT_LABS
    for name, fn in LAB_STEPS[lab_number]:
        if not ok and not independent:
            steps.append({"name": name, "status": "skipped", "message": "not run"})
            continue
        try:
            message = fn(ctx)
            steps.append({"name": name, "status": "ok", "message": message})
        except StepWarning as warning:
            steps.append({"name": name, "status": "warn", "message": warning.message})
        except StepFailed as failure:
            steps.append({"name": name, "status": failure.status, "message": failure.message, "hint": failure.hint})
            if failure.status == "fail":
                ok = False
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - errors in student code become step messages
            info = describe_exception(exc)
            steps.append({"name": name, "status": "fail",
                          "message": f"Error running your code: {info['type']}: {info['message']}",
                          "hint": info.get("hint"), "error": info})
            ok = False
    lab = LABS[lab_number]
    group = {"lab": lab_number, "title": f"Lab {lab_number} — {lab.title}", "ok": ok, "steps": steps}
    return group, ctx.out
