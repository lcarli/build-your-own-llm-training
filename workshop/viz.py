"""Visualization builders: the JSON behind each lab panel, computed now with the student's code.

Every number returned here comes from the student's modules, the session model or a direct
computation on the corpus. Each payload carries what the UI needs for its provenance caption
(training step, model seed, shapes).
"""
import math
import platform
from collections import Counter

from .catalog import MODEL_SEED, N_EMBD
from .checks import TEST_PHRASE
from .loader import describe_exception
from .textfmt import char_category, visible_char
from .torch_support import torch_info

MAX_TEXT = 64


class VizError(Exception):
    """A panel cannot be built (message in English). `info` describes an error in student code."""

    def __init__(self, message, status=422, info=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.info = info


def _student_error(exc, what):
    info = describe_exception(exc)
    return VizError(f"Error running {what}: {info['type']}: {info['message']}", 422, info)


def _text(params, key, default, max_len=MAX_TEXT):
    value = params.get(key)
    if value is None:
        value = default
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return value[:max_len]


def _float(params, key, default, low, high):
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(high, max(low, value))


def _int(params, key, default, low, high):
    try:
        value = int(float(params.get(key, default)))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def _char(s, token_id):
    """Character of one token id, decoded with the student's tokenizer."""
    cache = s.cache.setdefault("chars", {})
    if token_id not in cache:
        cache[token_id] = s.tokenizer.decode([token_id])
    return cache[token_id]


def _token(s, token_id, **extra):
    ch = _char(s, token_id)
    return {"id": int(token_id), "ch": ch, "show": "".join(visible_char(c) for c in ch), **extra}


def _encode(s, text):
    """Encode with the student's tokenizer; unknown characters become a teachable message."""
    try:
        return [int(i) for i in s.tokenizer.encode(text)], None
    except KeyError as exc:
        ch = exc.args[0] if exc.args else "?"
        shown = visible_char(ch) if isinstance(ch, str) and len(ch) == 1 else str(ch)
        return None, {"kind": "unknown_char", "char": shown, "type": "KeyError", "raw": f"KeyError: {ch!r}",
                      "message": f"The character '{shown}' is not in the corpus vocabulary "
                                 f"({len(s.tokenizer.vocab)} characters): the tokenizer has never seen it."}
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "tokenizer.encode") from None


def _labels(s):
    return [_token(s, i)["show"] for i in range(len(s.tokenizer.vocab))]


def _provenance(s):
    return {"step": s.model_step, "seed": s.model_seed, "trained": s.model_step > 0}


def _model_ready(s):
    if s.model is None:
        raise VizError("The model does not exist yet: verify Lab 4.", 409)
    return s.model


# --------------------------------------------------------------------------- Lab 0
def viz_lab0(s, params):
    lab0 = s.lab0 or {}
    labs, future = s.labs_json()
    return {"lab": 0, "steps": lab0.get("steps"), "bench": lab0.get("bench"), "greet": lab0.get("greet"),
            "torch": torch_info(), "python": platform.python_version(), "corpus": s.corpus.summary(),
            "labs": labs, "future_labs": future}


# --------------------------------------------------------------------------- Lab 1
def _id_counts(s):
    key = ("counts", id(s.ids))
    if key not in s.cache:
        s.cache[key] = Counter(s.ids)
    return s.cache[key]


def viz_lab1(s, params):
    base = _text(params, "base", TEST_PHRASE, 200)
    text = _text(params, "text", "roma", 200)
    tokenizer_class = s.modules[1].CharTokenizer
    mini = {"base": base, "text": text}
    try:
        tokenizer = tokenizer_class(base)
        mini["vocab"] = [{"ch": c, "show": visible_char(c), "id": int(tokenizer.encode(c)[0])}
                         for c in tokenizer.vocab]
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "CharTokenizer(base text)") from None
    try:
        ids = [int(i) for i in tokenizer.encode(text)]
        mini["ids"] = ids
        mini["pieces"] = [{"ch": c, "show": visible_char(c), "id": i} for c, i in zip(text, ids)]
        mini["decoded"] = tokenizer.decode(ids)
    except KeyError as exc:
        ch = exc.args[0] if exc.args else "?"
        shown = visible_char(ch) if isinstance(ch, str) and len(ch) == 1 else str(ch)
        info = describe_exception(exc)
        mini["encode_error"] = {
            "type": "KeyError", "raw": f"KeyError: {ch!r}", "char": shown,
            "message": f"The tokenizer has never seen the character '{shown}': it is not in the base text, "
                       "so it has no ID.",
            "where": info}
    except Exception as exc:  # noqa: BLE001
        info = describe_exception(exc)
        mini["encode_error"] = {"type": info["type"], "raw": f"{info['type']}: {info['message']}",
                                "message": f"Error in encode: {info['type']}: {info['message']}", "where": info}

    counts = _id_counts(s)
    table = []
    for token_id in range(len(s.tokenizer.vocab)):
        table.append(_token(s, token_id, count=int(counts.get(token_id, 0))))
    return {"lab": 1, "mini": mini,
            "corpus": {"name": s.corpus.name, "vocab_size": len(s.tokenizer.vocab), "tokens": len(s.ids),
                       "table": table}}


# --------------------------------------------------------------------------- Lab 2
def viz_lab2(s, params):
    torch = s.torch
    block_size = _int(params, "block_size", 8, 4, 16)
    seed = _int(params, "seed", MODEL_SEED, 0, 2 ** 31 - 1)
    batch_size = 4
    generator = torch.Generator().manual_seed(seed)
    try:
        x, y = s.modules[2].get_batch(s.train_data, block_size, batch_size, generator=generator)
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "get_batch") from None
    xl, yl = x.tolist(), y.tolist()
    tokens = {}
    for row in xl + yl:
        for token_id in row:
            if token_id not in tokens:
                tokens[token_id] = _token(s, token_id)
    pairs = []
    for t in range(block_size):
        context = xl[0][: t + 1]
        pairs.append({"context": context, "context_text": s.tokenizer.decode(context), "target": yl[0][t]})
    return {"lab": 2, "B": batch_size, "T": block_size, "seed": seed, "x": xl, "y": yl,
            "tokens": {str(k): v for k, v in tokens.items()}, "pairs": pairs,
            "shape_x": list(x.shape), "shape_y": list(y.shape),
            "train_size": len(s.train_data), "val_size": len(s.val_data), "total": len(s.ids)}


# --------------------------------------------------------------------------- Lab 3
def viz_lab3(s, params):
    torch = s.torch
    text = _text(params, "text", s.corpus.default_text(), 40)
    vocab_size = len(s.tokenizer.vocab)
    result = {"lab": 3, "text": text, "n_embd": N_EMBD, "vocab_size": vocab_size}
    use_model = s.model is not None and 4 in s.done and hasattr(s.model, "embed")
    if use_model:
        result.update(source="model.embed", **_provenance(s))
    else:
        result.update(source=f"TokenEmbedding({vocab_size}, {N_EMBD})", step=None, seed=MODEL_SEED, trained=False)
        if s.model is not None:
            result["note"] = "Your BigramLM has no embed attribute; the panel uses a new TokenEmbedding."
    ids, error = _encode(s, text)
    try:
        with s.model_lock, torch.no_grad():
            if use_model:
                embedding = s.model.embed
                result["step"] = s.model_step
            else:
                torch.manual_seed(MODEL_SEED)
                embedding = s.modules[3].TokenEmbedding(vocab_size, N_EMBD)
            table = embedding(torch.arange(vocab_size).view(1, vocab_size))[0]       # (V, n_embd)
            centered = table - table.mean(dim=0)
            u, sv, _vh = torch.linalg.svd(centered, full_matrices=False)
            coords = (u[:, :2] * sv[:2]).tolist()
            singular = sv.tolist()
            params_count = int(sum(p.numel() for p in embedding.parameters()))
            vectors = None
            if ids:
                out = embedding(torch.tensor([ids], dtype=torch.long))               # (1, T, n_embd)
                result["shape"] = list(out.shape)
                vectors = out[0].tolist()
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "TokenEmbedding") from None
    total = sum(v * v for v in singular)
    result["params"] = params_count
    result["pca"] = {
        "points": [_token(s, i, x=coords[i][0], y=coords[i][1], cat=char_category(_char(s, i)))
                   for i in range(vocab_size)],
        "explained": [(v * v / total) if total > 0 else 0.0 for v in singular[:2]],
    }
    if error:
        result["input_error"] = error
    elif not ids:
        result["input_error"] = {"kind": "empty", "message": "Enter at least one character."}
    else:
        result["ids"] = ids
        result["tokens"] = [_token(s, i) for i in ids]
        result["heat"] = [[round(v, 4) for v in row] for row in vectors]
    return result


# --------------------------------------------------------------------------- Lab 4
def _matrix(s, mode):
    torch = s.torch
    import torch.nn.functional as F
    model = _model_ready(s)
    vocab_size = len(s.tokenizer.vocab)
    try:
        with s.model_lock, torch.no_grad():
            ids = torch.arange(vocab_size)
            if hasattr(model, "embed") and hasattr(model, "lm_head"):
                logits = model.lm_head(model.embed(ids))                         # (V, V)
                source = f"model.lm_head(model.embed(arange({vocab_size})))"
            else:
                logits = model(ids.view(1, vocab_size))[0]
                source = f"model(arange({vocab_size}).view(1, {vocab_size}))[0]"
            values = F.softmax(logits, dim=-1) if mode == "probs" else logits
            if mode == "probs":
                source = f"softmax({source}, dim=-1)"
            matrix = values.tolist()
            shape = list(values.shape)
            step = s.model_step
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "the model") from None
    digits = 4 if mode == "probs" else 3
    return {"mode": mode, "matrix": [[round(v, digits) for v in row] for row in matrix], "labels": _labels(s),
            "shape": shape, "source": source, "step": step, "seed": s.model_seed, "trained": step > 0}


def viz_lab4(s, params):
    torch = s.torch
    mode = "probs" if params.get("mode") == "probs" else "logits"
    result = {"lab": 4, "matrix": _matrix(s, mode), **_provenance(s)}
    if params.get("part") == "matrix":
        return result
    model = _model_ready(s)
    text = _text(params, "text", s.corpus.default_text())
    result["text"] = text
    ids, error = _encode(s, text)
    if error or not ids:
        result["input_error"] = error or {"kind": "empty", "message": "Enter at least one character."}
        return result
    try:
        with s.model_lock, torch.no_grad():
            idx = torch.tensor([ids], dtype=torch.long)
            shapes = [{"label": "ids", "shape": list(idx.shape)}]
            if hasattr(model, "embed"):
                shapes.append({"label": "model.embed(ids)", "shape": list(model.embed(idx).shape)})
            logits = model(idx)
            shapes.append({"label": "model(ids)", "shape": list(logits.shape)})
            last = logits[0, -1].tolist()
            result["step"] = s.model_step
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "the model") from None
    order = sorted(range(len(last)), key=lambda i: -last[i])
    result.update(
        ids=ids, shapes=shapes, last=_token(s, ids[-1]),
        top=[_token(s, i, value=last[i]) for i in order[:15]],
        others={"count": max(0, len(last) - 15), "min": min(last), "max": max(last)},
        params=s.model_params())
    return result


# --------------------------------------------------------------------------- Lab 5
def viz_lab5(s, params):
    torch = s.torch
    model = _model_ready(s)
    next_token_probs = s.modules[5].next_token_probs
    temperature = _float(params, "temperature", 1.0, 0.05, 3.0)
    text = _text(params, "text", s.corpus.default_text())
    vocab_size = len(s.tokenizer.vocab)
    result = {"lab": 5, "text": text, "temperature": temperature, **_provenance(s)}
    ids, error = _encode(s, text)
    if error or not ids:
        result["input_error"] = error or {"kind": "empty", "message": "Enter at least one character."}
        return result

    if params.get("part") == "generate":
        seed = _int(params, "seed", 42, 0, 2 ** 31 - 1)
        new_tokens = 200
        result.update(part="generate", seed=seed, new_tokens=new_tokens)
        try:
            with s.model_lock:
                idx = torch.tensor([ids], dtype=torch.long)
                out = s.modules[5].generate(model, idx, new_tokens, temperature=temperature,
                                            generator=torch.Generator().manual_seed(seed))
                result["step"] = s.model_step
            out_ids = out.tolist()[0]
            generated = out_ids[len(ids):]
            rows = []
            with s.model_lock, torch.no_grad():
                for i in range(min(20, len(generated))):
                    prefix = torch.tensor([out_ids[: len(ids) + i]], dtype=torch.long)
                    probs = next_token_probs(model(prefix), temperature)[0].tolist()
                    chosen = generated[i]
                    order = sorted(range(vocab_size), key=lambda j: -probs[j])
                    rows.append({"i": i + 1,
                                 "context": s.tokenizer.decode(out_ids[max(0, len(ids) + i - 10): len(ids) + i]),
                                 "chosen": _token(s, chosen, p=probs[chosen], rank=order.index(chosen) + 1),
                                 "top": [_token(s, j, p=probs[j]) for j in order[:5]]})
        except Exception as exc:  # noqa: BLE001
            raise _student_error(exc, "generate") from None
        result.update(shape_in=[1, len(ids)], shape_out=list(out.shape), generated=s.tokenizer.decode(generated),
                      steps=rows)
        return result

    try:
        with s.model_lock, torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long))
            probs = next_token_probs(logits, temperature)
            row = probs[0].tolist()
            result.update(logits_shape=list(logits.shape), shape=list(probs.shape), step=s.model_step)
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "next_token_probs") from None
    order = sorted(range(len(row)), key=lambda j: -row[j])
    result.update(part="probs", last=_token(s, ids[-1]), top=[_token(s, j, p=row[j]) for j in order[:10]],
                  sum=sum(row), rest=sum(row[j] for j in order[10:]), vocab_size=len(row))
    return result


# --------------------------------------------------------------------------- Lab 6
def viz_lab6(s, params):
    torch = s.torch
    import torch.nn.functional as F
    model = _model_ready(s)
    loss_fn = s.modules[6].next_token_loss
    text = _text(params, "text", s.corpus.default_text(), 60)
    vocab_size = len(s.tokenizer.vocab)
    result = {"lab": 6, "text": text, "vocab_size": vocab_size, "ln_vocab": math.log(vocab_size),
              **_provenance(s)}
    ids, error = _encode(s, text)
    if error or not ids or len(ids) < 2:
        result["input_error"] = error or {"kind": "short", "message": "Enter at least 2 characters."}
        return result
    positions = []
    try:
        with s.model_lock, torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long))                   # (1, T, V)
            rows = logits[0].tolist()
            for t in range(len(ids) - 1):
                target = ids[t + 1]
                loss = float(loss_fn(torch.tensor([[rows[t]]]), torch.tensor([[target]], dtype=torch.long)))
                p_next = F.softmax(torch.tensor(rows[t]), dim=0).tolist()[target]
                positions.append({"t": t, "current": _token(s, ids[t]), "next": _token(s, target),
                                  "loss": loss, "p": p_next})
            whole = float(loss_fn(torch.tensor([rows[:-1]]), torch.tensor([ids[1:]], dtype=torch.long)))
            result["step"] = s.model_step
            result["logits_shape"] = list(logits.shape)
    except Exception as exc:  # noqa: BLE001
        raise _student_error(exc, "next_token_loss") from None
    result.update(positions=positions, mean=sum(p["loss"] for p in positions) / len(positions), whole=whole,
                  ids=ids)
    return result


# --------------------------------------------------------------------------- Lab 7
def viz_lab7(s, params):
    from .session import TRAIN_DEFAULTS
    from .training import TrainingConfig
    if params.get("part") == "matrix":
        mode = "logits" if params.get("mode") == "logits" else "probs"
        return {"lab": 7, "matrix": _matrix(s, mode)}
    return {"lab": 7, "training": s.trainer.status_json(), "history": s.history,
            "previous_curves": s.previous_curves,
            "floor": s.corpus.bigram_floor, "ln_vocab": s.corpus.ln_vocab, "vocab_size": s.corpus.vocab_size,
            "defaults": TRAIN_DEFAULTS, "learning_rates": list(TrainingConfig.LEARNING_RATES),
            "model": {"step": s.model_step, "seed": s.model_seed, "params": s.model_params()}}


BUILDERS = {0: viz_lab0, 1: viz_lab1, 2: viz_lab2, 3: viz_lab3, 4: viz_lab4, 5: viz_lab5, 6: viz_lab6,
            7: viz_lab7}


def build(session, n, params):
    if n not in BUILDERS:
        raise VizError("Unknown lab.", 404)
    if n not in session.done:
        raise VizError("Verify this lab to unlock the visualization.", 409)
    if n > 0 and session.torch is None:
        raise VizError("PyTorch is not installed in this Python — return to Lab 0.", 409)
    return BUILDERS[n](session, params)
