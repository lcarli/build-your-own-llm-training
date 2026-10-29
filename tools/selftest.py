"""Instructor self-test: run the whole workshop against the reference code, on a machine WITH PyTorch.

Usage:
    python tools/selftest.py

It verifies Labs 0-7 with the explore-mode loader (reference/), builds every visualization panel,
trains for 300 steps through the real training loop and reloads the saved model. It never touches
bricks/ or .workshop/: all state goes to a temporary folder. Exit code 0 = everything passed.
"""
import sys

sys.dont_write_bytecode = True

import platform  # noqa: E402
import queue  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

TRAIN_STEPS = 300


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    @property
    def failures(self):
        return [r for r in self.rows if not r[1]]


def first_failure(result):
    if result.get("error"):
        err = result["error"]
        return f"{err['title']}: {err['message']}"
    for group in result.get("groups", []):
        for step in group["steps"]:
            if step["status"] == "fail":
                return f"{step['name']}: {step['message']}"
    return "failed"


def check_panels(session, report):
    from workshop import viz

    def panel(name, n, params, check=None):
        try:
            data = viz.build(session, n, params)
            problem = check(data) if check else None
            return report.add(name, problem is None, problem or "")
        except Exception as exc:  # noqa: BLE001
            return report.add(name, False, f"{type(exc).__name__}: {getattr(exc, 'message', exc)}")

    panel("Lab 0 panel", 0, {}, lambda d: None if d.get("bench") else "missing benchmark result")
    panel("Lab 1 panel (roma)", 1, {}, lambda d: None if d["mini"].get("ids") == [8, 6, 5, 1] else "incorrect encode('roma')")
    panel("Lab 1 panel (Roma → expected KeyError)", 1, {"text": "Roma"},
          lambda d: None if d["mini"].get("encode_error") else "missing KeyError message")
    panel("Lab 2 panel", 2, {"block_size": 8, "seed": 1337},
          lambda d: None if d["shape_x"] == [4, 8] and all(y[:-1] == x[1:] for x, y in zip(d["x"], d["y"])) else "invalid batch")
    panel("Lab 2 panel (block size 16)", 2, {"block_size": 16, "seed": 7},
          lambda d: None if d["shape_x"] == [4, 16] else "invalid shape")
    panel("Lab 3 panel", 3, {}, lambda d: None if d.get("heat") and len(d["pca"]["points"]) == session.corpus.vocab_size else "incomplete data")
    panel("Lab 4 panel", 4, {}, lambda d: None if len(d["top"]) == 15 and len(d["matrix"]["matrix"]) == session.corpus.vocab_size else "incomplete data")
    panel("Lab 4 panel (probabilities)", 4, {"part": "matrix", "mode": "probs"},
          lambda d: None if all(abs(sum(r) - 1) < 1e-2 for r in d["matrix"]["matrix"]) else "rows do not sum to 1")
    panel("Lab 5 panel (temperature 1)", 5, {}, lambda d: None if abs(d["sum"] - 1) < 1e-4 else "sum differs from 1")
    panel("Lab 5 panel (temperature 0.1)", 5, {"temperature": 0.1}, lambda d: None if abs(d["sum"] - 1) < 1e-4 else "sum differs from 1")
    panel("Lab 5 panel (Generate)", 5, {"part": "generate", "seed": 42},
          lambda d: None if len(d["generated"]) == 200 and len(d["steps"]) == 20 else "incomplete generation")
    panel("Lab 6 panel", 6, {}, lambda d: None if d["positions"] and abs(d["mean"] - d["whole"]) < 1e-3 else "inconsistent loss")
    panel("Lab 7 panel", 7, {}, lambda d: None if d["floor"] > 0 else "invalid bigram floor")
    panel("Lab 7 panel (V × V table)", 7, {"part": "matrix"}, None)


def check_training(session, report, state_dir):
    from workshop.session import Session

    events = session.hub.subscribe()
    t0 = time.perf_counter()
    try:
        session.start_training({"learning_rate": 0.01, "steps": TRAIN_STEPS, "batch_size": 32, "seed": 1337})
    except Exception as exc:  # noqa: BLE001
        return report.add("300-step training", False, f"{type(exc).__name__}: {exc}")
    session.trainer.join(timeout=900)
    elapsed = time.perf_counter() - t0
    names = []
    error = None
    while True:
        try:
            event, payload = events.get_nowait()
        except queue.Empty:
            break
        names.append(event)
        if event == "error":
            error = payload
    if error or "done" not in names:
        return report.add("300-step training", False, error or "training did not finish")
    evals = session.history["evals"]
    report.add("300-step training", True, f"{elapsed:.1f} s".replace(".", ","))
    report.add("Evaluations at steps 0, 250, and 300", [e["step"] for e in evals] == [0, 250, TRAIN_STEPS],
               str([e["step"] for e in evals]))
    report.add("Training loss decreased", evals[-1]["train"] < evals[0]["train"],
               f"{evals[0]['train']:.3f} → {evals[-1]['train']:.3f}".replace(".", ","))
    samples = session.history["samples"]
    report.add("Gallery: samples from step 0 and the final step", [s["step"] for s in samples] == [0, TRAIN_STEPS]
               and all(len(s["text"]) == 200 for s in samples), str([s["step"] for s in samples]))
    report.add("Progress events sent", "progress" in names and "eval" in names and "sample" in names)
    report.add("Model saved (model.pt)", (Path(state_dir) / "model.pt").exists())

    restored = Session(mode="explore", state_dir=state_dir, log=lambda message: None)
    restored.restore()
    report.add("Model restored after restarting", restored.model_step == TRAIN_STEPS,
               f"step {restored.model_step}")


def main():
    from workshop.torch_support import torch_info

    print("Build Your Own LLM — self-test (reference code)\n")
    info = torch_info()
    report = Report()
    if not info["available"]:
        report.add("PyTorch installed", False, info["error"])
        print("\nInstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 1
    threads = info["threads"]
    print(f"Python {platform.python_version()} · PyTorch {info['version']} · {threads} "
          f"{'thread' if threads == 1 else 'threads'}")

    from workshop.session import Session
    from workshop.textfmt import fmt_int

    with tempfile.TemporaryDirectory(prefix="byollm-selftest-") as state_dir:
        session = Session(mode="explore", state_dir=state_dir, log=lambda message: None)
        corpus = session.corpus
        print(f"Corpus: {corpus.name} · {fmt_int(corpus.n_chars)} characters · V = {corpus.vocab_size}\n")
        all_labs = True
        for n in range(8):
            t0 = time.perf_counter()
            try:
                result = session.verify(n, silent=True)
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": {"title": type(exc).__name__, "message": str(exc)}}
            seconds = f"{time.perf_counter() - t0:.1f} s".replace(".", ",")
            ok = report.add(f"Lab {n} verified", result["ok"], seconds if result["ok"] else first_failure(result))
            all_labs = all_labs and ok
            if not ok:
                break
        if all_labs:
            check_panels(session, report)
            check_training(session, report, state_dir)

    total = len(report.rows)
    failed = len(report.failures)
    print()
    if failed:
        print(f"✗ {failed} of {total} checks failed.")
        return 1
    print(f"✓ All good: {total} of {total} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
