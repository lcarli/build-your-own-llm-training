"""Session state: which labs are done, the student's modules, tokenizer, data and model.

Verifying lab N reloads bricks 0..N (fresh modules), runs the checks of lab N (and of any earlier
lab whose code changed since it was verified), then commits the results. Progress is persisted to
.workshop/progress.json and the trained model to .workshop/model.pt.
"""
import json
import os
import threading
import time
from pathlib import Path

from .catalog import (BRICKS_DIR, DATA_DIR, DEFAULT_GUIDE_LANGUAGE, FUTURE_LABS, GUIDE_LANGUAGES,
                      LABS, LAST_LAB, MODEL_SEED, N_EMBD, REFERENCE_DIR,
                      STATE_DIR, brick_header, BLOCK_SIZE, EVAL_BATCHES, EVAL_INTERVAL, SAMPLE_CHARS,
                      SAMPLE_SEED, SAMPLE_STEPS, TRAIN_FRACTION)
from .checks import CheckContext, run_lab_checks
from .corpus import load_corpus
from .loader import (BrickProblem, has_student_code, install_modules, load_brick, restore_modules,
                     snapshot_modules)
from .textfmt import fmt_int
from .torch_support import load_torch, torch_info
from .training import EventHub, Trainer, TrainingConfig, new_history

PROGRESS_FILE = "progress.json"
MODEL_FILE = "model.pt"
HISTORY_FILE = "training.json"
CURVES_FILE = "curves.json"          # loss curves of previous models, to compare learning rates
MAX_PREVIOUS_CURVES = 3
TRAIN_DEFAULTS = {"learning_rate": 0.01, "steps": 3000, "batch_size": 32, "seed": MODEL_SEED}


class SessionError(Exception):
    """A request that cannot be served right now (message in English, HTTP status)."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.message = message
        self.status = status


class _ChecksFailed(Exception):
    pass


def _write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Session:
    def __init__(self, mode="build", state_dir=None, log=print, language=DEFAULT_GUIDE_LANGUAGE):
        if language not in GUIDE_LANGUAGES:
            raise ValueError(f"Unsupported guide language: {language!r}. Choose pt, en, or fr.")
        self.language = language
        self.mode = mode
        self.explore = mode == "explore"
        self.source_dir = REFERENCE_DIR if self.explore else BRICKS_DIR
        if state_dir is not None:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = STATE_DIR / "explore" if self.explore else STATE_DIR
        self.log = log
        self.corpus = load_corpus(DATA_DIR)
        self.lock = threading.RLock()          # serializes verification / reset / training start
        self.model_lock = threading.Lock()     # shared by the training loop and the visualizations
        self.hub = EventHub()
        self.trainer = Trainer(self)
        self.done = set()
        self.modules = {}                      # lab -> module that passed its checks
        self.digests = {}                      # lab -> hash of the code that passed its checks
        self.lab0 = None
        self.tokenizer = None
        self.ids = None
        self.train_data = None
        self.val_data = None
        self.model = None
        self.model_signature = None
        self.model_seed = MODEL_SEED
        self.model_step = 0
        self.history = new_history()
        self.previous_curves = self._read_curves()
        self.notices = []                      # shown once in the UI (e.g. restore problems)
        self.cache = {}

    @property
    def torch(self):
        return load_torch()[0]

    # ------------------------------------------------------------------ status
    def available(self, n):
        return n == 0 or (n - 1) in self.done

    def lab_status(self, n):
        if n in self.done:
            return "done"
        return "available" if self.available(n) else "locked"

    def labs_json(self):
        labs = [{"n": lab.number, "title": lab.title, "short": lab.short, "status": self.lab_status(lab.number),
                 "brick": lab.brick, "path": lab.rel_path} for lab in LABS]
        future = [{"n": n, "title": title, "short": short, "status": "soon"} for n, title, short in FUTURE_LABS]
        return labs, future

    def model_params(self):
        if self.model is None:
            return None
        return int(sum(p.numel() for p in self.model.parameters()))

    def state_json(self):
        labs, future = self.labs_json()
        return {
            "mode": self.mode,
            "language": self.language,
            "corpus": self.corpus.summary(),
            "torch": torch_info(),
            "labs": labs,
            "future_labs": future,
            "model": {"exists": self.model is not None, "step": self.model_step, "seed": self.model_seed,
                      "params": self.model_params()},
            "training": self.trainer.status_json(),
            "history": self.history,
            "previous_curves": self.previous_curves,
            "notices": self.notices,
            "constants": {"n_embd": N_EMBD, "block_size": BLOCK_SIZE, "eval_interval": EVAL_INTERVAL,
                          "eval_batches": EVAL_BATCHES, "sample_seed": SAMPLE_SEED, "sample_chars": SAMPLE_CHARS,
                          "sample_steps": list(SAMPLE_STEPS), "train_fraction": TRAIN_FRACTION,
                          "model_seed": MODEL_SEED},
            "train_defaults": TRAIN_DEFAULTS,
            "learning_rates": list(TrainingConfig.LEARNING_RATES),
        }

    # ------------------------------------------------------------------ verification
    def _context(self, modules, staged):
        return CheckContext(self, modules, staged)

    def verify(self, n, silent=False):
        """Reload bricks 0..n, check lab n (and earlier labs whose code changed), commit on success."""
        with self.lock:
            if self.trainer.running:
                raise SessionError("Training is running. Click Stop before verifying.")
            if not self.available(n):
                raise SessionError(f"This lab is still locked: complete Lab {n - 1} first.")
            result = {"ok": False, "lab": n, "groups": [], "error": None, "warnings": [], "notices": [],
                      "kept_previous": False}
            snapshot = snapshot_modules()
            loaded = {}
            to_check = []
            try:
                for k in range(n + 1):
                    loaded[k] = load_brick(LABS[k], self.source_dir)
                    result["warnings"].extend(loaded[k].source.warnings)
                changed = [k for k in range(n + 1) if k in self.done and self.digests.get(k) != loaded[k].digest]
                to_check = [k for k in changed if k != n] + [n]
                modules = {k: b.module for k, b in loaded.items()}
                staged = {}
                for k in to_check:
                    group, out = run_lab_checks(k, self._context(modules, staged))
                    group["changed"] = k != n
                    result["groups"].append(group)
                    if not group["ok"]:
                        raise _ChecksFailed()
                    staged[k] = out
            except BrickProblem as problem:
                restore_modules(snapshot)
                result["error"] = problem.as_dict()
                result["kept_previous"] = n in self.done
                if n == 0 and problem.lab == 0:
                    # Lab 0 is also the computer checklist: run it even when greet cannot be loaded yet.
                    context = self._context({}, {})
                    context.brick_problem = problem
                    group, _out = run_lab_checks(0, context)
                    result["groups"].append(group)
                return result
            except _ChecksFailed:
                restore_modules(snapshot)
                result["kept_previous"] = n in self.done
                return result

            for k in range(n + 1):
                self.modules[k] = loaded[k].module
            for k in to_check:
                self.digests[k] = loaded[k].digest
                self._commit(k, staged[k], next(g for g in result["groups"] if g["lab"] == k))
            self.done.add(n)
            if changed:
                self._refresh_later(n, result)
            model_note = self._ensure_model(result)
            if model_note:
                for group in result["groups"]:
                    if group["lab"] == 4:
                        group["steps"].append({"name": "Session model", "status": "ok", "message": model_note})
            if not silent:
                self._save_progress()
            result["ok"] = True
            return result

    def _commit(self, k, out, group):
        """Keep what later labs need from a successful check."""
        if k == 0:
            self.lab0 = {"steps": group["steps"], "bench": out.get("bench"), "greet": out.get("greet"),
                         "torch": out.get("torch"), "verified_at": time.time()}
        elif k == 1:
            new_ids = out["ids"]
            ids_changed = self.ids is not None and new_ids != self.ids
            self.tokenizer = out["tokenizer"]
            self.ids = new_ids
            self.cache.clear()
            if ids_changed and 2 in self.done and 2 in self.modules:
                self.train_data, self.val_data = self.modules[2].split_train_val(self.ids)
        elif k == 2:
            self.train_data, self.val_data = out["train"], out["val"]

    def _refresh_later(self, n, result):
        """Code of lab <= n changed: reload and re-check the later labs that were already done."""
        later = sorted(m for m in self.done if m > n)
        refreshed = []
        for m in later:
            snapshot = snapshot_modules()
            try:
                brick = load_brick(LABS[m], self.source_dir)
                modules = dict(self.modules)
                modules[m] = brick.module
                group, out = run_lab_checks(m, self._context(modules, {}))
                if not group["ok"]:
                    failed = next((s for s in group["steps"] if s["status"] == "fail"), None)
                    raise BrickProblem("Verification", failed["message"] if failed else "failed", lab=m)
            except BrickProblem as problem:
                restore_modules(snapshot)
                self._undo_from(m)
                result["notices"].append(
                    f"Lab {m} needs to be verified again because the code in an earlier lab changed "
                    f"({problem.message})")
                break
            self.modules[m] = brick.module
            self.digests[m] = brick.digest
            self._commit(m, out, group)
            refreshed.append(m)
        install_modules(self.modules)
        if refreshed:
            labs = ", ".join(str(m) for m in refreshed)
            result["notices"].append(f"The code changed, so Labs {labs} were verified again with it.")

    def _undo_from(self, m):
        for k in range(m, LAST_LAB + 1):
            self.done.discard(k)
            self.digests.pop(k, None)
        if m <= 4 and self.model is not None:
            with self.model_lock:
                self.model = None
                self.model_signature = None

    def _signature(self):
        return (self.digests.get(3), self.digests.get(4), "".join(self.tokenizer.vocab), N_EMBD)

    def _ensure_model(self, result):
        """Create the session model at Lab 4, or rebuild it when the code of Lab 3/4 changed."""
        if 4 not in self.done:
            return None
        signature = self._signature()
        if self.model is not None and self.model_signature == signature:
            if self.model_step:
                return f"preserved: the code has not changed (trained for {fmt_int(self.model_step)} steps)"
            return f"preserved: the code has not changed (seed {self.model_seed})"
        had_model = self.model is not None
        # At startup there is no model yet: keep .workshop/model.pt so restore() can load it.
        self._create_model(self.model_seed, discard_saved=had_model)
        self.model_signature = signature
        if had_model:
            result["notices"].append("Model reset because the code changed: a new BigramLM was created "
                                     f"with seed {self.model_seed}.")
            result["model_reset"] = True
        return f"BigramLM({self.corpus.vocab_size}, {N_EMBD}) created with seed {self.model_seed}"

    def _archive_curve(self):
        """Keep the loss curve of the model being replaced, so the student can compare runs."""
        evals = self.history.get("evals") or []
        if len(evals) < 2:
            return
        rates = []
        for run in self.history.get("runs") or []:
            if run["learning_rate"] not in rates:
                rates.append(run["learning_rate"])
        rates_text = ", ".join(f"{lr:g}".replace(".", ",") for lr in rates) or "?"
        self.previous_curves.append({
            "label": f"lr {rates_text} · {fmt_int(self.model_step)} steps",
            "points": [[e["step"], e["train"], e["val"]] for e in evals],
        })
        self.previous_curves = self.previous_curves[-MAX_PREVIOUS_CURVES:]
        try:
            _write_json_atomic(self.state_dir / CURVES_FILE, {"curves": self.previous_curves})
        except OSError as exc:
            self.log(f"[warning] could not save the previous curves: {exc}")

    def _read_curves(self):
        path = self.state_dir / CURVES_FILE
        if not path.exists():
            return []
        try:
            curves = json.loads(path.read_text(encoding="utf-8")).get("curves", [])
            return [c for c in curves if isinstance(c, dict) and "label" in c and "points" in c][-MAX_PREVIOUS_CURVES:]
        except (ValueError, OSError, AttributeError):
            return []

    def _create_model(self, seed, discard_saved):
        torch = self.torch
        if discard_saved:
            self._archive_curve()
        with self.model_lock:
            torch.manual_seed(seed)
            self.model = self.modules[4].BigramLM(len(self.tokenizer.vocab), N_EMBD)
            self.model_seed = seed
            self.model_step = 0
            self.history = new_history()
        self.cache.clear()
        if not discard_saved:
            return
        for name in (MODEL_FILE, HISTORY_FILE):
            path = self.state_dir / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------ Plan B
    def complete(self, n):
        """Write header + marker + reference code into bricks/bNN.py, then verify."""
        with self.lock:
            if self.trainer.running:
                raise SessionError("Training is running. Click Stop before completing this step.")
            if not self.available(n):
                raise SessionError(f"This lab is still locked: complete Lab {n - 1} first.")
            lab = LABS[n]
            backup = None
            if not self.explore:
                target = BRICKS_DIR / f"{lab.brick}.py"
                reference = (REFERENCE_DIR / f"{lab.brick}.py").read_text(encoding="utf-8")
                if target.exists():
                    old = target.read_text(encoding="utf-8", errors="replace")
                    if has_student_code(old) and old != brick_header(lab) + reference:
                        backup_dir = self.state_dir / "backup"
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        backup = backup_dir / f"{lab.brick}-{time.strftime('%Y%m%d-%H%M%S')}.py"
                        backup.write_text(old, encoding="utf-8")
                target.write_text(brick_header(lab) + reference, encoding="utf-8")
            result = self.verify(n)
            if backup is not None:
                result["notices"].append(f"Your previous file was saved to .workshop/backup/{backup.name}.")
            return result

    # ------------------------------------------------------------------ model reset / training
    def reset_model(self, seed=MODEL_SEED):
        with self.lock:
            if self.trainer.running:
                raise SessionError("Training is running. Click Stop before resetting the model.")
            if 4 not in self.done:
                raise SessionError("The model is created in Lab 4.")
            self._create_model(int(seed), discard_saved=True)
            self.model_signature = self._signature()
            return {"ok": True, "message": f"Model reset with seed {self.model_seed}.", "state": self.state_json()}

    def start_training(self, data):
        with self.lock:
            if LAST_LAB not in self.done:
                raise SessionError("Verify Lab 7 to unlock training.")
            if self.trainer.running:
                raise SessionError("Training is already running.")
            if self.model is None:
                raise SessionError("No model exists: verify Lab 4.")
            try:
                config = TrainingConfig.from_json(data)
            except ValueError as exc:
                raise SessionError(str(exc), status=400) from None
            self.trainer.start(config)
            return {"ok": True, "training": self.trainer.status_json()}

    def stop_training(self):
        self.trainer.stop()
        return {"ok": True, "training": self.trainer.status_json()}

    # ------------------------------------------------------------------ persistence
    def _save_progress(self):
        if self.explore:
            return
        try:
            _write_json_atomic(self.state_dir / PROGRESS_FILE, {"version": 1, "done": sorted(self.done)})
        except OSError as exc:
            self.log(f"[warning] could not save progress: {exc}")

    def _read_progress(self):
        path = self.state_dir / PROGRESS_FILE
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            done = sorted(int(n) for n in data.get("done", []) if 0 <= int(n) <= LAST_LAB)
        except (ValueError, TypeError, OSError, AttributeError):
            self.log("[warning] .workshop/progress.json is corrupted; starting from scratch.")
            return []
        prefix = []
        for expected, n in enumerate(done):
            if n != expected:
                break
            prefix.append(n)
        return prefix

    def save_model(self):
        """Save the weights (state_dict) and the training history. Returns True on success."""
        torch = self.torch
        if self.model is None or torch is None:
            return False
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with self.model_lock:
                payload = {"format": 1, "step": int(self.model_step), "seed": int(self.model_seed),
                           "vocab": "".join(self.tokenizer.vocab), "n_embd": N_EMBD,
                           "state_dict": self.model.state_dict()}
                tmp = self.state_dir / (MODEL_FILE + ".tmp")
                torch.save(payload, str(tmp))
                os.replace(tmp, self.state_dir / MODEL_FILE)
            _write_json_atomic(self.state_dir / HISTORY_FILE, {"step": self.model_step, "history": self.history})
            return True
        except Exception as exc:  # disk full, permissions...
            self.log(f"[warning] could not save the model: {exc}")
            return False

    def _load_saved_model(self):
        path = self.state_dir / MODEL_FILE
        if not path.exists() or self.model is None:
            return
        torch = self.torch
        try:
            payload = torch.load(str(path), weights_only=True)
            state = payload["state_dict"]
            if payload.get("vocab") != "".join(self.tokenizer.vocab):
                raise ValueError("the corpus vocabulary changed")
            current = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}
            saved = {k: tuple(v.shape) for k, v in state.items()}
            if current != saved:
                raise ValueError("the parameter shapes do not match the current code")
            with self.model_lock:
                self.model.load_state_dict(state)
                self.model_step = int(payload.get("step", 0))
                self.model_seed = int(payload.get("seed", MODEL_SEED))
            history_path = self.state_dir / HISTORY_FILE
            if history_path.exists():
                data = json.loads(history_path.read_text(encoding="utf-8"))
                if data.get("step") == self.model_step:
                    self.history = data.get("history") or new_history()
            self.log(f"Trained model restored (step {fmt_int(self.model_step)}).")
        except Exception as exc:
            self.notices.append(f"The model saved in .workshop/{MODEL_FILE} could not be used ({exc}); "
                                "a new model was created.")

    def restore(self):
        """Rebuild the state at startup by re-verifying the labs that were done (silently)."""
        target = list(range(LAST_LAB + 1)) if self.explore else self._read_progress()
        for n in target:
            try:
                result = self.verify(n, silent=True)
            except SessionError as exc:
                self.notices.append(exc.message)
                break
            if not result["ok"]:
                reason = ""
                if result["error"]:
                    err = result["error"]
                    reason = f"{err['title']} in {err.get('path') or 'bricks/'}"
                else:
                    failed = [s for g in result["groups"] for s in g["steps"] if s["status"] == "fail"]
                    if failed:
                        reason = failed[0]["message"]
                self.notices.append(f"Lab {n} was already completed, but verification now failed: {reason}. "
                                    "Open the lab and click Verify.")
                self.log(f"[warning] Lab {n}: {reason}")
                break
        if LAST_LAB in self.done:
            self._load_saved_model()
        return sorted(self.done)
