"""Training loop (background thread) and the event hub that feeds the Server-Sent Events stream.

The loop only calls the student's code: get_batch (Lab 2), make_optimizer and train_step (Lab 7),
next_token_loss (Lab 6) for the evaluations and generate (Lab 5) for the gallery samples.
"""
import json
import math
import queue
import threading
import time

from .catalog import (BLOCK_SIZE, EVAL_BATCHES, EVAL_INTERVAL, SAMPLE_CHARS, SAMPLE_SEED, SAMPLE_STEPS)
from .loader import describe_exception
from .textfmt import fmt_float, fmt_int


class EventHub:
    """Broadcast events to every connected SSE client; each client has its own queue."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues = set()

    def subscribe(self):
        q = queue.Queue(maxsize=5000)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._queues.discard(q)

    def publish(self, event, data):
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock:
            targets = list(self._queues)
        for q in targets:
            try:
                q.put_nowait((event, payload))
            except queue.Full:
                pass

    @property
    def client_count(self):
        with self._lock:
            return len(self._queues)


def new_history():
    return {"evals": [], "samples": [], "runs": []}


class TrainingConfig:
    LEARNING_RATES = (0.1, 0.01, 0.001)

    def __init__(self, learning_rate=0.01, steps=3000, batch_size=32, seed=1337):
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    @classmethod
    def from_json(cls, data):
        """Validate the values sent by the UI. Raises ValueError with an English message."""
        try:
            cfg = cls(data.get("learning_rate", 0.01), data.get("steps", 3000),
                      data.get("batch_size", 32), data.get("seed", 1337))
        except (TypeError, ValueError):
            raise ValueError("Invalid training values.") from None
        if not (0 < cfg.learning_rate <= 1):
            raise ValueError("The learning rate must be between 0 and 1.")
        if not (1 <= cfg.steps <= 20000):
            raise ValueError("The number of steps must be between 1 and 20.000.")
        if not (1 <= cfg.batch_size <= 512):
            raise ValueError("The batch size must be between 1 and 512.")
        if not (0 <= cfg.seed < 2 ** 31):
            raise ValueError("The seed must be a positive integer.")
        return cfg

    def as_dict(self):
        return {"learning_rate": self.learning_rate, "steps": self.steps,
                "batch_size": self.batch_size, "seed": self.seed}


class Trainer:
    """Runs the training loop in a daemon thread; one run at a time."""

    PROGRESS_INTERVAL = 0.2  # seconds between "progress" events

    def __init__(self, session):
        self.session = session
        self.thread = None
        self.stop_event = threading.Event()
        self.running = False
        self.config = None
        self.run_start_step = 0
        self.run_target_step = 0
        self.last_loss = None
        self.started_at = None
        self.last_error = None

    def status_json(self):
        s = self.session
        elapsed = (time.perf_counter() - self.started_at) if (self.running and self.started_at) else None
        return {
            "running": self.running,
            "step": s.model_step,
            "run_start": self.run_start_step,
            "run_target": self.run_target_step,
            "last_loss": self.last_loss,
            "elapsed": elapsed,
            "config": self.config.as_dict() if self.config else None,
            "error": self.last_error,
        }

    def start(self, config):
        if self.running:
            raise RuntimeError("Training is already running.")
        self.stop_event.clear()
        self.config = config
        self.last_error = None
        self.running = True
        self.run_start_step = self.session.model_step
        self.run_target_step = self.session.model_step + config.steps
        self.thread = threading.Thread(target=self._run, args=(config,), name="training", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def join(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)

    # ------------------------------------------------------------------ helpers (run in the thread)
    def _evaluate(self, torch, model, get_batch, loss_fn, generator, batch_size):
        """Mean loss of EVAL_BATCHES batches of each split, without gradients."""
        s = self.session
        result = {}
        with s.model_lock, torch.no_grad():
            model.eval()
            try:
                for split, data in (("train", s.train_data), ("val", s.val_data)):
                    total = 0.0
                    for _ in range(EVAL_BATCHES):
                        xb, yb = get_batch(data, BLOCK_SIZE, batch_size, generator=generator)
                        total += float(loss_fn(model(xb), yb))
                    result[split] = total / EVAL_BATCHES
            finally:
                model.train()
            result["step"] = s.model_step
        return result

    def _sample(self, torch, model, generate, label):
        s = self.session
        start_id = s.tokenizer.encode(s.corpus.start_char())
        with s.model_lock:
            idx = torch.tensor([list(start_id)], dtype=torch.long)
            out = generate(model, idx, SAMPLE_CHARS, temperature=1.0,
                           generator=torch.Generator().manual_seed(SAMPLE_SEED))
            step = s.model_step
        ids = out.tolist()[0][1:]
        return {"step": step, "label": label, "text": s.tokenizer.decode(ids)}

    def _record_eval(self, ev):
        evals = self.session.history["evals"]
        if evals and evals[-1]["step"] == ev["step"]:
            evals[-1] = ev
        else:
            evals.append(ev)
        self.session.hub.publish("eval", ev)

    def _record_sample(self, sample):
        samples = self.session.history["samples"]
        if samples and samples[-1]["step"] == sample["step"]:
            samples[-1] = sample
        else:
            samples.append(sample)
        self.session.hub.publish("sample", sample)

    # ------------------------------------------------------------------ the loop
    def _run(self, config):
        s = self.session
        hub = s.hub
        torch = s.torch
        stopped = False
        run_info = {"start": s.model_step, "end": s.model_step, **config.as_dict(), "stopped": False,
                    "seconds": 0.0}
        t0 = time.perf_counter()
        self.started_at = t0
        try:
            get_batch = s.modules[2].get_batch
            generate = s.modules[5].generate
            loss_fn = s.modules[6].next_token_loss
            make_optimizer = s.modules[7].make_optimizer
            train_step = s.modules[7].train_step
            model = s.model
            batch_generator = torch.Generator().manual_seed(config.seed)
            eval_generator = torch.Generator().manual_seed(config.seed + 1)
            with s.model_lock:
                optimizer = make_optimizer(model, config.learning_rate)
            hub.publish("progress", self._progress(config, t0))

            if s.model_step == 0:
                self._record_eval(self._evaluate(torch, model, get_batch, loss_fn, eval_generator,
                                                 config.batch_size))
                self._record_sample(self._sample(torch, model, generate, "step 0 (untrained)"))

            last_emit = 0.0
            for _ in range(config.steps):
                if self.stop_event.is_set():
                    stopped = True
                    break
                with s.model_lock:
                    model.train()
                    xb, yb = get_batch(s.train_data, BLOCK_SIZE, config.batch_size, generator=batch_generator)
                    loss = float(train_step(model, optimizer, xb, yb))
                    s.model_step += 1
                    step = s.model_step
                self.last_loss = loss
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        "The loss became infinite or NaN: the learning rate is too high for this model. "
                        "Click Reset model and try a lower learning rate.")
                now = time.perf_counter()
                if now - last_emit >= self.PROGRESS_INTERVAL:
                    last_emit = now
                    hub.publish("progress", self._progress(config, t0))
                if step % EVAL_INTERVAL == 0:
                    self._record_eval(self._evaluate(torch, model, get_batch, loss_fn, eval_generator,
                                                     config.batch_size))
                if step in SAMPLE_STEPS:
                    self._record_sample(self._sample(torch, model, generate, f"step {fmt_int(step)}"))
                time.sleep(0)  # let the visualization requests take the model lock

            final_step = s.model_step
            if not s.history["evals"] or s.history["evals"][-1]["step"] != final_step:
                self._record_eval(self._evaluate(torch, model, get_batch, loss_fn, eval_generator,
                                                 config.batch_size))
            label = f"training finished · step {fmt_int(final_step)}"
            if stopped:
                label = f"training stopped · step {fmt_int(final_step)}"
            last_sample = s.history["samples"][-1] if s.history["samples"] else None
            if last_sample and last_sample["step"] == final_step:
                last_sample = dict(last_sample, label=label)
                self._record_sample(last_sample)
            else:
                self._record_sample(self._sample(torch, model, generate, label))
            run_info.update(end=final_step, stopped=stopped, seconds=time.perf_counter() - t0)
            s.history["runs"].append(run_info)
            saved = s.save_model()
            self.running = False
            hub.publish("progress", self._progress(config, t0))
            hub.publish("done", {"step": final_step, "stopped": stopped, "saved": saved,
                                 "seconds": run_info["seconds"],
                                 "message": (f"Training {'stopped' if stopped else 'completed'} at step "
                                             f"{fmt_int(final_step)} in {fmt_float(run_info['seconds'], 1)} s.")})
        except BaseException as exc:  # noqa: BLE001 - report everything to the UI
            info = describe_exception(exc)
            if isinstance(exc, FloatingPointError):
                info["hint"] = None
            self.last_error = info
            run_info.update(end=s.model_step, stopped=True, seconds=time.perf_counter() - t0)
            s.history["runs"].append(run_info)
            self.running = False
            hub.publish("error", {"step": s.model_step, **info,
                                  "text": f"{info['type']}: {info['message']}"})
        finally:
            self.running = False

    def _progress(self, config, t0):
        s = self.session
        elapsed = time.perf_counter() - t0
        done = s.model_step - self.run_start_step
        return {
            "step": s.model_step,
            "run_start": self.run_start_step,
            "run_target": self.run_target_step,
            "run_done": done,
            "run_total": config.steps,
            "loss": self.last_loss,
            "elapsed": elapsed,
            "steps_per_sec": (done / elapsed) if elapsed > 0 else None,
            "running": self.running,
        }
