"""Lazy PyTorch import: the server must start (and explain the problem) even without torch."""
import os
import sys
import threading

_lock = threading.Lock()
_cache = {}


def load_torch():
    """Import torch once and configure threads. Returns (torch_module or None, error_text or None)."""
    with _lock:
        if "result" in _cache:
            return _cache["result"]
        try:
            import torch
        except Exception as exc:  # ImportError, or OSError when a DLL is missing on Windows
            _cache["result"] = (None, f"{type(exc).__name__}: {exc}")
            return _cache["result"]
        try:
            # Leave one core free so the browser and the server stay responsive.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:
            pass
        _cache["result"] = (torch, None)
        return _cache["result"]


def torch_info():
    """Summary for the UI header and Lab 0."""
    torch, error = load_torch()
    if torch is None:
        hint = None
        if error and os.name == "nt" and ("DLL" in error or "WinError" in error):
            hint = ("On Windows, this error can usually be resolved by installing the "
                    "Microsoft Visual C++ Redistributable (x64) and restarting the computer.")
        return {"available": False, "version": None, "threads": None, "error": error, "hint": hint,
                "python": sys.executable}
    try:
        threads = int(torch.get_num_threads())
    except Exception:
        threads = None
    return {"available": True, "version": str(getattr(torch, "__version__", "?")), "threads": threads,
            "error": None, "hint": None, "python": sys.executable}
