"""Build Your Own LLM - Interactive Lab.

Usage:
    python run.py                 # build mode: the student's code in bricks/
    python run.py --explore       # explore mode: reference code, every lab unlocked
    python run.py --lang fr       # lab guides in French (default: Portuguese)
    python run.py --port 8001 --no-browser
"""
import sys

if sys.version_info < (3, 10):
    print("This workshop requires Python 3.10 or newer. You are using Python "
          f"{sys.version.split()[0]}. Install a newer version at https://www.python.org/downloads/")
    sys.exit(1)

sys.dont_write_bytecode = True  # bricks are reloaded from source on every "Verify"

import argparse  # noqa: E402
import errno  # noqa: E402
import threading  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles may not encode every character: never crash because of a print. Line buffering
# keeps the messages visible even when the output goes to a pipe or a file.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

PORT_BUSY_ERRNOS = {errno.EADDRINUSE, 10048}   # 10048 = WSAEADDRINUSE (Windows)
PORT_DENIED_ERRNOS = {errno.EACCES, 10013}      # 10013 = WSAEACCES (port reserved on Windows)


def parse_args():
    from workshop.catalog import DEFAULT_GUIDE_LANGUAGE, GUIDE_LANGUAGES

    parser = argparse.ArgumentParser(
        description="Build Your Own LLM — Interactive Lab (local server).")
    parser.add_argument("--port", type=int, default=8000, help="local server port (default: 8000)")
    parser.add_argument("--explore", action="store_true",
                        help="Explore mode: use the reference code and unlock all labs")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    parser.add_argument("--lang", choices=GUIDE_LANGUAGES, default=DEFAULT_GUIDE_LANGUAGE,
                        help="lab guide language: pt (Portuguese), en (English), fr (French); default: pt")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("the port must be a number between 1 and 65535")
    return args


def main():
    args = parse_args()
    from workshop.corpus import CorpusError
    from workshop.server import create_server
    from workshop.session import Session
    from workshop.textfmt import fmt_int
    from workshop.torch_support import torch_info

    print("Build Your Own LLM — Interactive Lab")
    mode = "explore" if args.explore else "build"
    try:
        session = Session(mode=mode, language=args.lang)
    except CorpusError as exc:
        print(f"Error: {exc}")
        return 1

    info = torch_info()
    if info["available"]:
        threads = info["threads"]
        print(f"PyTorch {info['version']} · {threads} CPU {'thread' if threads == 1 else 'threads'}")
    else:
        print("Warning: PyTorch was not found in this Python environment. Lab 0 explains how to install it.")
    corpus = session.corpus
    print(f"Corpus: {corpus.name} · {fmt_int(corpus.n_chars)} characters · V = {corpus.vocab_size}")
    if corpus.demo:
        print("Using the short demonstration text. For the workshop: python data/prepare_data.py")

    try:
        server = create_server(session, args.port)
    except OSError as exc:
        code = exc.errno if exc.errno is not None else getattr(exc, "winerror", None)
        if code in PORT_BUSY_ERRNOS or getattr(exc, "winerror", None) in PORT_BUSY_ERRNOS:
            print(f"Port {args.port} is already in use (the workshop may already be open in another window). "
                  f"Close the other window or use another port: python run.py --port {args.port + 1}")
            return 1
        if code in PORT_DENIED_ERRNOS or getattr(exc, "winerror", None) in PORT_DENIED_ERRNOS:
            print(f"The system denied access to port {args.port}. Use another port: "
                  f"python run.py --port {args.port + 1}")
            return 1
        raise

    if args.explore:
        print("Explore mode: loading the reference code...")
    else:
        print("Restoring progress...")
    done = session.restore()
    if done:
        print(f"Completed labs: {', '.join(str(n) for n in done)}")
    for notice in session.notices:
        print(f"Warning: {notice}")

    url = f"http://127.0.0.1:{server.port}/"
    print(f"\nOpen in your browser: {url}")
    print("To stop: Ctrl+C\n")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nShutting down the workshop...")
    finally:
        server.shutting_down.set()
        session.trainer.stop()
        session.trainer.join(timeout=5)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
