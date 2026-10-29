"""Local HTTP server (standard library only), bound to 127.0.0.1.

Security: every POST must carry the header `X-Workshop: 1` (a custom header forces a CORS
preflight, which this server never approves, so other websites cannot call the API); the Host
header must be 127.0.0.1:<port> or localhost:<port> (blocks DNS rebinding); static files are
served only from web/.
"""
import json
import os
import queue
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .catalog import LAST_LAB, WEB_DIR
from .guide import guide_payload, reference_code
from .session import SessionError
from .viz import VizError, build as build_viz

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
       "form-action 'none'")
HEARTBEAT_SECONDS = 15
MAX_BODY = 64 * 1024
LAB_ROUTE = re.compile(r"^/api/(verify|complete|guide|viz)/(\d{1,2})$")


class WorkshopServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server silently share the port; keep it off there.
    allow_reuse_address = os.name != "nt"

    def __init__(self, port, session, log=print):
        self.session = session
        self.log = log
        self.shutting_down = threading.Event()
        super().__init__(("127.0.0.1", port), WorkshopHandler)
        self.port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.allowed_origins = {f"http://{host}" for host in self.allowed_hosts}


class BadRequest(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


class WorkshopHandler(BaseHTTPRequestHandler):
    server_version = "BuildYourOwnLLM/1.0"
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):  # noqa: A002 - signature of the base class
        pass  # keep the console readable for students

    # ------------------------------------------------------------------ responses
    def _common_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status, message, **extra):
        self._json(status, {"ok": False, "error": message, **extra})

    # ------------------------------------------------------------------ entry points
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_OPTIONS(self):
        # Never approve a CORS preflight: only the workshop page (same origin) may call the API.
        self._dispatch("OPTIONS")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        try:
            host = (self.headers.get("Host") or "").strip().lower()
            if host not in self.server.allowed_hosts:
                return self._error(403, "Host not allowed.")
            url = urlsplit(self.path)
            if method == "POST":
                if self.headers.get("X-Workshop") != "1":
                    return self._error(403, "Request rejected: missing X-Workshop header.")
                origin = self.headers.get("Origin")
                if origin and origin.lower() not in self.server.allowed_origins:
                    return self._error(403, "Origin not allowed.")
                return self._route_post(url.path, self._read_json())
            if method != "GET":
                return self._error(403 if method == "OPTIONS" else 405, "Method not allowed.")
            if url.path.startswith("/api/"):
                params = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
                return self._route_get(url.path, params)
            return self._static(url.path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except BadRequest as exc:
            self._error(exc.status, exc.message)
        except SessionError as exc:
            self._error(exc.status, exc.message)
        except VizError as exc:
            self._error(exc.status, exc.message, detail=exc.info)
        except Exception as exc:  # noqa: BLE001 - never leave the browser without an answer
            self.server.log("[internal error]\n" + traceback.format_exc())
            try:
                self._error(500, f"Internal workshop error: {type(exc).__name__}: {exc}")
            except OSError:
                pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("Invalid Content-Length.") from None
        if length > MAX_BODY:
            raise BadRequest("Request body too large.", 413)
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise BadRequest("Invalid JSON.") from None
        if not isinstance(data, dict):
            raise BadRequest("Invalid JSON: expected an object.")
        return data

    # ------------------------------------------------------------------ routes
    def _lab_number(self, text):
        n = int(text)
        if not 0 <= n <= LAST_LAB:
            raise BadRequest("Unknown lab.", 404)
        return n

    def _route_get(self, path, params):
        session = self.server.session
        if path == "/api/state":
            return self._json(200, session.state_json())
        if path == "/api/train/stream":
            return self._stream()
        match = LAB_ROUTE.match(path)
        if match and match.group(1) == "guide":
            n = self._lab_number(match.group(2))
            return self._json(200, guide_payload(n, session.corpus, log=self.server.log,
                                                 language=session.language))
        if match and match.group(1) == "viz":
            n = self._lab_number(match.group(2))
            return self._json(200, build_viz(session, n, params))
        if path.startswith("/api/code/"):
            brick = unquote(path[len("/api/code/"):])
            code = reference_code(brick)
            if code is None:
                raise BadRequest("Code not found.", 404)
            return self._json(200, {"brick": brick, "code": code, "path": f"bricks/{brick}.py"})
        raise BadRequest("Unknown route.", 404)

    def _route_post(self, path, body):
        session = self.server.session
        match = LAB_ROUTE.match(path)
        if match and match.group(1) in ("verify", "complete"):
            n = self._lab_number(match.group(2))
            result = session.verify(n) if match.group(1) == "verify" else session.complete(n)
            result["state"] = session.state_json()
            return self._json(200, result)
        if path == "/api/train/start":
            return self._json(200, session.start_training(body))
        if path == "/api/train/stop":
            return self._json(200, session.stop_training())
        if path == "/api/model/reset":
            try:
                seed = int(body.get("seed", 1337))
            except (TypeError, ValueError):
                raise BadRequest("Invalid seed.") from None
            if not 0 <= seed < 2 ** 31:
                raise BadRequest("Invalid seed.")
            return self._json(200, session.reset_model(seed))
        raise BadRequest("Unknown route.", 404)

    def _static(self, path):
        rel = unquote(path).lstrip("/") or "index.html"
        if "\\" in rel or "\x00" in rel or rel.startswith("."):
            raise BadRequest("File not found.", 404)
        root = WEB_DIR.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise BadRequest("File not found.", 404) from None
        content_type = CONTENT_TYPES.get(target.suffix.lower())
        if not target.is_file() or content_type is None:
            raise BadRequest("File not found.", 404)
        self._send(200, target.read_bytes(), content_type)

    # ------------------------------------------------------------------ Server-Sent Events
    def _sse(self, event, payload):
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream(self):
        session = self.server.session
        q = session.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self._common_headers()
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b"retry: 3000\n\n")
            snapshot = dict(session.trainer.status_json(), snapshot=True)
            self._sse("progress", json.dumps(snapshot, ensure_ascii=False))
            while not self.server.shutting_down.is_set():
                try:
                    event, payload = q.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                self._sse(event, payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            session.hub.unsubscribe(q)


def create_server(session, port, log=print):
    """Bind 127.0.0.1:<port> (raises OSError when the port is busy)."""
    return WorkshopServer(port, session, log=log)
