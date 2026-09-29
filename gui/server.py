"""Loopback-only standard-library HTTP server for the JARVIS GUI prototype."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gui.application import JarvisApplication

STATIC = Path(__file__).with_name("static").resolve()
MAX_BODY = 64 * 1024


def make_server(app: JarvisApplication | None = None, host: str = "127.0.0.1", port: int = 8765):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The GUI server only binds to loopback.")
    application = app or JarvisApplication()

    class Handler(BaseHTTPRequestHandler):
        server_version = "JARVISLocal/1"

        def log_message(self, format, *args):
            # Avoid writing message contents or task data to the terminal log.
            print(f"[JARVIS GUI] {self.address_string()} {self.command} {self.path.split('?')[0]}")

        def _write(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value):
            self._write(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _mode(self, query):
            mode = query.get("mode", ["demo"])[0]
            if mode not in {"demo", "real"}:
                raise ValueError("Unknown mode.")
            return mode

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._write(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/static/app.js":
                self._write(200, (STATIC / "app.js").read_bytes(), "text/javascript; charset=utf-8")
                return
            if parsed.path == "/static/style.css":
                self._write(200, (STATIC / "style.css").read_bytes(), "text/css; charset=utf-8")
                return
            query = parse_qs(parsed.query)
            try:
                mode = self._mode(query)
                if parsed.path == "/api/conversation":
                    self._json(200, application.default_conversation(mode))
                elif parsed.path == "/api/snapshot":
                    self._json(200, application.snapshot(mode))
                elif parsed.path == "/api/events":
                    after = int(query.get("after", ["0"])[0])
                    self._json(200, application.events(mode, after, query.get("epoch", [None])[0]))
                elif parsed.path == "/api/status":
                    self._json(200, {"status": application.current_status(mode, query.get("task_id", [None])[0])})
                else:
                    self._json(404, {"error": "Not found."})
            except (ValueError, OSError) as error:
                self._json(400, {"error": str(error)})

        def do_POST(self):
            if not self._valid_local_post():
                self._json(403, {"error": "Local request origin was rejected."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length < 1 or length > MAX_BODY:
                self._json(413, {"error": "Request body is missing or too large."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required.")
                mode = payload.get("mode", "demo")
                if mode not in {"demo", "real"}:
                    raise ValueError("Unknown mode.")
                path = urlparse(self.path).path
                if path == "/api/conversations":
                    self._json(200, {"conversation_id": application.manager(mode).create_conversation()})
                elif path == "/api/messages":
                    result = application.submit(mode, payload.get("text", ""), payload.get("conversation_id"), payload.get("priority", "normal"))
                    self._json(200 if result.get("accepted") else 400, result)
                elif path.startswith("/api/tasks/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 4 or parts[1] != "tasks":
                        self._json(404, {"error": "Not found."})
                        return
                    result = application.control(mode, parts[2], parts[3], payload)
                    self._json(200 if result.get("accepted") else 400, result)
                elif path == "/api/math-notebooks":
                    result = application.math_notebook_new(payload.get("notebook_id"))
                    self._json(200, result)
                elif path.startswith("/api/math-notebooks/"):
                    parts = path.strip("/").split("/")
                    if len(parts) < 3:
                        self._json(404, {"error": "Not found."})
                        return
                    notebook_id = parts[2]
                    action = parts[3] if len(parts) > 3 else ""
                    
                    if action == "cells" and self.command == "POST":
                        # Add cell
                        result = application.math_notebook_add_cell(
                            notebook_id, 
                            payload.get("cell_type", "code"), 
                            payload.get("source", ""), 
                            payload.get("index")
                        )
                        self._json(200, result)
                    elif action == "cells" and self.command == "GET":
                        # Get notebook
                        result = application.math_notebook_get(notebook_id)
                        self._json(200, result)
                    elif action == "eval" and self.command == "POST":
                        # Evaluate cell
                        cell_id = payload.get("cell_id")
                        if not cell_id:
                            self._json(400, {"error": "cell_id required"})
                            return
                        result = application.math_notebook_eval_cell(notebook_id, cell_id)
                        self._json(200, result)
                    elif action == "eval-all" and self.command == "POST":
                        # Evaluate all cells
                        result = application.math_notebook_eval_all(notebook_id)
                        self._json(200, result)
                    elif action == "save" and self.command == "POST":
                        result = application.math_notebook_save(notebook_id, payload.get("path", ""))
                        self._json(200, result)
                    elif action == "load" and self.command == "POST":
                        result = application.math_notebook_load(notebook_id, payload.get("path", ""))
                        self._json(200, result)
                    elif action and self.command == "DELETE":
                        # Delete cell
                        result = application.math_notebook_delete_cell(notebook_id, action)
                        self._json(200, result)
                    else:
                        self._json(404, {"error": "Not found."})
                else:
                    self._json(404, {"error": "Not found."})
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                self._json(400, {"error": str(error)})

        def _valid_local_post(self):
            host_header = self.headers.get("Host", "").lower()
            permitted_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host_header not in permitted_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlparse(origin)
                return parsed_origin.scheme == "http" and parsed_origin.netloc.lower() == host_header
            return True

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.application = application
    return server


def main():
    server = make_server()
    print(f"JARVIS GUI listening at http://127.0.0.1:{server.server_port} (demo mode by default)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.application.close()


if __name__ == "__main__":
    main()
