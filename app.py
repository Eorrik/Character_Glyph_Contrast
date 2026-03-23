from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

ROOT: Final = Path(__file__).resolve().parent
HOST: Final = "0.0.0.0"
DEFAULT_PORT: Final = 8050


def build_index_html() -> str:
    template_path = ROOT / "static" / "index.html"
    return template_path.read_text(encoding="utf-8")


class GlyphRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_index()
            return
        if parsed.path == "/config.json":
            self._serve_config()
            return
        return super().do_GET()

    def _serve_index(self) -> None:
        content = build_index_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_config(self) -> None:
        payload = {
            "expertImage": "/expert.jpg",
            "userImage": "/user.jpg",
            "canvasSize": 256,
            "padding": 12,
        }
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int) -> None:
    handler = functools.partial(GlyphRequestHandler, directory=str(ROOT))
    with socketserver.ThreadingTCPServer((HOST, port), handler) as httpd:
        print(f"Character Glyph Contrast app running at http://127.0.0.1:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Character Glyph Contrast web application.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    run_server(args.port)
