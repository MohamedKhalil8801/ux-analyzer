"""Loopback static server for viewing a rendered report over HTTP.

The report's live sidecar views fetch ``ux-audit.json`` / ``pagespeed.json``
at page load. Browsers block that fetch for ``file://`` URLs, so reports
that want live updates must be served over HTTP. This is deliberately
dependency-free (stdlib ``http.server``) and loopback-only (127.0.0.1).
"""

from __future__ import annotations

import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _loopback_port(preferred: int | None) -> int:
    if preferred is not None:
        return preferred
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve_report(
    directory: Path,
    *,
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Serve ``directory`` over loopback HTTP until interrupted.

    Blocks the caller; Ctrl+C / SIGINT ends the server.
    """

    def handler_factory(*args: object, **kwargs: object) -> SimpleHTTPRequestHandler:
        return SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)

    serve_port = _loopback_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", serve_port), handler_factory)
    url = f"http://127.0.0.1:{serve_port}/report.html"
    print(f"serving report directory: {directory}")
    print(f"open: {url}")
    print("press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
