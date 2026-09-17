"""Dogfood run: analyze uxa's own generated report with uxa itself.

Serves reports/exploration-generated over loopback HTTPS (self-signed cert,
the same one Chromium accepts via --allow-insecure-localhost), then runs:

    uxa run   benchmarks/report-self/project.yaml  -> reports/report-self/
    uxa report reports/report-self/                 -> reports/report-self/report.html

Throwaway tool; the browsers and server live only for the duration of the run.
"""
import os
import ssl
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO, "reports", "exploration-generated")
TLS_DIR = os.path.join(REPO, "scripts", "_tls")
PORT = 8791
OUT = os.path.join(REPO, "reports", "report-self")
PROJECT = os.path.join(REPO, "benchmarks", "report-self", "project.yaml")
LOG = os.path.join(REPO, "scripts", "_self_report_run.log")


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def start_server() -> ThreadingHTTPServer:
    handler = lambda *a, **k: Quiet(*a, directory=REPORT_DIR, **k)  # noqa: E731
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(
        os.path.join(TLS_DIR, "cert.pem"), os.path.join(TLS_DIR, "key.pem")
    )
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(LOG, "a", encoding="utf-8") as log:
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"subprocess failed with exit code {code}: {cmd}")
    return code


def main() -> int:
    start_server()
    print(f"serving {REPORT_DIR} over https://127.0.0.1:{PORT}/report.html", flush=True)
    time.sleep(0.4)
    env = dict(os.environ)
    # Node-side TLS (Playwright route.fetch / APIRequestContext) must trust the
    # self-signed loopback cert; Chromium accepts it via --allow-insecure-localhost.
    env.setdefault("NODE_EXTRA_CA_CERTS", os.path.join(TLS_DIR, "cert.pem"))

    os.makedirs(OUT, exist_ok=True)
    run(
        [
            "uv", "run", "uxa", "run", PROJECT,
            "--experiment", "self-review",
            "--output", OUT,
            "--workers", os.environ.get("UXA_SELF_WORKERS", "2"),
            "--yes",
        ],
        env,
    )
    run(
        ["uv", "run", "uxa", "report", OUT, "--output", os.path.join(OUT, "report.html")],
        env,
    )
    print("\nDONE. report: ", os.path.join(OUT, "report.html"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())