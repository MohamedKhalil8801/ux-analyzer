"""Screenshot the generated report for a UX review (throwaway tool)."""
import argparse
import http.server
import os
import socketserver
import threading
import time

from playwright.sync_api import sync_playwright

REPORT_DIR = os.path.abspath("reports/exploration-generated")
OUT_DIR = os.path.abspath("scripts/_report_shots")
PORT = 8765


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass


def serve():
    os.chdir(REPORT_DIR)
    handler = lambda *a, **k: QuietHandler(*a, directory=REPORT_DIR, **k)  # noqa: E731
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        httpd.serve_forever()


VIEWS = [
    ("overview", "#view-overview"),
    ("findings", "#view-findings"),
    ("audit", "#view-audit"),
    ("performance", "#view-performance"),
    ("redesign", "#view-redesign"),
    ("evidence", "#view-evidence"),
]


def main(full_page: bool, viewport_w: int, viewport_h: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(0.5)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})
        page.goto(f"http://127.0.0.1:{PORT}/report.html", wait_until="networkidle")
        page.wait_for_timeout(800)

        tag = f"{viewport_w}x{viewport_h}"
        # full-scroll capture of each view
        for name, sel in VIEWS:
            page.evaluate("document.querySelectorAll('.view').forEach(v => v.hidden = true)")
            page.evaluate(f"document.querySelector('{sel}').hidden = false")
            page.evaluate("document.querySelector('.view-rail a[href=\"" + sel + "\"]')?.click()")
            time.sleep(0.4)
            if full_page:
                shot = os.path.join(OUT_DIR, f"{tag}_{name}_full.png")
                page.locator(sel).first.screenshot(path=shot)
            else:
                shot = os.path.join(OUT_DIR, f"{tag}_{name}_vp.png")
                page.screenshot(path=shot)
            print("saved", shot)

        # also a blank-visibility check: what's actually visible in the overview first viewport
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-page", action="store_true")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()
    main(args.full_page, args.width, args.height)