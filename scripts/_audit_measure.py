"""Measure the generated report (contrast, layout, structure) — throwaway audit tool."""
import http.server
import json
import os
import re
import socketserver
import threading
import time

from playwright.sync_api import sync_playwright

REPORT_DIR = os.path.abspath("reports/exploration-generated")
PORT = 8766


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    os.chdir(REPORT_DIR)
    with socketserver.TCPServer(("127.0.0.1", PORT), QuietHandler) as httpd:
        httpd.serve_forever()


def rel_lum(hexv):
    hexv = hexv.lstrip("#")
    r, g, b = (int(hexv[i:i + 2], 16) / 255 for i in (0, 2, 4))
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = rel_lum(a), rel_lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main():
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(0.4)

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # --- contrast table from CSS vars
        css = open(os.path.join(REPORT_DIR, "report.html"), encoding="utf-8").read()
        m = re.search(r":root\s*\{(.*?)\}", css, re.S)
        vars_ = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\));", m.group(1)))
        pairs = {
            "body ink/paper": (vars_.get("--ink"), vars_.get("--paper")),
            "ink-soft/paper": (vars_.get("--ink-soft"), vars_.get("--paper")),
            "ink-soft/paper-raised": (vars_.get("--ink-soft"), vars_.get("--paper-raised")),
            "ink-soft/folder": (vars_.get("--ink-soft"), vars_.get("--folder")),
            "stamp/paper-raised": (vars_.get("--stamp"), vars_.get("--paper-raised")),
            "stamp-deep/stamp-soft": (vars_.get("--stamp-deep"), vars_.get("--stamp-soft")),
            "verified/verified-soft": (vars_.get("--verified"), vars_.get("--verified-soft")),
            "flag/flag-soft": (vars_.get("--flag"), vars_.get("--flag-soft")),
            "archive/archive-soft": (vars_.get("--archive"), vars_.get("--archive-soft")),
            "stamp-deep/folder": (vars_.get("--stamp-deep"), vars_.get("--folder")),
            "rule-strong/paper": (vars_.get("--rule-strong"), vars_.get("--paper")),
            "white/stamp (btn)": ("#ffffff", vars_.get("--stamp")),
            "ink/paper-raised": (vars_.get("--ink"), vars_.get("--paper-raised")),
        }
        out["contrast"] = {k: round(contrast(*v), 2) for k, v in pairs.items() if v[0] and v[1]}

        for w, h in ((1440, 900), (390, 844)):
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(f"http://127.0.0.1:{PORT}/report.html", wait_until="networkidle")
            page.wait_for_timeout(600)
            view_out = {}
            for name, sel in [("overview", "#view-overview"), ("findings", "#view-findings"),
                              ("audit", "#view-audit"), ("performance", "#view-performance"),
                              ("redesign", "#view-redesign"), ("evidence", "#view-evidence")]:
                page.evaluate("document.querySelectorAll('.view').forEach(v => v.hidden=true)")
                page.evaluate(f"document.querySelector('{sel}').hidden=false")
                page.wait_for_timeout(120)
                metrics = page.evaluate("""(sel) => {
                    const el = document.querySelector(sel);
                    const doc = document.documentElement;
                    const hs = [];
                    el.querySelectorAll('h1,h2,h3,h4').forEach(x => {
                        const t = (x.textContent||'').trim().replace(/\\s+/g,' ');
                        if (t) hs.push(t.slice(0,90));
                    });
                    const imgs = [...el.querySelectorAll('img')];
                    const noAlt = imgs.filter(i => !(i.getAttribute('alt') !== null && i.getAttribute('alt') !== ''));
                    const links = [...el.querySelectorAll('a')];
                    const btns = [...el.querySelectorAll('button')];
                    const detailsOpen = [...el.querySelectorAll('details')].filter(d => d.open).length;
                    return {
                        scrollHeight: doc.scrollHeight,
                        innerHeight: window.innerHeight,
                        elHeight: el.scrollHeight,
                        headings: hs,
                        hCount: hs.length,
                        imgCount: imgs.length,
                        noAlt: noAlt.map(i => (i.src||'').split('/').pop().slice(0,40)),
                        linkCount: links.length,
                        btnCount: btns.length,
                        detailsTotal: el.querySelectorAll('details').length,
                        detailsOpen,
                        hasHScroll: doc.scrollWidth > doc.clientWidth + 2,
                        docScrollWidth: doc.scrollWidth, docClientWidth: doc.clientWidth,
                    };
                }""", sel)
                view_out[name] = metrics

            # un-hide all: find the actual first-viewport composition (overview)
            page.evaluate("document.querySelectorAll('.view').forEach(v => v.hidden=true)")
            page.evaluate("document.querySelector('#view-overview').hidden=false")
            first = page.evaluate("""() => {
                const el = document.querySelector('#view-overview');
                const pct = h => Math.round(h * 100 / window.innerHeight);
                const items = [];
                let prevTop = 0;
                el.querySelectorAll('*').forEach(n => {
                    if (n.children.length) return;
                    const r = n.getBoundingClientRect();
                    if (r.width === 0 || r.height === 0) return;
                    const t = (n.textContent||'').trim().replace(/\\s+/g,' ').slice(0,60);
                    if (!t) return;
                    if (r.bottom <= 0 || r.top >= window.innerHeight) return;
                    items.push({top: Math.round(r.top+r.scrollTop*0), bottom: Math.round(r.bottom), text: t});
                });
                return items.slice(0, 40);
            }""")
            view_out["_firstViewport"] = first
            out[f"layout_{w}x{h}"] = view_out
            page.close()
        browser.close()
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()