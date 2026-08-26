"""Harvest UEye challenge designs into standalone fixtures + evidence.

For each of the 15 UEye modules (https://designcourse.com/app/course/ueye):
- locate the design root (`.module-content main`)
- save a standalone design-only HTML fixture (all page CSS inlined)
- save a computed-style snapshot tree for every element
- save a clipped screenshot of the design

Ground-truth answers come from the DOM: each option label carries class
`correct` when that fundamental was applied incorrectly (selecting it earns a
point) and `incorrect` otherwise. The answer key is written to answers.json
and is used ONLY for evaluation, never by the detectors.

Usage:
    python benchmarks/ueye/tools/harvest_ueye.py [--out benchmarks/ueye]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

SLUGS = [
    "activity-feed",
    "feature-blip",
    "simple-login-form",
    "dropdown-menu",
    "url-shortener",
    "mobile-app-onboarding",
    "ueye-footer-design",
    "reporting-statistics",
    "user-testimonials",
    "pricing-comparison",
    "sports-stats",
    "food-ingredients",
    "toggle-settings",
    "news-listings",
    "user-support-popup",
]

BASE = "https://designcourse.com/app/course/ueye/module/{slug}"

# Computed styles captured per element (visual fundamentals need these).
_STYLE_PROPS = [
    "display",
    "position",
    "flex-direction",
    "justify-content",
    "align-items",
    "gap",
    "row-gap",
    "column-gap",
    "grid-template-columns",
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "line-height",
    "letter-spacing",
    "text-transform",
    "text-align",
    "text-decoration-line",
    "color",
    "background-color",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "border-top-width",
    "border-right-width",
    "border-bottom-width",
    "border-left-width",
    "border-top-color",
    "border-radius",
    "opacity",
    "width",
    "height",
    "box-shadow",
]

_SNAPSHOT_JS = """
(props) => {
  const mcs = [...document.querySelectorAll('.module-content')];
  const mc = mcs.find(m => m.firstElementChild && m.getBoundingClientRect().width > 0);
  if (!mc) return null;
  const root = mc.firstElementChild;
  if (!root) return null;
  const nodes = [];
  const walk = (el, parentIdx, depth) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const idx = nodes.length;
    const styles = {};
    for (const p of props) { styles[p] = cs.getPropertyValue(p); }
    let ownText = '';
    for (const n of el.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) ownText += n.textContent;
    }
    nodes.push({
      i: idx,
      parent: parentIdx,
      depth,
      tag: el.tagName.toLowerCase(),
      cls: (typeof el.className === 'string') ? el.className : '',
      id: el.id || '',
      text: ownText.trim().slice(0, 300),
      box: {
        x: Math.round(r.x * 100) / 100,
        y: Math.round(r.y * 100) / 100,
        w: Math.round(r.width * 100) / 100,
        h: Math.round(r.height * 100) / 100,
      },
      styles,
    });
    for (const c of el.children) walk(c, idx, depth + 1);
  };
  const rr = root.getBoundingClientRect();
  walk(root, -1, 0);
  return {
    rootBox: { x: rr.x, y: rr.y, w: rr.width, h: rr.height },
    rootClass: root.className,
    nodes,
  };
}
"""

_ANSWERS_JS = """
() => {
  const out = {};
  for (const lb of document.querySelectorAll('.check-label')) {
    const r = lb.getBoundingClientRect();
    if (r.width <= 0) continue;  // skip hidden duplicates
    const text = (lb.innerText || '').trim();
    if (!text) continue;
    out[text] = lb.className.includes('incorrect') ? 'misapplied-choice'
        : (lb.className.includes('correct') ? 'correct-choice' : 'unknown');
  }
  return out;
}
"""

_CSS_JS = """
() => {
  let css = '';
  for (const sheet of document.styleSheets) {
    try {
      for (const rule of sheet.cssRules) css += rule.cssText + '\\n';
    } catch (e) { /* cross-origin: skip */ }
  }
  return css;
}
"""


def _load_module(pg, slug: str) -> None:
    """Navigate to a module and wait for the design to render; retry on flake."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            pg.goto(
                BASE.format(slug=slug),
                wait_until="domcontentloaded",
                timeout=45000,
            )
            pg.wait_for_selector(".module-content", state="attached", timeout=30000)
            pg.wait_for_timeout(2000)
            ready = pg.evaluate(
                """() => {
                  const mcs = [...document.querySelectorAll('.module-content')];
                  const vis = mcs.find(m => m.firstElementChild &&
                    m.getBoundingClientRect().width > 0);
                  return !!vis;
                }"""
            )
            if not ready:
                raise RuntimeError("no visible design root")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            pg.wait_for_timeout(1500 * (attempt + 1))
    raise RuntimeError(f"could not load {slug}: {last}")


def harvest_slug(pg, slug: str, out_dir: pathlib.Path) -> dict:
    _load_module(pg, slug)

    snap = pg.evaluate(_SNAPSHOT_JS, _STYLE_PROPS)
    if snap is None:
        raise RuntimeError(f"design root not found for {slug}")
    answers = pg.evaluate(_ANSWERS_JS)
    css = pg.evaluate(_CSS_JS)
    html = pg.evaluate(
        """() => {
          const mcs = [...document.querySelectorAll('.module-content')];
          const mc = mcs.find(m => m.firstElementChild &&
            m.getBoundingClientRect().width > 0);
          if (!mc) return null;
          const root = mc.firstElementChild;
          const r = root.getBoundingClientRect();
          root.setAttribute('data-ueye-w', String(Math.round(r.width)));
          return root.outerHTML;
        }"""
    )
    if not html:
        raise RuntimeError("design html missing")
    snap_w = snap["rootBox"]["w"]
    fixture = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>ueye-fixture:{slug}</title>\n"
        "<style>\nhtml,body{margin:0;padding:24px;background:#ffffff;}\n"
        f".__design-root{{width:{snap_w:.0f}px;}}\n"
        ".__design-root > *{width:100%;}\n"
        f"{css}\n</style>\n</head>\n<body>\n"
        f"<div class='__design-root'>{html}</div>\n</body>\n</html>\n"
    )

    (out_dir / "fixtures" / f"{slug}.html").write_text(fixture, encoding="utf-8")
    (out_dir / "evidence" / f"{slug}.styles.json").write_text(
        json.dumps(snap, indent=1), encoding="utf-8"
    )
    (out_dir / "evidence" / f"{slug}.design.html").write_text(html, encoding="utf-8")

    root_handle = pg.evaluate_handle(
        """() => {
          const mcs = [...document.querySelectorAll('.module-content')];
          const mc = mcs.find(m => m.firstElementChild &&
            m.getBoundingClientRect().width > 0);
          return mc ? mc.firstElementChild : null;
        }"""
    )
    root_handle.as_element().screenshot(path=str(out_dir / "evidence" / f"{slug}.png"))

    # key: fundamental -> True when misapplied in the design (the earn-a-point choice)
    label_to_fundamental = {
        "white space": "white-space",
        "contrast": "contrast",
        "color": "color",
        "typography": "typography",
        "scale": "scale",
        "alignment": "alignment",
        "visual hierarchy": "visual-hierarchy",
        "nothing is wrong": "nothing-wrong",
    }
    key: dict[str, bool] = {}
    for label, choice in answers.items():
        fund = label_to_fundamental.get(label.strip().lower())
        if fund is None:
            continue
        key[fund] = choice == "correct-choice"
    return {"answers_dom": answers, "key": key}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/ueye")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    for sub in ("fixtures", "evidence", "tools"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    all_keys: dict[str, dict] = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        for slug in SLUGS:
            try:
                res = harvest_slug(pg, slug, out_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {slug}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            all_keys[slug] = res["key"]
            misapplied = sorted(k for k, v in res["key"].items() if v and k != "nothing-wrong")
            clean = res["key"].get("nothing-wrong") is True
            print(f"OK {slug}: misapplied={misapplied}{' CLEAN' if clean else ''}")
        b.close()

    (out_dir / "answers.json").write_text(
        json.dumps(all_keys, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {len(all_keys)}/15 keys -> {out_dir / 'answers.json'}")
    total_pts = sum(1 for k in all_keys.values() for v in k.values() if v)
    print(f"total ground-truth selections (expect ~28): {total_pts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
