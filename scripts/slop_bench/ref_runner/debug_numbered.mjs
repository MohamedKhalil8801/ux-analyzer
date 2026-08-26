// Debug: dump the elements the published numbered_steps pattern matches on a URL.
// Usage: node debug_numbered.mjs <url>
import { chromium } from 'playwright';

const url = process.argv[2];
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  userAgent: 'Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)',
  deviceScaleFactor: 1,
});
const page = await context.newPage();
try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
  await page.waitForTimeout(500);
  const data = await page.evaluate(`(() => {
    const numberPat = /^(?:step\\s*)?(\\d{1,2})(?:[.):]|\\s*[\\u2014-])?\\s*/i;
    const isVisible = (el) => {
      if (!el) return false;
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
      if (parseFloat(cs.opacity) === 0) return false;
      const r = el.getBoundingClientRect();
      return r.width >= 4 && r.height >= 4;
    };
    const res = [];
    const all = document.body.querySelectorAll('*');
    for (const el of all) {
      if (!isVisible(el)) continue;
      const txt = (el.textContent || '').trim();
      const m = txt.match(numberPat);
      if (!m) continue;
      const n = parseInt(m[1], 10);
      if (n < 1 || n > 9) continue;
      const r = el.getBoundingClientRect();
      res.push({ n, txt: txt.slice(0, 50), tag: el.tagName, cls: (el.className || '').toString().slice(0, 60), w: Math.round(r.width), h: Math.round(r.height), top: Math.round(r.top), parent: el.parentElement ? el.parentElement.tagName + '.' + ((el.parentElement.className||'').toString().slice(0,40)) : '' });
    }
    // Group by parent like the pattern does
    const parents = new Map();
    for (const el of all) {
      if (!isVisible(el)) continue;
      const txt = (el.textContent || '').trim();
      const m = txt.match(numberPat);
      if (!m) continue;
      const n = parseInt(m[1], 10);
      if (n < 1 || n > 9) continue;
      if (!el.parentElement) continue;
      const key = el.parentElement;
      if (!parents.has(key)) parents.set(key, new Set());
      parents.get(key).add(n);
    }
    const groups = [];
    for (const [k, set] of parents) {
      if (set.has(1) && set.has(2) && set.has(3)) {
        let run = 3 + (set.has(4) ? 1 : 0) + (set.has(5) ? 1 : 0);
        groups.push({ run, nums: [...set].sort(), parentCls: (k.className||'').toString().slice(0,60), parentTag: k.tagName });
      }
    }
    return { matches: res.slice(0, 60), groups };
  })()`);
  console.log(JSON.stringify(data, null, 1));
} finally {
  await browser.close();
}
