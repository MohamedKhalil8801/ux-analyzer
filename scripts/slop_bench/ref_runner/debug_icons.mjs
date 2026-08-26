// Debug: replicate icon_card_grid grouping on a URL and dump groups.
// Usage: node debug_icons.mjs [--nojs] <url>
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const noJs = args[0] === '--nojs';
const url = noJs ? args[1] : args[0];
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  userAgent: 'Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)',
  deviceScaleFactor: 1,
  javaScriptEnabled: !noJs,
});
const page = await context.newPage();
try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
  await page.waitForTimeout(500);
  const data = await page.evaluate(`(() => {
    const isVisible = (el) => {
      if (!el) return false;
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
      if (parseFloat(cs.opacity) === 0) return false;
      const r = el.getBoundingClientRect();
      return r.width >= 4 && r.height >= 4;
    };
    const groups = new Map();
    const all = document.body.querySelectorAll('*');
    for (const el of all) {
      if (!isVisible(el)) continue;
      const parent = el.parentElement;
      if (!parent) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 150 || r.width > 600 || r.height < 100 || r.height > 600) continue;
      const icon = el.querySelector(':scope > svg, :scope > img, :scope > div > svg, :scope > div > img');
      if (!icon) continue;
      const ir = icon.getBoundingClientRect();
      if (ir.width > 80 || ir.height > 80) continue;
      if (ir.top > r.top + r.height * 0.4) continue;
      const key = parent.tagName + ':' + Math.round(r.width / 20);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ tag: el.tagName, cls: (el.className || '').toString().slice(0, 45), w: Math.round(r.width), iconTag: icon.tagName, iconCls: (icon.className || '').toString().slice(0, 30) });
    }
    let max = 0, bestKey = '';
    for (const [k, v] of groups) { if (v.length > max) { max = v.length; bestKey = k; } }
    return { maxGroupSize: max, bestKey, best: bestKey ? groups.get(bestKey).slice(0, 8) : [] };
  })()`);
  console.log(JSON.stringify(data, null, 1));
} finally {
  await browser.close();
}
