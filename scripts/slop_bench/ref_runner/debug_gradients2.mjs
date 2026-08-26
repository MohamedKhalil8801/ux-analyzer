// Debug: dump gradient elements + specific targets with computed styles.
// Usage: node debug_gradients2.mjs [--nojs] <url>
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
    const res = [];
    const all = document.body.querySelectorAll('*');
    for (const el of all) {
      if (!isVisible(el)) continue;
      const cs = getComputedStyle(el);
      const bgImg = cs.backgroundImage || '';
      if (!/gradient\\(/.test(bgImg)) continue;
      const r = el.getBoundingClientRect();
      res.push({ tag: el.tagName, cls: (el.className || '').toString().slice(0, 50), top: Math.round(r.top), w: Math.round(r.width), bg: bgImg.slice(0, 60) });
    }
    const targets = [];
    for (const el of document.querySelectorAll('h4.vertical, a.btn--card')) {
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      targets.push({ cls: (el.className || '').toString().slice(0, 55), display: cs.display, vis: cs.visibility, opacity: cs.opacity, bgImg: (cs.backgroundImage || '').slice(0, 60), w: Math.round(r.width), h: Math.round(r.height), visible: isVisible(el) });
    }
    return { res, targets };
  })()`);
  console.log(JSON.stringify(data, null, 1));
} finally {
  await browser.close();
}
