// Debug: run the EXACT published page script on a URL and dump gradient evidence
// plus a manual recount with the same regex, plus visibleCount.
// Usage: node debug_script.mjs [--nojs] <url>
import { chromium } from 'playwright';
import {
  createColorHelpers,
  createVisibilityHelpers,
  isSlopFont,
  isAccentSerif,
  SLOP_FONT_PREFIXES,
  ACCENT_SERIF_PREFIXES,
} from 'slop-detect-core';

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
  const script = `(() => {
    ${createColorHelpers.toString()}
    ${createVisibilityHelpers.toString()}
    ${isSlopFont.toString()}
    ${isAccentSerif.toString()}
    const SLOP_FONT_PREFIXES = ${JSON.stringify(SLOP_FONT_PREFIXES)};
    const ACCENT_SERIF_PREFIXES = ${JSON.stringify(ACCENT_SERIF_PREFIXES)};
    const colorHelpers = createColorHelpers();
    const visHelpers = createVisibilityHelpers();
    const visible = visHelpers.getVisible(document.body, 4000);
    let n = 0;
    const grads = [];
    for (const el of visible) {
      const cs = getComputedStyle(el);
      const bgImg = cs.backgroundImage || '';
      if (/gradient\\(/.test(bgImg)) {
        const rgba = bgImg.match(/rgba?\\(\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+(?:\\s*,\\s*([\\d.]+))?\\s*\\)/g);
        let hasOpaqueStop = !rgba;
        if (rgba) {
          for (const r of rgba) {
            const a = r.match(/,\\s*([\\d.]+)\\s*\\)/);
            if (!a || parseFloat(a[1]) > 0.05) { hasOpaqueStop = true; break; }
          }
        }
        if (hasOpaqueStop) {
          n++;
          if (grads.length < 30) grads.push({ tag: el.tagName, cls: (el.className || '').toString().slice(0, 40), bg: bgImg.slice(0, 50) });
        }
      }
    }
    return { visibleCount: visible.length, bgElements: n, grads, title: document.title };
  })()`;
  const data = await page.evaluate(script);
  console.log(JSON.stringify(data, null, 1));
} finally {
  await browser.close();
}
