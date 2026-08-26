// Debug: replicate nested_cards on a URL; dump card-like elements + nested verdicts.
// Usage: node debug_cards.mjs <url>
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
    const isVisible = (el) => {
      if (!el) return false;
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
      if (parseFloat(cs.opacity) === 0) return false;
      const r = el.getBoundingClientRect();
      return r.width >= 4 && r.height >= 4;
    };
    const SKIP = /^(input|select|textarea|img|video|canvas|picture|pre|code|svg|button|a|nav|li)$/i;
    function isCardLike(el) {
      const tag = el.tagName.toLowerCase();
      if (SKIP.test(tag)) return false;
      const cs = getComputedStyle(el);
      if (cs.position === 'absolute' || cs.position === 'fixed') return false;
      const cls = (el.getAttribute('class') || '').toLowerCase();
      if (/(dropdown|popover|tooltip|menu|modal|dialog|overlay)/.test(cls)) return false;
      if ((el.textContent || '').trim().length < 10) return false;
      const r = el.getBoundingClientRect();
      if (r.width < 50 || r.height < 30) return false;
      const hasShadow = cs.boxShadow && cs.boxShadow !== 'none';
      const hasBorder = parseFloat(cs.borderTopWidth) > 0 || parseFloat(cs.borderLeftWidth) > 0 || /\\bborder\\b/.test(cls);
      const radius = parseFloat(cs.borderRadius) || 0;
      const hasRadius = radius > 0;
      const bg = cs.backgroundColor;
      const hasBg = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
      return (hasShadow || hasBorder) && (hasRadius || hasBg);
    }
    const visible = [];
    const all = document.body.querySelectorAll('*');
    for (const el of all) { if (isVisible(el)) visible.push(el); }
    const cards = [];
    for (const el of visible) { try { if (isCardLike(el)) cards.push(el); } catch {} }
    const cardSet = new Set(cards);
    const nestedList = [];
    for (const el of cards) {
      let anc = el.parentElement, hasCardAncestor = false;
      let guard = 0;
      while (anc && guard++ < 30) { if (cardSet.has(anc)) { hasCardAncestor = true; break; } anc = anc.parentElement; }
      if (!hasCardAncestor) continue;
      let containsCard = false;
      for (const other of cards) { if (other !== el && el.contains(other)) { containsCard = true; break; } }
      if (containsCard) continue;
      const a = el.parentElement, g = 0;
      let transformed = false;
      let aa = el.parentElement, gg = 0;
      while (aa && gg++ < 20) { const cs = getComputedStyle(aa); if (cs.transform !== 'none' || cs.perspective !== 'none') { transformed = true; break; } aa = aa.parentElement; }
      if (transformed) continue;
      nestedList.push({ cls: (el.getAttribute('class') || el.tagName.toLowerCase()).slice(0, 45), w: Math.round(el.getBoundingClientRect().width) });
    }
    // Also list ALL card-like elements with their nearest ancestor card info
    const cardInfo = cards.slice(0, 40).map((el) => {
      let anc = el.parentElement, hasAnc = false, guard = 0;
      while (anc && guard++ < 30) { if (cardSet.has(anc)) { hasAnc = true; break; } anc = anc.parentElement; }
      return { cls: (el.getAttribute('class') || el.tagName.toLowerCase()).slice(0, 40), w: Math.round(el.getBoundingClientRect().width), nestedCandidate: hasAnc };
    });
    return { cardsTotal: cards.length, nested: nestedList, nestedCount: nestedList.length, cardInfo: cardInfo.slice(0, 25) };
  })()`);
  console.log(JSON.stringify(data, null, 1));
} finally {
  await browser.close();
}
