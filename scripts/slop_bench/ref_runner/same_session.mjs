// Same-session A/B: evaluate the published detector page script AND our
// snapshot capture in ONE page load, so both sides see the identical DOM.
// Usage: node same_session.mjs <url> <out.json>
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';
import {
  PATTERNS,
  createColorHelpers,
  createVisibilityHelpers,
  isSlopFont,
  isAccentSerif,
  SLOP_FONT_PREFIXES,
  ACCENT_SERIF_PREFIXES,
  extractTextContext,
} from 'slop-detect-core';

const STYLE_PROPS = [
  'display','position','flex-direction','justify-content','align-items','gap',
  'row-gap','column-gap','grid-template-columns','font-family','font-size',
  'font-weight','font-style','line-height','letter-spacing','text-transform',
  'text-align','text-decoration-line','color','background-color','background-image',
  'background-clip','backdrop-filter','-webkit-backdrop-filter','filter',
  'margin-top','margin-right','margin-bottom','margin-left','padding-top',
  'padding-right','padding-bottom','padding-left','border-top-width',
  'border-right-width','border-bottom-width','border-left-width',
  'border-top-color','border-right-color','border-bottom-color','border-left-color',
  'border-radius','opacity','width','height','box-shadow','grid-column',
  'transform','perspective',
];

function buildPageScript() {
  const patternCalls = PATTERNS.map(
    (p) => `
    try {
      signals[${JSON.stringify(p.id)}] = (${p.extract.toString()})(ctx);
    } catch (e) {
      signals[${JSON.stringify(p.id)}] = { triggered: false, error: e.message };
    }`
  ).join('\n');
  return `(() => {
    ${createColorHelpers.toString()}
    ${createVisibilityHelpers.toString()}
    ${isSlopFont.toString()}
    ${isAccentSerif.toString()}
    const SLOP_FONT_PREFIXES = ${JSON.stringify(SLOP_FONT_PREFIXES)};
    const ACCENT_SERIF_PREFIXES = ${JSON.stringify(ACCENT_SERIF_PREFIXES)};
    const colorHelpers = createColorHelpers();
    const visHelpers = createVisibilityHelpers();
    const visible = visHelpers.getVisible(document.body, 4000);
    let h1 = null;
    for (const el of document.querySelectorAll('h1')) {
      if (visHelpers.isVisible(el)) { h1 = el; break; }
    }
    const ctx = {
      visible, h1,
      parseColor: colorHelpers.parseColor,
      rgbToHsl: colorHelpers.rgbToHsl,
      isPurple: colorHelpers.isPurple,
      isDark: colorHelpers.isDark,
      isMidGrey: colorHelpers.isMidGrey,
      relativeLuminance: colorHelpers.relativeLuminance,
      contrastRatio: colorHelpers.contrastRatio,
      channelSpread: colorHelpers.channelSpread,
      effectiveBackground: colorHelpers.effectiveBackground,
      isSlopFont, isAccentSerif,
      SLOP_FONT_PREFIXES, ACCENT_SERIF_PREFIXES,
      inHero: visHelpers.inHero
    };
    const signals = {};
    ${patternCalls}
    return { signals, visibleCount: visible.length, h1Text: h1 ? h1.textContent.trim().slice(0, 120) : null };
  })();`;
}

const SNAPSHOT_JS = `(props) => {
  const root = document.querySelector('[data-uxa-snapshot-root]') || document.body;
  const nodes = [];
  const walk = (el, parentIdx, depth) => {
    if (nodes.length > 8000) return;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const idx = nodes.length;
    const styles = {};
    for (const p of props) { styles[p] = cs.getPropertyValue(p); }
    let ownText = '';
    for (const n of el.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) ownText += n.textContent;
    }
    nodes.push({i: idx, parent: parentIdx, depth, tag: el.tagName.toLowerCase(), cls: (typeof el.className === 'string') ? el.className : '', id: el.id || '', text: ownText.trim().slice(0, 300), box: {x: Math.round(r.x * 100) / 100, y: Math.round(r.y * 100) / 100, w: Math.round(r.width * 100) / 100, h: Math.round(r.height * 100) / 100}, styles});
    for (const c of el.children) walk(c, idx, depth + 1);
  };
  walk(root, -1, 0);
  const rb = document.querySelector('[data-uxa-snapshot-root], body').getBoundingClientRect();
  return { rootBox: {x: rb.x, y: rb.y, w: rb.width, h: rb.height}, nodes };
}`;

const META_JS = `() => {
  function vt(root){ if(!root) return ''; var t = root.innerText != null ? root.innerText : root.textContent; return (t||'').replace(/\\u00AD/g,''); }
  var main = document.querySelector('main, article, [role="main"]') || document.body;
  var clone = main.cloneNode(true);
  var strip = clone.querySelectorAll('nav, footer, header, script, style, noscript, svg, code, pre, [aria-hidden="true"]');
  for (var i=0;i<strip.length;i++){ if (strip[i].parentNode) strip[i].parentNode.removeChild(strip[i]); }
  var text = vt(clone).trim();
  var headings=[]; var hs=clone.querySelectorAll('h1,h2,h3,h4,li,dt');
  for (var j=0;j<hs.length && headings.length<200;j++){ var ht=(hs[j].innerText||hs[j].textContent||'').trim(); if (ht) headings.push(ht.slice(0,200)); }
  var paragraphs=[]; var ps=clone.querySelectorAll('p');
  for (var k=0;k<ps.length && paragraphs.length<200;k++){ var pt=(ps[k].innerText||ps[k].textContent||'').trim(); if (pt) paragraphs.push(pt.slice(0,400)); }
  var words = text ? text.split(/\\s+/).filter(Boolean) : [];
  var centerEl = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
  return { viewport:{w:window.innerWidth,h:window.innerHeight}, docHeight:document.documentElement.scrollHeight, scrollY:window.scrollY,
    surface:{ htmlBg:getComputedStyle(document.documentElement).backgroundColor, bodyBg:getComputedStyle(document.body).backgroundColor, centerBg:centerEl ? getComputedStyle(centerEl).backgroundColor : '' },
    textContext:{ text:text.slice(0,200000), headings:headings, paragraphs:paragraphs, wordCount:words.length } };
}`;

const url = process.argv[2];
const outPath = process.argv[3];
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
  const oracle = await page.evaluate(buildPageScript());
  const snapshot = await page.evaluate(`(${SNAPSHOT_JS})(${JSON.stringify(STYLE_PROPS)})`);
  const meta = await page.evaluate(`(${META_JS})()`);
  const cardProbe = await page.evaluate(`(() => {
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
    const out = [];
    for (const el of document.querySelectorAll('[class*="chat-prompt-bg"]')) {
      const chain = [];
      let anc = el.parentElement;
      let g = 0;
      while (anc && g++ < 25) {
        chain.push({ i: g, cls: (anc.getAttribute('class') || anc.tagName).slice(0, 50), cardLike: isCardLike(anc), shadow: (getComputedStyle(anc).boxShadow || '').slice(0, 30), bTop: getComputedStyle(anc).borderTopWidth, radius: getComputedStyle(anc).borderRadius, bg: getComputedStyle(anc).backgroundColor });
        anc = anc.parentElement;
      }
      out.push({ self: (el.getAttribute('class') || '').slice(0, 40), selfCardLike: isCardLike(el), chain });
    }
    return out;
  })()`);
  writeFileSync(outPath, JSON.stringify({ oracle, snapshot, meta, cardProbe }));
  console.log(`written ${outPath} (${oracle.visibleCount} visible, ${snapshot.nodes.length} nodes)`);
} finally {
  await browser.close();
}
