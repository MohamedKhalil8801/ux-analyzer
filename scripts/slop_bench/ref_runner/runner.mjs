// Reference oracle runner: drives the PUBLISHED @slop-detect/core 0.5.1
// engine (the exact code behind `npx slop-detect`), replicating the CLI's
// page-script assembly verbatim, and emits full JSON including the copy axis
// (which the published CLI drops from --json).
//
// Usage: node runner.mjs <url> [<url2> ...]

import { chromium } from 'playwright';
import {
  PATTERNS,
  createColorHelpers,
  createVisibilityHelpers,
  isSlopFont,
  isAccentSerif,
  SLOP_FONT_PREFIXES,
  ACCENT_SERIF_PREFIXES,
  scorePatterns,
  applyPreset,
  extractTextContext,
  scoreCopy,
  combineAxes,
} from 'slop-detect-core';

// Verbatim from packages/cli/src/detector.js (published CLI 0.5.2).
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

    const extractTextContext = ${extractTextContext.toString()};
    let textContext = null;
    try { textContext = extractTextContext(); } catch (e) { textContext = { error: e.message }; }

    return {
      title: document.title,
      url: location.href,
      viewport: { w: window.innerWidth, h: window.innerHeight },
      docHeight: document.documentElement.scrollHeight,
      h1Text: h1 ? h1.textContent.trim().slice(0, 120) : null,
      h1Font: h1 ? getComputedStyle(h1).fontFamily : null,
      visibleCount: visible.length,
      signals,
      textContext
    };
  })();`;
}

async function scanUrl(url, noJs) {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    userAgent: 'Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)',
    deviceScaleFactor: 1,
    javaScriptEnabled: !noJs,
  });
  const page = await context.newPage();
  let result;
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
    await page.waitForTimeout(500);
    const data = await page.evaluate(buildPageScript());
    const patterns = PATTERNS.map((p) => {
      const sig = (data.signals || {})[p.id] || { triggered: false };
      return {
        id: p.id,
        label: p.label,
        short: p.short,
        category: p.category,
        weight: p.weight,
        triggered: !!sig.triggered,
        evidence: sig,
      };
    });
    const scored = applyPreset(patterns, 'full');
    const scoring = scorePatterns(scored);
    const axes = {
      design: {
        axis: 'design',
        score: scoring.score,
        tier: scoring.tier,
        grade: scoring.grade,
        patternsFlagged: scoring.patternsFlagged,
        patternsTotal: scoring.patternsTotal,
        patterns: scored,
      },
      copy: scoreCopy(data.textContext || {}),
    };
    result = {
      url,
      finalUrl: data.url,
      title: data.title,
      h1: data.h1Text,
      h1Font: data.h1Font,
      preset: 'full',
      ...scoring,
      patterns: scored,
      axes,
      ...combineAxes({
        design: axes.design,
        copy: axes.copy,
      }),
    };
  } finally {
    await browser.close();
  }
  return result;
}

const urls = process.argv.slice(2);
const noJs = urls[0] === '--nojs';
const realUrls = noJs ? urls.slice(1) : urls;
for (const url of realUrls) {
  try {
    const r = await scanUrl(url, noJs);
    console.log(JSON.stringify(r));
  } catch (e) {
    console.log(JSON.stringify({ url, error: String((e && e.message) || e) }));
  }
}
