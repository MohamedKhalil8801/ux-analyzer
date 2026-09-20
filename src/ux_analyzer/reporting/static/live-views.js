/* Live sidecar views.
 *
 * The report renders its Page checks and Page speed tabs from two JSON
 * sidecars (the audit and pagespeed JSON files). Instead of re-rendering
 * report.html every time one of those files changes, this script fetches
 * them at page load (no-store, so a refresh always picks up a regenerated
 * sidecar) and re-renders the two tabs client-side, mirroring the exact
 * markup the server template emits.
 *
 * When fetch is unavailable (typically file:// with no HTTP server) or the
 * sidecar is missing, the server-embedded content stays in place and a
 * small note explains why the tabs are not live.
 *
 * This file is inlined into both report.js and report-index.js; it is
 * self-contained and must not depend on either script's internals.
 */
(function () {
  "use strict";
  var configNode = document.getElementById("data-sources");
  if (!configNode) return;
  var CONFIG;
  try {
    CONFIG = JSON.parse(configNode.textContent || "{}") || {};
  } catch (error) {
    return;
  }
  var SOURCES = CONFIG.sources || {};
  if (!window.fetch) return;

  var SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"];
  var PSI_PASS_BUCKETS = [
    ["passed", "Passed"],
    ["not_applicable", "Not applicable"],
    ["manual", "Manual"],
    ["informative", "Informative"],
    ["error", "Error"]
  ];
  var AUDIT_SUMMARY =
    "Deterministic checks (GEO, meta/semantic HTML, performance, accessibility, " +
    "imagery, visual UI fundamentals, and the 27-rule AI-slop fingerprint with " +
    "the 9 copy-axis tells) run against each application start URL. These are " +
    "recorded page facts, not simulated-user findings.";
  var PSI_SUMMARY =
    "Complete PageSpeed Insights results for each application start URL, " +
    "fetched from the same API that powers pagespeed.web.dev. Category " +
    "scores, per-audit pass/fail results, and opportunity savings are " +
    "taken verbatim from the API response; the audit pass threshold is " +
    "the pagespeed.web.dev convention (score &lt; 0.9 = failed). " +
    '"View saved report" opens an offline replica of this exact ' +
    'recorded analysis with identical numbers; "Re-run on ' +
    'pagespeed.web.dev" starts a new Lighthouse run, and scores vary ' +
    "between runs.";

  function esc(value) {
    return String(value === undefined || value === null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function has(value) {
    return value !== undefined && value !== null && value !== "";
  }
  function titleCase(value) {
    return String(value || "").replace(/\b\w/g, function (ch) {
      return ch.toUpperCase();
    });
  }
  function truncate(value, max) {
    var s = String(value === undefined || value === null ? "" : value);
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
  }
  function numberValue(value) {
    return typeof value === "number" && isFinite(value) ? value : null;
  }
  function scoreClass(score) {
    if (score === null || score === undefined) return "bad";
    if (score >= 0.9) return "good";
    if (score >= 0.5) return "ok";
    return "bad";
  }
  function jsonPre(value) {
    return esc(
      JSON.stringify(value === undefined || value === null ? {} : value, null, 2)
    );
  }
  function fetchJson(path) {
    return window
      .fetch(path, { cache: "no-store", credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      });
  }
  function liveNoteHtml(path, off) {
    var text = off ? "bundled snapshot" : "live · " + String(path).split("/").pop();
    var title = off
      ? "Serving this page over HTTP lets the report fetch regenerated JSON sidecars automatically"
      : "Viewed from " + path + " at " + new Date().toLocaleTimeString();
    return (
      '<span class="live-note' +
      (off ? " live-note-off" : "") +
      '" title="' +
      esc(title) +
      '">' +
      esc(text) +
      "</span>"
    );
  }
  function markLive(section, path, off) {
    if (!section) return;
    var heading = section.querySelector(".report-section-heading");
    if (!heading) return;
    heading.insertAdjacentHTML("beforeend", liveNoteHtml(path, off));
  }
  function showLocalView(view) {
    if (!view) return;
    view.hidden = false;
    document.querySelectorAll(".view").forEach(function (item) {
      if (item !== view) item.hidden = true;
    });
    document.querySelectorAll(".view-rail a").forEach(function (tab) {
      if (tab.getAttribute("href") === "#" + view.id) {
        tab.setAttribute("aria-current", "page");
      } else {
        tab.removeAttribute("aria-current");
      }
    });
  }
  function ensureView(viewId, tabLabel, sectionHtml) {
    var existing = document.getElementById(viewId);
    if (existing) return existing;
    var rail = document.querySelector("nav.view-rail");
    var main = document.querySelector("main");
    if (!rail || !main) return null;
    var tab = document.createElement("a");
    tab.className = "view-tab";
    tab.href = "#" + viewId;
    tab.textContent = tabLabel;
    rail.appendChild(tab);
    tab.addEventListener("click", function () {
      showLocalView(document.getElementById(viewId));
    });
    var view = document.createElement("div");
    view.className = "view";
    view.id = viewId;
    view.hidden = true;
    view.innerHTML = sectionHtml;
    main.appendChild(view);
    if (window.location.hash === "#" + viewId) showLocalView(view);
    return view;
  }
  function installSectionHtml(sectionId, sectionHtml) {
    var section = document.getElementById(sectionId);
    if (section) {
      section.outerHTML = sectionHtml;
      return document.getElementById(sectionId);
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Page checks tab (audit sidecar)
  // ------------------------------------------------------------------
  function auditElementInfoHtml(evidence) {
    var selectors =
      evidence.element_selectors && evidence.element_selectors.length
        ? evidence.element_selectors
        : evidence.element_selector
        ? [evidence.element_selector]
        : [];
    var xpaths =
      evidence.element_xpaths && evidence.element_xpaths.length
        ? evidence.element_xpaths
        : evidence.element_xpath
        ? [evidence.element_xpath]
        : [];
    var boxes = evidence.element_boxes || [];
    if (!selectors.length && !xpaths.length && !boxes.length) return "";
    var html = '<div class="ux-audit-element-info"><h4>Offending element(s)</h4>';
    selectors.forEach(function (sel) {
      html +=
        '<div class="ux-audit-selector-row"><span class="ux-audit-label">Selector:</span>' +
        '<code class="ux-audit-selector" title="' + esc(sel) + '">' + esc(sel) + "</code></div>";
    });
    xpaths.forEach(function (xp) {
      html +=
        '<div class="ux-audit-xpath-row"><span class="ux-audit-label">XPath:</span>' +
        '<code class="ux-audit-xpath" title="' + esc(xp) + '">' + esc(xp) + "</code></div>";
    });
    if (boxes.length) {
      html += '<div class="ux-audit-boxes"><span class="ux-audit-label">Bounding boxes:</span>';
      boxes.forEach(function (box) {
        html +=
          '<code class="ux-audit-box">[x:' + esc(box.x) + ", y:" + esc(box.y) +
          ", w:" + esc(box.w) + ", h:" + esc(box.h) + "]</code>";
      });
      html += "</div>";
    }
    html += "</div>";
    return html;
  }
  function auditScreenshotsHtml(evidence) {
    var combined = evidence.combined_screenshots || [];
    var single = evidence.combined_screenshot;
    var cropped = evidence.element_screenshots || [];
    var selectors = evidence.element_selectors || [];
    if (!combined.length && !single && !cropped.length) return "";
    var html = '<div class="ux-audit-screenshots">';
    combined.forEach(function (shot, index) {
      html +=
        '<figure class="ux-audit-figure"><img src="' + esc(shot) +
        '" alt="Annotated elements — red outlines show offending element(s)" class="ux-audit-screenshot" loading="lazy">' +
        "<figcaption>Annotated view " + (index + 1) +
        " — red outlines mark the offending element(s) in context</figcaption></figure>";
    });
    if (single && !combined.length) {
      html +=
        '<figure class="ux-audit-figure"><img src="' + esc(single) +
        '" alt="Annotated elements — red outlines show offending element(s)" class="ux-audit-screenshot" loading="lazy">' +
        "<figcaption>Annotated view — red outlines mark the offending element(s) in context</figcaption></figure>";
    }
    cropped.forEach(function (shot, index) {
      var code =
        selectors.length && selectors[index]
          ? " — <code>" + esc(selectors[index]) + "</code>"
          : "";
      html +=
        '<figure class="ux-audit-figure"><img src="' + esc(shot) +
        '" alt="Cropped view of offending element with red outline" class="ux-audit-screenshot" loading="lazy">' +
        "<figcaption>Cropped element " + (index + 1) + code +
        " — red outline marks exact bounds</figcaption></figure>";
    });
    html += "</div>";
    return html;
  }
  function auditIssueHtml(issue) {
    var evidence = issue.evidence || {};
    var selectors =
      evidence.element_selectors && evidence.element_selectors.length
        ? evidence.element_selectors
        : evidence.element_selector
        ? [evidence.element_selector]
        : [];
    var hint = selectors.length ? selectors[0] : "";
    var html =
      '<li class="ux-audit-issue severity-' + esc(issue.severity) + '">' +
      '<div class="ux-audit-issue-header">' +
      '<span class="severity-label severity-' + esc(issue.severity) + '">' +
      titleCase(issue.severity) + "</span>" +
      "<strong>" + esc(issue.title) + "</strong>" +
      '<span class="ux-audit-category">' + esc(issue.category) + "</span>";
    if (hint) {
      html +=
        '<span class="ux-audit-elements-hint" title="' + esc(selectors.join(", ")) + '">' +
        esc(truncate(hint, 48)) + "</span>";
    }
    html +=
      "</div>" +
      '<details class="ux-audit-evidence"><summary>Evidence</summary>' +
      auditElementInfoHtml(evidence) +
      auditScreenshotsHtml(evidence) +
      '<pre class="ux-audit-evidence-data">' + jsonPre(evidence) + "</pre>" +
      "</details></li>";
    return html;
  }
  function slopPatternListHtml(patterns) {
    patterns = patterns || [];
    var html = '<ul class="slop-pattern-list">';
    patterns.forEach(function (pattern) {
      if (!pattern.triggered) return;
      html +=
        '<li class="slop-pattern slop-pattern-' + esc(pattern.category) + '">' +
        '<span class="slop-pattern-weight">+' + esc(pattern.weight) + "</span>" +
        "<strong>" + esc(pattern.label) + "</strong>" +
        '<code class="slop-pattern-id">' + esc(pattern.id) + "</code>";
      if (pattern.evidence) {
        html +=
          '<details class="slop-evidence"><summary>Evidence</summary>' +
          '<pre class="slop-evidence-data">' + jsonPre(pattern.evidence) + "</pre></details>";
      }
      html += "</li>";
    });
    html += "</ul>";
    return html;
  }
  function slopCardHtml(slop) {
    if (!slop) return "";
    var tierClass = esc(String(slop.tier || "mild").toLowerCase());
    var unified = numberValue(slop.unifiedScore);
    var html =
      '<div class="slop-card slop-' + tierClass + '">' +
      '<div class="slop-card-head">' +
      '<div class="slop-score"><span class="slop-score-number">' + esc(slop.score) + "</span>" +
      '<span class="slop-score-max">/100</span></div>' +
      '<div class="slop-meta">' +
      '<span class="slop-tier slop-tier-' + tierClass + '">' + esc(slop.tier) + "</span>" +
      '<span class="slop-grade">Grade ' + esc(slop.grade) + "</span>" +
      '<span class="slop-count">' + esc(slop.patternsFlagged) + "/" + esc(slop.patternsTotal) +
      " design patterns triggered</span>";
    if (unified !== null) {
      html +=
        '<span class="slop-unified">Unified (design + copy): ' + esc(slop.unifiedScore) +
        "/100 · " + esc(slop.unifiedTier || "") + "</span>";
    }
    html += "</div>";
    if (slop.verdict) html += '<p class="slop-verdict">' + esc(slop.verdict) + "</p>";
    html +=
      "</div>" +
      '<details class="slop-patterns"><summary>Triggered patterns (' +
      esc(slop.patternsFlagged) + ")</summary>" + slopPatternListHtml(slop.patterns);
    if (slop.copy && numberValue(slop.copy.score) !== null) {
      html +=
        '<div class="slop-copy"><h4>Copy axis · ' + esc(slop.copy.tier) + " · " +
        esc(slop.copy.score) + "/100 (" + esc(slop.copy.patternsFlagged) + "/" +
        esc(slop.copy.patternsTotal) + ")</h4>" + slopPatternListHtml(slop.copy.patterns) + "</div>";
    }
    html += "</details></div>";
    return html;
  }
  function sectionTag(sectionId, sectionClass, titleId) {
    return (
      "<section " + 'id="' + sectionId + '" class="' + sectionClass +
      '" aria-labelledby="' + titleId + '">'
    );
  }
  function auditSectionHtml(payload) {
    var urlReports = Array.isArray(payload.urls) ? payload.urls : [];
    var total = numberValue(payload.total_issues);
    if (total === null) {
      total = 0;
      urlReports.forEach(function (report) {
        total += Array.isArray(report.issues) ? report.issues.length : 0;
      });
    }
    var html =
      sectionTag("ux-audit", "report-section ux-audit", "ux-audit-title") +
      '<div class="report-section-heading"><div><h2 id="ux-audit-title">Static &amp; visual page findings</h2></div>' +
      '<span class="finding-count">' + esc(total) + " issue" + (total === 1 ? "" : "s") +
      "</span></div>" +
      '<p class="summary-assessment">' + AUDIT_SUMMARY + "</p>";
    urlReports.forEach(function (urlReport) {
      html += '<div class="ux-audit-url"><h3 class="ux-audit-url-label">' + esc(urlReport.url) + "</h3>";
      html += slopCardHtml(urlReport.slop);
      var issues = Array.isArray(urlReport.issues) ? urlReport.issues : [];
      if (issues.length) {
        html += '<ul class="ux-audit-issues">';
        SEVERITY_ORDER.forEach(function (severity) {
          issues.forEach(function (issue) {
            if (issue.severity === severity) html += auditIssueHtml(issue);
          });
        });
        html += "</ul>";
      } else {
        html += '<p class="ux-audit-clean">No static page issues found for this URL.</p>';
      }
      html += "</div>";
    });
    html += "</section>";
    return html;
  }
  function hydrateAudit() {
    var path = SOURCES.ux_audit;
    if (!path) return;
    // Browsers block local-file fetch for file:// documents: skip it and
    // keep the embedded snapshot (with an explanatory note) instead of
    // opening the console with an expected fetch failure.
    if (window.location.protocol === "file:") {
      markLive(document.getElementById("ux-audit"), path, true);
      return;
    }
    fetchJson(path)
      .then(function (payload) {
        if (!payload || payload.schema_version !== "ux-audit-v1") return;
        var html = auditSectionHtml(payload);
        var section = document.getElementById("ux-audit");
        if (section) {
          section.outerHTML = html;
        } else {
          ensureView("view-audit", "Page checks", html);
        }
        markLive(document.getElementById("ux-audit"), path, false);
      })
      .catch(function () {
        markLive(document.getElementById("ux-audit"), path, true);
      });
  }

  // ------------------------------------------------------------------
  // Performance tab (pagespeed sidecar)
  // ------------------------------------------------------------------
  function psiMetaHtml(name, strategy) {
    var parts = [];
    if (has(strategy.lighthouse_version)) parts.push("Lighthouse " + esc(strategy.lighthouse_version));
    if (has(strategy.fetched_at)) parts.push("fetched " + esc(strategy.fetched_at));
    if (strategy.from_cache) parts.push("served from cache");
    if (has(strategy.final_url) && strategy.final_url !== strategy.requested_url && strategy.final_url !== name) {
      parts.push("final URL " + esc(strategy.final_url));
    }
    return parts.length ? '<p class="psi-meta">' + parts.join(" · ") + "</p>" : "";
  }
  function psiFieldHtml(strategy) {
    var field = strategy.field_data;
    if (!field) return "";
    var html =
      '<div class="psi-field"><span class="psi-field-label">Field data (CrUX):</span>' +
      '<span class="psi-field-overall">' + esc(field.overall_category) + "</span>";
    if (field.metrics) {
      Object.keys(field.metrics).forEach(function (metricId) {
        var metric = field.metrics[metricId];
        html += '<span class="psi-field-metric">' + esc(metricId) + " " + esc(metric.percentile);
        if (has(metric.category)) html += " (" + esc(metric.category) + ")";
        html += "</span>";
      });
    }
    html += "</div>";
    return html;
  }
  function psiCategoriesHtml(categories) {
    if (!categories || !categories.length) return "";
    var html = '<div class="psi-categories">';
    categories.forEach(function (category) {
      var score = numberValue(category.score);
      html +=
        '<div class="psi-category psi-score-' + scoreClass(score) + '">' +
        '<span class="psi-category-title">' + esc(category.title) + "</span>" +
        '<span class="psi-category-score">' + esc(category.score_percent) + "</span>";
      if (has(category.display_value)) {
        html += '<span class="psi-category-value">' + esc(category.display_value) + "</span>";
      }
      html += "</div>";
    });
    html += "</div>";
    return html;
  }
  function psiTotalsHtml(audits) {
    var totals = (audits && audits.totals) || {};
    return (
      '<p class="psi-totals">' +
      esc(totals.failed) + " failed · " + esc(totals.passed) + " passed · " +
      esc(totals.not_applicable) + " not applicable · " + esc(totals.manual) + " manual · " +
      esc(totals.informative) + " informative · " + esc(totals.error) + " error</p>"
    );
  }
  function psiAuditHeadHtml(audit) {
    var score = has(audit.score_percent) ? esc(audit.score_percent) : "—";
    var html =
      '<div class="psi-audit-head"><span class="psi-audit-score">' + score + "</span>" +
      "<strong>" + esc(audit.title) + "</strong>";
    if (has(audit.display_value)) {
      html += '<span class="psi-audit-value">' + esc(audit.display_value) + "</span>";
    }
    html += '<code class="psi-audit-id">' + esc(audit.id) + "</code></div>";
    return html;
  }
  function psiAuditListHtml(audits) {
    var html = '<ul class="psi-audit-list">';
    audits.forEach(function (audit) {
      html += '<li class="psi-audit' + (audit.failed ? " psi-audit-failed" : "") + '">' + psiAuditHeadHtml(audit);
      if (has(audit.description)) {
        html +=
          '<details class="psi-audit-detail"><summary>Detail</summary>' +
          '<p class="psi-audit-description">' + esc(audit.description) + "</p></details>";
      }
      html += "</li>";
    });
    html += "</ul>";
    return html;
  }
  function psiSavingsHtml(opportunity) {
    var html = '<div class="psi-opportunity-head"><strong>' + esc(opportunity.title) + "</strong>";
    if (has(opportunity.display_value)) {
      html += '<span class="psi-audit-value">' + esc(opportunity.display_value) + "</span>";
    }
    var ms = numberValue(opportunity.savings_ms);
    var bytes = numberValue(opportunity.savings_bytes);
    if (ms !== null) html += '<span class="psi-savings">~' + esc(ms) + " ms</span>";
    if (bytes !== null) html += '<span class="psi-savings">~' + esc(bytes) + " bytes</span>";
    html += '<code class="psi-audit-id">' + esc(opportunity.id) + "</code></div>";
    return html;
  }
  function psiOpportunityItemsHtml(items) {
    items = items || [];
    if (!items.length) return "";
    var html =
      '<details class="psi-audit-detail"><summary>' + esc(items.length) + " item(s)</summary>" +
      '<ul class="psi-opportunity-items">';
    items.forEach(function (item) {
      html += "<li>";
      var first = true;
      function join(prefix) {
        first = false;
        return prefix;
      }
      if (has(item.url)) {
        html += '<code class="psi-item-url">' + esc(item.url) + "</code>";
        first = false;
      }
      if (numberValue(item.wastedBytes) !== null) {
        html += (first ? join("") : " · ") + "~" + esc(item.wastedBytes) + " bytes wasted";
      }
      if (numberValue(item.wastedMs) !== null) {
        html += (first ? join("") : " · ") + "~" + esc(item.wastedMs) + " ms wasted";
      }
      if (numberValue(item.totalBytes) !== null) {
        html += (first ? join("") : " · ") + esc(item.totalBytes) + " bytes total";
      }
      if (numberValue(item.responseTime) !== null) {
        html += (first ? join("") : " · ") + esc(item.responseTime) + " ms response";
      }
      html += "</li>";
    });
    html += "</ul></details>";
    return html;
  }
  function psiOpportunityListHtml(opportunities) {
    var html = '<ul class="psi-opportunity-list">';
    opportunities.forEach(function (opportunity) {
      html +=
        '<li class="psi-opportunity">' + psiSavingsHtml(opportunity) +
        psiOpportunityItemsHtml(opportunity.items) + "</li>";
    });
    html += "</ul>";
    return html;
  }
  function psiStrategyHtml(name, strategy) {
    var html =
      '<div class="psi-strategy psi-strategy-' + esc(strategy.status) + '">' +
      '<h4 class="psi-strategy-label">' + titleCase(name) + " strategy</h4>";
    if (strategy.status !== "ok") {
      html += '<p class="psi-error">Unavailable: ' + esc(strategy.error || "no error detail recorded") + "</p>";
      if (has(strategy.error_code)) {
        html += '<p class="psi-error-detail"><code>' + esc(strategy.error_code) + "</code></p>";
      }
      if (strategy.error_code && String(strategy.error_code).indexOf("FAILED_DOCUMENT_REQUEST") !== -1) {
        html +=
          '<p class="psi-error-help">Lighthouse couldn\'t load this URL (government portal, likely blocking ' +
          'automated audits or slow ASPX). The local <code>ux-audit</code> and live-page checks in this report ' +
          'still ran — use the link above to retry on pagespeed.web.dev directly.</p>';
      }
      html += "</div>";
      return html;
    }
    html += psiMetaHtml(name, strategy);
    html += psiFieldHtml(strategy);
    html += psiCategoriesHtml(strategy.categories);
    html += psiTotalsHtml(strategy.audits);
    if (strategy.audits && strategy.audits.failed && strategy.audits.failed.length) {
      html +=
        '<details class="psi-audit-bucket" open><summary>Failed audits (' +
        esc(strategy.audits.failed.length) + ")</summary>" + psiAuditListHtml(strategy.audits.failed) + "</details>";
    }
    if (strategy.opportunities && strategy.opportunities.length) {
      html +=
        '<details class="psi-audit-bucket" open><summary>Opportunities (' +
        esc(strategy.opportunities.length) + ")</summary>" + psiOpportunityListHtml(strategy.opportunities) + "</details>";
    }
    if (strategy.metric_savings && strategy.metric_savings.length) {
      html +=
        '<details class="psi-audit-bucket"><summary>Metric savings (' +
        esc(strategy.metric_savings.length) + ")</summary>" + psiOpportunityListHtml(strategy.metric_savings) + "</details>";
    }
    PSI_PASS_BUCKETS.forEach(function (pair) {
      var bucket = pair[0];
      var label = pair[1];
      var audits = strategy.audits && strategy.audits[bucket];
      if (audits && audits.length) {
        html +=
          '<details class="psi-audit-bucket"><summary>' + label + " (" + esc(audits.length) + ")</summary>" +
          psiAuditListHtml(audits) + "</details>";
      }
    });
    html += "</div>";
    return html;
  }
  function savedReportPath() {
    var path = SOURCES.pagespeed || "";
    var parts = String(path).split("/");
    parts.pop();
    parts.push("pagespeed-report.html");
    return parts.join("/");
  }
  function psiUrlHtml(urlReport) {
    var html = '<div class="psi-url"><h3 class="psi-url-label">' + esc(urlReport.url);
    if (has(urlReport.pagespeed_web_url) || urlReport.pagespeed_web_saved) {
      html +=
        '<a class="psi-web-link psi-web-link-saved" href="' + esc(savedReportPath()) +
        '" title="Offline replica of the recorded analysis — identical numbers to this tab">View saved report</a>';
    }
    if (has(urlReport.pagespeed_web_fresh_url)) {
      html +=
        '<a class="psi-web-link psi-web-link-fresh" href="' + esc(urlReport.pagespeed_web_fresh_url) +
        '" target="_blank" rel="noopener noreferrer" title="Run a fresh analysis for this URL on pagespeed.web.dev — a new Lighthouse run, scores will vary">Re-run on pagespeed.web.dev</a>';
    }
    html += "</h3>";
    if (has(urlReport.error)) html += '<p class="psi-error">Report unavailable: ' + esc(urlReport.error) + "</p>";
    var strategies = urlReport.strategies || {};
    var strategyNames = Object.keys(strategies);
    if (!strategyNames.length) {
      html += '<p class="ux-audit-clean">No PageSpeed reports could be fetched for this URL.</p>';
    }
    strategyNames.forEach(function (name) {
      html += psiStrategyHtml(name, strategies[name]);
    });
    html += "</div>";
    return html;
  }
  function pagespeedSectionHtml(payload) {
    var urlReports = Array.isArray(payload.urls) ? payload.urls : [];
    var okCount = numberValue(payload.ok_strategy_count);
    if (okCount === null) {
      okCount = 0;
      urlReports.forEach(function (report) {
        Object.keys(report.strategies || {}).forEach(function (name) {
          if ((report.strategies[name] || {}).status === "ok") okCount += 1;
        });
      });
    }
    var html =
      sectionTag("pagespeed", "report-section pagespeed", "pagespeed-title") +
      '<div class="report-section-heading"><div><h2 id="pagespeed-title">Lighthouse performance reports</h2></div>' +
      '<span class="finding-count">' + esc(okCount) + " report" + (okCount === 1 ? "" : "s") +
      "</span></div>" +
      '<p class="summary-assessment">' + PSI_SUMMARY + "</p>";
    urlReports.forEach(function (urlReport) {
      html += psiUrlHtml(urlReport);
    });
    html += "</section>";
    return html;
  }
  function hydratePagespeed() {
    var path = SOURCES.pagespeed;
    if (!path) return;
    if (window.location.protocol === "file:") {
      markLive(document.getElementById("pagespeed"), path, true);
      return;
    }
    fetchJson(path)
      .then(function (payload) {
        if (!payload || payload.schema_version !== "pagespeed-insights-v1") return;
        var html = pagespeedSectionHtml(payload);
        var section = document.getElementById("pagespeed");
        if (section) {
          section.outerHTML = html;
        } else {
          ensureView("view-performance", "Page speed", html);
        }
        markLive(document.getElementById("pagespeed"), path, false);
      })
      .catch(function () {
        markLive(document.getElementById("pagespeed"), path, true);
      });
  }

  // fire both hydrations (independent; failures fall back to embedded HTML)
  hydrateAudit();
  hydratePagespeed();
}());