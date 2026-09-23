// Element-reference chips (e{n} aliases in synthesis prose):
// hover shows the evidence detail card (+ element crop when the alias
// points at a recorded element or executed action); click copies a
// verified-unique locator, the recorded page URL (viewport aliases), or
// the evidence ID. Shared by the index and run pages; owns #alias-chips.
(function () {
  "use strict";

  var aliasChips = (function () {
    try {
      var node = document.getElementById("alias-chips");
      return node ? JSON.parse(node.textContent || "{}") : {};
    } catch (_) { return {}; }
  })();

  function copyText(text, done) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
      return;
    }
    try {
      var area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(area);
      done(ok);
    } catch (_) { done(false); }
  }

  function chipDetail(alias) {
    var entry = aliasChips[alias];
    return entry && entry.detail ? entry.detail : null;
  }

  function chipElement(alias) {
    var entry = aliasChips[alias];
    return entry && entry.element ? entry.element : null;
  }

  function chipPageUrl(alias) {
    var entry = aliasChips[alias];
    return entry && entry.page_url ? entry.page_url : "";
  }

  // What clicking this chip should put on the clipboard. Element and
  // action chips copy a verified-unique locator (css preferred, xpath
  // fallback); viewport chips copy the recorded page URL; everything else
  // copies the evidence ID. The label says exactly which, so a paste is
  // never mistaken for another kind of value.
  function copyTarget(alias) {
    var element = chipElement(alias);
    var locator = element && element.locator;
    if (locator && locator.selector) return { text: locator.selector, label: "copied css" };
    if (locator && locator.xpath) return { text: locator.xpath, label: "copied xpath" };
    var pageUrl = chipPageUrl(alias);
    if (pageUrl) return { text: pageUrl, label: "copied page url" };
    var detail = chipDetail(alias);
    if (detail && detail.evidence_id) return { text: detail.evidence_id, label: "copied evidence id" };
    return null;
  }

  function formatDetail(detail) {
    if (!detail) return "";
    var lines = [];
    if (detail.kind) lines.push("kind: " + detail.kind);
    if (detail.surface) lines.push("surface: " + detail.surface);
    if (detail.name) lines.push("metric: " + detail.name);
    if (detail.value !== undefined && detail.value !== null) lines.push("value: " + detail.value);
    if (detail.evidence_class) lines.push("class: " + detail.evidence_class);
    if (detail.label) lines.push("element: " + detail.label);
    if (!lines.length && detail.evidence_id) lines.push(detail.evidence_id);
    return lines.join("\n");
  }

  function flashChip(chip, label) {
    var original = chip.textContent;
    chip.textContent = label;
    chip.classList.add("alias-chip-copied");
    setTimeout(function () {
      chip.textContent = original;
      chip.classList.remove("alias-chip-copied");
    }, 1400);
  }

  var aliasPopover = null;
  function ensureAliasPopover() {
    if (aliasPopover) return aliasPopover;
    aliasPopover = document.createElement("div");
    aliasPopover.className = "alias-popover";
    aliasPopover.setAttribute("role", "tooltip");
    aliasPopover.hidden = true;
    document.body.appendChild(aliasPopover);
    return aliasPopover;
  }

  function showAliasPopover(chip) {
    var alias = chip.getAttribute("data-alias");
    var detail = chipDetail(alias);
    var element = chipElement(alias);
    if (!detail && !element) return;
    var pop = ensureAliasPopover();
    var title = document.createElement("strong");
    title.textContent = alias;
    pop.innerHTML = "";
    pop.appendChild(title);
    var detailText = formatDetail(detail);
    if (detailText) {
      var pre = document.createElement("pre");
      pre.className = "alias-popover-detail";
      pre.textContent = detailText;
      pop.appendChild(pre);
    }
    if (element && element.preview) {
      var img = document.createElement("img");
      img.className = "alias-popover-preview";
      img.src = element.preview;
      img.alt = element.label || ("recorded element " + alias);
      pop.appendChild(img);
      if (element.label) {
        var caption = document.createElement("div");
        caption.className = "alias-popover-caption";
        caption.textContent = element.label;
        pop.appendChild(caption);
      }
    }
    var hint = document.createElement("div");
    hint.className = "alias-popover-hint";
    if (element && element.locator) {
      hint.textContent = "Click to copy the unique locator";
    } else if (element) {
      hint.textContent = "No unique locator could be verified for this element";
    } else if (chipPageUrl(alias)) {
      hint.textContent = "Click to copy the recorded page URL";
    } else {
      hint.textContent = "Click to copy the evidence ID";
    }
    pop.appendChild(hint);
    pop.hidden = false;
    var rect = chip.getBoundingClientRect();
    pop.style.top = rect.bottom + window.scrollY + 6 + "px";
    pop.style.left = Math.max(8, rect.left + window.scrollX) + "px";
  }

  function hideAliasPopover() {
    if (aliasPopover) { aliasPopover.hidden = true; }
  }

  document.querySelectorAll(".alias-chip[data-alias]").forEach(function (chip) {
    var alias = chip.getAttribute("data-alias");
    function showIfKnown() {
      if (chipDetail(alias) || chipElement(alias)) showAliasPopover(chip);
    }
    ["mouseenter", "focus"].forEach(function (event) {
      chip.addEventListener(event, showIfKnown);
    });
    ["mouseleave", "blur"].forEach(function (event) {
      chip.addEventListener(event, hideAliasPopover);
    });
    chip.addEventListener("click", function () {
      var target = copyTarget(alias);
      if (target) {
        copyText(target.text, function () {
          flashChip(chip, target.label);
        });
      }
    });
  });

  // Redesign proposal section chips: the verified locator is rendered into
  // the element (capture-inventory uniqueness was checked at render time);
  // chips without one say so instead of copying a dead value.
  document.querySelectorAll(".redesign-section-chip[data-copy-locator]").forEach(function (chip) {
    var locator = chip.getAttribute("data-copy-locator");
    chip.addEventListener("click", function () {
      if (!locator) return;
      copyText(locator, function () {
        flashChip(chip, "copied locator");
      });
    });
  });
}());
