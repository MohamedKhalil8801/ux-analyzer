(function () {
  "use strict";

  var dataNode = document.getElementById("report-data");
  if (!dataNode) return;
  var data = JSON.parse(dataNode.textContent || "{}");
  var runs = data.runs || [];
  var state = { runId: runs.length ? runs[0].run_id : "", eventIndex: 0 };
  var runFilter = document.getElementById("run-filter");
  var versionFilter = document.getElementById("version-filter");
  var policyFilter = document.getElementById("policy-filter");
  var runSelect = document.getElementById("run-select");
  var detail = document.getElementById("run-detail");
  var timelineList = document.getElementById("timeline-list");
  var timelinePosition = document.getElementById("timeline-position");
  var timelineDetail = document.getElementById("timeline-detail");

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function uniqueValues(key) {
    return runs.map(function (run) { return run[key]; }).filter(function (value, index, values) {
      return value && values.indexOf(value) === index;
    }).sort();
  }

  function fillSelect(select, values, emptyLabel) {
    while (select.options.length > 1) select.remove(1);
    values.forEach(function (value) {
      var option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
    select.options[0].textContent = emptyLabel;
  }

  function visibleRuns() {
    var search = (runFilter.value || "").toLowerCase();
    return runs.filter(function (run) {
      var text = [run.run_id, run.scenario_label, run.version_label, run.persona_label, run.policy].join(" ").toLowerCase();
      return (!search || text.indexOf(search) !== -1) &&
        (!versionFilter.value || run.version_id === versionFilter.value) &&
        (!policyFilter.value || run.policy === policyFilter.value);
    });
  }

  function renderRunOptions() {
    var visible = visibleRuns();
    while (runSelect.firstChild) runSelect.removeChild(runSelect.firstChild);
    visible.forEach(function (run) {
      var option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.run_id + " | " + run.version_label;
      runSelect.appendChild(option);
    });
    if (!visible.some(function (run) { return run.run_id === state.runId; })) {
      state.runId = visible.length ? visible[0].run_id : "";
    }
    runSelect.value = state.runId;
  }

  function currentRun() {
    return runs.find(function (run) { return run.run_id === state.runId; }) || null;
  }

  function renderTimeline(run) {
    while (timelineList.firstChild) timelineList.removeChild(timelineList.firstChild);
    var events = run ? (run.timeline || []) : [];
    state.eventIndex = Math.min(state.eventIndex, Math.max(events.length - 1, 0));
    events.forEach(function (event, index) {
      var item = document.createElement("li");
      var button = element("button", "timeline-event", (event.sequence || index + 1) + "  " + event.kind);
      button.type = "button";
      button.setAttribute("aria-current", index === state.eventIndex ? "true" : "false");
      button.addEventListener("click", function () { state.eventIndex = index; render(); });
      item.appendChild(button);
      timelineList.appendChild(item);
    });
    timelinePosition.textContent = events.length ? (state.eventIndex + 1) + " / " + events.length : "0 / 0";
    timelineDetail.textContent = events.length ? describe(events[state.eventIndex]) : "No timeline events recorded.";
  }

  function describe(value) {
    if (!value) return "No event selected.";
    var parts = [value.kind];
    if (value.reason) parts.push(value.reason);
    if (value.succeeded !== undefined) parts.push(value.succeeded ? "succeeded" : "failed");
    if (value.outcome) parts.push(value.outcome);
    return parts.join(" | ");
  }

  function addMeta(parent, label, value) {
    var item = element("div", "meta-item");
    item.appendChild(element("span", "meta-label", label));
    item.appendChild(element("span", "meta-value", value));
    parent.appendChild(item);
  }

  function addPanel(parent, title, records, formatter) {
    var panel = element("section", "detail-panel");
    panel.appendChild(element("h3", "", title));
    var list = element("ul", "record-list");
    if (!records || !records.length) {
      list.appendChild(element("li", "empty", "No records."));
    } else {
      records.forEach(function (record) {
        var item = element("li", "record");
        formatter(record, item);
        list.appendChild(item);
      });
    }
    panel.appendChild(list);
    parent.appendChild(panel);
  }

  function addJsonRecord(record, item) {
    item.appendChild(element("strong", "", record.name || record.kind || record.category || record.evidence_id || "Record"));
    var copy = Object.assign({}, record);
    delete copy.name; delete copy.kind; delete copy.category; delete copy.evidence_id;
    item.appendChild(element("small", "", JSON.stringify(copy)));
  }

  function renderViewports(parent, run) {
    var panel = element("section", "detail-panel");
    panel.appendChild(element("h3", "", "Screenshots and attention overlay"));
    var strip = element("div", "viewport-strip");
    (run.snapshots || []).forEach(function (snapshot) {
      var card = element("div", "viewport-card");
      var title = element("div", "viewport-title");
      title.appendChild(element("span", "", snapshot.id));
      title.appendChild(element("span", "", snapshot.viewport.width + " x " + snapshot.viewport.height));
      card.appendChild(title);
      var frame = element("div", "viewport-frame");
      frame.dataset.viewportWidth = String(snapshot.viewport.width);
      frame.dataset.viewportHeight = String(snapshot.viewport.height);
      frame.style.aspectRatio = snapshot.viewport.width + " / " + snapshot.viewport.height;
      if (snapshot.screenshot) {
        var image = document.createElement("img");
        image.src = snapshot.screenshot;
        image.alt = "Recorded screenshot for " + snapshot.id;
        frame.appendChild(image);
      } else {
        frame.appendChild(element("span", "empty", "Screenshot artifact unavailable."));
      }
      (snapshot.elements || []).forEach(function (item) {
        var overlay = element("div", "viewport-overlay" + (item.inspected ? " inspected" : item.noticed ? " noticed" : ""));
        var bounds = item.bounds;
        overlay.style.left = (bounds.x / snapshot.viewport.width * 100) + "%";
        overlay.style.top = (bounds.y / snapshot.viewport.height * 100) + "%";
        overlay.style.width = (bounds.width / snapshot.viewport.width * 100) + "%";
        overlay.style.height = (bounds.height / snapshot.viewport.height * 100) + "%";
        overlay.appendChild(element("span", "overlay-label", item.label + (item.inspected ? " | inspected" : item.noticed ? " | noticed" : "")));
        frame.appendChild(overlay);
      });
      card.appendChild(frame);
      strip.appendChild(card);
    });
    if (!strip.childNodes.length) strip.appendChild(element("p", "empty", "No viewport captures recorded."));
    panel.appendChild(strip);
    parent.appendChild(panel);
  }

  function renderRun(run) {
    while (detail.firstChild) detail.removeChild(detail.firstChild);
    if (!run) {
      detail.appendChild(element("p", "muted", "No run matches filters."));
      return;
    }
    var meta = element("div", "run-meta");
    addMeta(meta, "Run", run.run_id);
    addMeta(meta, "Scenario", run.scenario_label);
    addMeta(meta, "Version", run.version_label);
    addMeta(meta, "Persona", run.persona_label);
    addMeta(meta, "Policy", run.policy);
    addMeta(meta, "Outcome", run.outcome);
    addMeta(meta, "Verified", run.verified ? "yes" : "no");
    detail.appendChild(meta);
    if (run.run_page) {
      var pageLink = document.createElement("a");
      pageLink.href = run.run_page;
      pageLink.textContent = "Open self-contained run page";
      detail.appendChild(pageLink);
    }

    renderViewports(detail, run);
    var grid = element("div", "detail-grid");
    addPanel(grid, "Prominence contributions", (run.prominence || []).reduce(function (all, record) { return all.concat(record.scores || []); }, []), addJsonRecord);
    addPanel(grid, "Scent records", run.scent_records, addJsonRecord);
    addPanel(grid, "Decisions", run.decisions, addJsonRecord);
    addPanel(grid, "Actions", run.actions, addJsonRecord);
    addPanel(grid, "Verification", run.verification ? [run.verification] : [], addJsonRecord);
    addPanel(grid, "Memory", run.memory, addJsonRecord);
    addPanel(grid, "Model manifests", run.manifests && run.manifests.provider_manifests ? run.manifests.provider_manifests : [], addJsonRecord);
    addPanel(grid, "Evidence", run.evidence, addJsonRecord);
    addPanel(grid, "Findings", run.findings, addJsonRecord);
    addPanel(grid, "Metrics", run.metrics, addJsonRecord);
    addPanel(grid, "Limitations", (run.limitations || []).map(function (value) { return { name: value }; }), addJsonRecord);
    detail.appendChild(grid);
  }

  function render() {
    renderRunOptions();
    var run = currentRun();
    renderTimeline(run);
    renderRun(run);
  }

  fillSelect(versionFilter, uniqueValues("version_id"), "All versions");
  fillSelect(policyFilter, uniqueValues("policy"), "All policies");
  [runFilter, versionFilter, policyFilter].forEach(function (control) { control.addEventListener("input", render); control.addEventListener("change", render); });
  runSelect.addEventListener("change", function () { state.runId = runSelect.value; state.eventIndex = 0; render(); });
  document.getElementById("timeline-prev").addEventListener("click", function () { if (state.eventIndex > 0) { state.eventIndex -= 1; render(); } });
  document.getElementById("timeline-next").addEventListener("click", function () { var run = currentRun(); if (run && state.eventIndex < (run.timeline || []).length - 1) { state.eventIndex += 1; render(); } });
  render();
}());
