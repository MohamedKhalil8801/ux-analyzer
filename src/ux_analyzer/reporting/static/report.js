(function () {
  "use strict";

  var dataNode = document.getElementById("report-data");
  if (!dataNode) return;
  var data = JSON.parse(dataNode.textContent || "{}");
  var runs = data.runs || [];
  var state = {
    runId: runs.length ? runs[0].run_id : "",
    eventIndex: 0,
    viewportId: "",
    elementId: ""
  };
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

  function exact(value) {
    if (value === undefined || value === null || value === "") return "n/a";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  }

  function titleCase(value) {
    return String(value || "record").replace(/[-_]/g, " ").replace(/\b\w/g, function (letter) {
      return letter.toUpperCase();
    });
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
      option.textContent = run.run_id + " | " + run.version_label + (run.failed ? " | failed" : "");
      runSelect.appendChild(option);
    });
    if (!visible.some(function (run) { return run.run_id === state.runId; })) {
      state.runId = visible.length ? visible[0].run_id : "";
      state.viewportId = "";
      state.elementId = "";
    }
    runSelect.value = state.runId;
  }

  function currentRun() {
    return runs.find(function (run) { return run.run_id === state.runId; }) || null;
  }

  function addField(parent, label, value) {
    var row = element("div", "field-row");
    row.appendChild(element("dt", "field-label", label));
    row.appendChild(element("dd", "field-value", exact(value)));
    parent.appendChild(row);
  }

  function addFields(parent, fields) {
    var list = element("dl", "field-list");
    fields.forEach(function (field) { addField(list, field[0], field[1]); });
    parent.appendChild(list);
  }

  function addValues(parent, label, values) {
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    var list = element("ul", "compact-list");
    (values || []).forEach(function (value) { list.appendChild(element("li", "", value)); });
    if (!list.childNodes.length) list.appendChild(element("li", "empty", "None recorded."));
    block.appendChild(list);
    parent.appendChild(block);
  }

  function addMapping(parent, label, mapping) {
    var entries = Object.keys(mapping || {}).sort();
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    var list = element("dl", "field-list compact-fields");
    entries.forEach(function (key) { addField(list, titleCase(key), mapping[key]); });
    if (!entries.length) block.appendChild(element("p", "empty", "None recorded."));
    else block.appendChild(list);
    parent.appendChild(block);
  }

  function eventSummary(value) {
    if (!value) return "No event selected.";
    var parts = ["Step " + exact(value.sequence), titleCase(value.kind)];
    if (value.reason) parts.push(value.reason);
    if (value.succeeded !== undefined) parts.push(value.succeeded ? "Succeeded" : "Failed");
    if (value.outcome) parts.push("Outcome: " + value.outcome);
    if (value.viewport_id) parts.push("Viewport: " + value.viewport_id);
    return parts.join(" | ");
  }

  function renderTimeline(run) {
    while (timelineList.firstChild) timelineList.removeChild(timelineList.firstChild);
    var events = run ? (run.timeline || []) : [];
    state.eventIndex = Math.min(state.eventIndex, Math.max(events.length - 1, 0));
    events.forEach(function (event, index) {
      var item = document.createElement("li");
      var button = element("button", "timeline-event", (event.sequence || index + 1) + "  " + titleCase(event.kind));
      button.type = "button";
      button.setAttribute("aria-current", index === state.eventIndex ? "true" : "false");
      button.addEventListener("click", function () { state.eventIndex = index; render(); });
      item.appendChild(button);
      timelineList.appendChild(item);
    });
    timelinePosition.textContent = events.length ? (state.eventIndex + 1) + " / " + events.length : "0 / 0";
    timelineDetail.textContent = events.length ? eventSummary(events[state.eventIndex]) : "No timeline events recorded.";
  }

  function addMeta(parent, label, value) {
    var item = element("div", "meta-item");
    item.appendChild(element("span", "meta-label", label));
    item.appendChild(element("span", "meta-value", exact(value)));
    parent.appendChild(item);
  }

  function addPanel(parent, title, records, formatter, panelClass) {
    var panel = element("section", "detail-panel" + (panelClass ? " " + panelClass : ""));
    panel.appendChild(element("h3", "", title));
    var list = element("ol", "record-list");
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
    return panel;
  }

  function actionText(action) {
    if (!action) return "No action";
    var text = titleCase(action.kind || "action");
    if (action.element_id) text += " on " + action.element_id;
    if (action.direction) text += " " + action.direction;
    return text;
  }

  function observationCard(record, item) {
    item.appendChild(element("strong", "", "Step " + exact(record.sequence) + " observation"));
    addFields(item, [
      ["Viewport", record.viewport_id],
      ["Region", record.region_context ? record.region_context.label : "none"]
    ]);
    addValues(item, "Noticed now", (record.newly_revealed_elements || []).map(function (entry) {
      return entry.label + " [" + entry.id + "]";
    }));
    addValues(item, "Remembered", (record.remembered_elements || []).map(function (entry) {
      return entry.label + " [" + entry.id + "]";
    }));
  }

  function selectionCard(record, item) {
    item.appendChild(element("strong", "", "Step " + exact(record.sequence) + " attention selection"));
    addFields(item, [
      ["Viewport", record.viewport_id],
      ["Mode", record.selection_mode],
      ["Region", record.region_id]
    ]);
    addValues(item, "Selected element IDs", record.selected_ids || []);
    addMapping(item, "Element probabilities", record.element_probabilities);
    addMapping(item, "Region probabilities", record.region_probabilities);
  }

  function decisionCard(record, item) {
    item.appendChild(element("strong", "", "Step " + exact(record.sequence) + " decision"));
    addFields(item, [
      ["Reason", record.reason || "No reason recorded"],
      ["Action", actionText(record.action)],
      ["Claimed success", record.claimed_success]
    ]);
  }

  function actionCard(record, item) {
    item.appendChild(element("strong", "", "Step " + exact(record.sequence) + " action result"));
    addFields(item, [
      ["Action", actionText(record.action)],
      ["Result", record.succeeded ? "succeeded" : "failed"],
      ["Failure reason", record.error]
    ]);
  }

  function verificationCard(record, item) {
    item.appendChild(element("strong", "", "Independent verification"));
    addFields(item, [
      ["Verified", record.verified],
      ["Details", record.details]
    ]);
    addValues(item, "Evidence IDs", record.evidence_ids || []);
  }

  function memoryCard(record, item) {
    item.appendChild(element("strong", "", record.key || "Memory item"));
    addFields(item, [
      ["Value", record.value],
      ["Strength", record.strength],
      ["Importance", record.importance],
      ["Age", record.age],
      ["Failure memory", record.is_failure]
    ]);
  }

  function manifestCard(record, item) {
    item.appendChild(element("strong", "", titleCase(record.role || "provider")));
    addFields(item, [
      ["Provider", record.provider_id],
      ["Model", record.model_id],
      ["Endpoint origin", record.endpoint_origin],
      ["Version", record.version],
      ["Prompt version", record.prompt_version],
      ["Schema version", record.schema_version]
    ]);
  }

  function modelCallCard(record, item) {
    item.appendChild(element("strong", "", titleCase(record.role || "model call")));
    addFields(item, [
      ["Model", record.model_id || record.model],
      ["Attempts", record.attempts],
      ["Latency ms", record.latency_ms],
      ["Prompt tokens", record.prompt_tokens],
      ["Completion tokens", record.completion_tokens],
      ["Total tokens", record.total_tokens],
      ["Failure", record.error]
    ]);
  }

  function evidenceCard(record, item) {
    item.appendChild(element("strong", "", record.description || record.evidence_id));
    addFields(item, [
      ["Evidence ID", record.evidence_id],
      ["Evidence class", record.evidence_class]
    ]);
    addValues(item, "Source events", record.source_event_ids || []);
  }

  function findingCard(record, item) {
    item.appendChild(element("strong", "finding-title", record.title || titleCase(record.category)));
    item.appendChild(element("p", "finding-cause", record.cause || record.explanation || "Cause unavailable."));
    addFields(item, [
      ["Severity", record.severity],
      ["Evidence class", record.evidence_class],
      ["Reproducibility", record.reproducibility]
    ]);
    addMapping(item, "Supporting metric values", record.supporting_metrics);
    addValues(item, "Run references", record.run_ids || []);
    addValues(item, "Viewport references", record.viewport_ids || []);
    addValues(item, "Element references", record.element_ids || []);
    addValues(item, "Action sequence", record.action_sequence || []);
    addValues(item, "Limitations", record.limitations || []);
    var links = element("div", "replay-links");
    (record.replay_links || []).forEach(function (href) {
      var link = element("a", "replay-link", "Replay evidence");
      link.href = href;
      links.appendChild(link);
    });
    if (links.childNodes.length) item.appendChild(links);
  }

  function metricCard(record, item) {
    item.appendChild(element("strong", "", titleCase(record.name)));
    addFields(item, [
      ["Exact value", record.value],
      ["Evidence class", record.evidence_class]
    ]);
    addValues(item, "Evidence IDs", record.evidence_ids || []);
  }

  function findSnapshot(run, viewportId) {
    return (run.snapshots || []).find(function (snapshot) { return snapshot.id === viewportId; }) || null;
  }

  function findElement(run, viewportId, elementId) {
    var snapshot = findSnapshot(run, viewportId);
    if (!snapshot) return null;
    return (snapshot.elements || []).find(function (item) { return item.id === elementId; }) || null;
  }

  function findProminence(run, viewportId, elementId) {
    var records = run.prominence || [];
    var preferred = records.filter(function (record) { return record.viewport_id === viewportId; });
    var source = preferred.length ? preferred : records;
    for (var index = source.length - 1; index >= 0; index -= 1) {
      var score = (source[index].scores || []).find(function (item) { return item.element_id === elementId; });
      if (score) return score;
    }
    return null;
  }

  function scentFor(run, viewportId, elementId) {
    return (run.scent_records || []).filter(function (record) {
      return !record.viewport_id || record.viewport_id === viewportId;
    }).reduce(function (values, record) {
      (record.scores || []).forEach(function (score) {
        if (score.element_id === elementId) values.push(titleCase(record.kind) + ": " + exact(score.score));
      });
      return values;
    }, []);
  }

  function featureTable(score) {
    var table = element("table", "feature-table");
    var head = element("thead", "");
    var headRow = element("tr", "");
    ["Feature", "Raw", "Normalized", "Contribution"].forEach(function (label) {
      headRow.appendChild(element("th", "", label));
    });
    head.appendChild(headRow);
    table.appendChild(head);
    var body = element("tbody", "");
    var names = Object.keys(Object.assign({}, score.raw_values || {}, score.normalized_values || {}, score.feature_contributions || {})).sort();
    names.forEach(function (name) {
      var row = element("tr", "");
      row.appendChild(element("td", "", titleCase(name)));
      row.appendChild(element("td", "raw-value", exact((score.raw_values || {})[name])));
      row.appendChild(element("td", "normalized-value", exact((score.normalized_values || {})[name])));
      row.appendChild(element("td", "contribution-value", exact((score.feature_contributions || {})[name])));
      body.appendChild(row);
    });
    table.appendChild(body);
    return table;
  }

  function renderSelectedElement(panel, run) {
    while (panel.firstChild) panel.removeChild(panel.firstChild);
    panel.appendChild(element("h3", "", "Selected element evidence"));
    var selected = findElement(run, state.viewportId, state.elementId);
    if (!selected) {
      panel.appendChild(element("p", "empty", "Hover, focus, or click recorded element overlay."));
      return;
    }
    panel.dataset.selectedElementId = selected.id;
    panel.appendChild(element("strong", "selected-element-title", selected.label));
    addFields(panel, [
      ["Element ID", selected.id],
      ["Viewport ID", state.viewportId],
      ["Role", selected.role],
      ["Region", selected.region_id],
      ["Visibility fraction", selected.visibility_fraction],
      ["Actionable", selected.actionable],
      ["Disabled", selected.disabled],
      ["Noticed", selected.noticed],
      ["Inspected", selected.inspected]
    ]);
    var score = findProminence(run, state.viewportId, selected.id);
    if (score) {
      panel.appendChild(element("h4", "", "Prominence contributions"));
      addFields(panel, [
        ["Prominence raw score", score.raw_score],
        ["Normalized probability", score.normalized_probability],
        ["First notice probability", score.first_notice_probability],
        ["Notice within budget probability", score.notice_within_budget_probability]
      ]);
      panel.appendChild(featureTable(score));
    } else {
      panel.appendChild(element("p", "empty", "No prominence record for selected element."));
    }
    addValues(panel, "Scent", scentFor(run, state.viewportId, selected.id));
    addValues(panel, "Related findings", (run.findings || []).filter(function (finding) {
      return (finding.element_ids || []).indexOf(selected.id) !== -1;
    }).map(function (finding) { return finding.title + ": " + finding.cause; }));
  }

  function renderViewports(parent, run) {
    var panel = element("section", "detail-panel viewport-panel");
    panel.appendChild(element("h3", "", "Screenshots and attention overlay"));
    var selectedPanel = element("section", "detail-panel selected-element-panel");
    selectedPanel.id = "selected-element-evidence";
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
        var overlay = element("button", "viewport-overlay" + (item.inspected ? " inspected" : item.noticed ? " noticed" : ""));
        overlay.type = "button";
        overlay.dataset.elementId = item.id;
        overlay.dataset.viewportId = snapshot.id;
        overlay.setAttribute("aria-label", "Inspect evidence for " + item.label);
        var bounds = item.bounds;
        overlay.style.left = (bounds.x / snapshot.viewport.width * 100) + "%";
        overlay.style.top = (bounds.y / snapshot.viewport.height * 100) + "%";
        overlay.style.width = (bounds.width / snapshot.viewport.width * 100) + "%";
        overlay.style.height = (bounds.height / snapshot.viewport.height * 100) + "%";
        overlay.appendChild(element("span", "overlay-label", item.label + (item.inspected ? " | inspected" : item.noticed ? " | noticed" : "")));
        var select = function () {
          state.viewportId = snapshot.id;
          state.elementId = item.id;
          renderSelectedElement(selectedPanel, run);
        };
        overlay.addEventListener("mouseenter", select);
        overlay.addEventListener("focus", select);
        overlay.addEventListener("click", select);
        frame.appendChild(overlay);
      });
      card.appendChild(frame);
      strip.appendChild(card);
    });
    if (!strip.childNodes.length) strip.appendChild(element("p", "empty", "No viewport captures recorded."));
    panel.appendChild(strip);
    parent.appendChild(panel);
    parent.appendChild(selectedPanel);
    if (!state.elementId && run.snapshots && run.snapshots.length && run.snapshots[0].elements.length) {
      state.viewportId = run.snapshots[0].id;
      state.elementId = run.snapshots[0].elements[0].id;
    }
    renderSelectedElement(selectedPanel, run);
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
    addMeta(meta, "Verified", run.verified);
    addMeta(meta, "Terminal state", run.terminal_state);
    detail.appendChild(meta);
    if (run.run_page) {
      var pageLink = document.createElement("a");
      pageLink.href = run.run_page;
      pageLink.textContent = "Open self-contained run page";
      detail.appendChild(pageLink);
    }

    renderViewports(detail, run);
    var grid = element("div", "detail-grid");
    addPanel(grid, "Observations and notice state", run.observations || [], observationCard);
    addPanel(grid, "Attention selections", run.selections || [], selectionCard);
    addPanel(grid, "Scent records", run.scent_records || [], function (record, item) {
      item.appendChild(element("strong", "", "Step " + exact(record.sequence) + " " + titleCase(record.kind)));
      addFields(item, [["Viewport", record.viewport_id]]);
      addValues(item, "Element scores", (record.scores || []).map(function (score) {
        return score.element_id + ": " + exact(score.score);
      }));
    });
    addPanel(grid, "Decisions and reasons", run.decisions || [], decisionCard);
    addPanel(grid, "Actions and results", run.actions || [], actionCard);
    addPanel(grid, "Verification", run.verification ? [run.verification] : [], verificationCard);
    addPanel(grid, "Memory", run.memory || [], memoryCard);
    addPanel(grid, "Model manifests", run.manifests && run.manifests.provider_manifests ? run.manifests.provider_manifests : [], manifestCard);
    addPanel(grid, "Model calls", run.model_calls || [], modelCallCard);
    addPanel(grid, "Evidence", run.evidence || [], evidenceCard);
    addPanel(grid, "Findings", run.findings || [], findingCard, "finding-panel");
    addPanel(grid, "Metrics", run.metrics || [], metricCard);
    addPanel(grid, "Terminal status", [{
      terminal_state: run.terminal_state,
      stage: run.stage,
      outcome: run.outcome,
      failure_reason: run.failure_reason
    }], function (record, item) {
      item.appendChild(element("strong", "", run.failed ? "Run failed or partial" : "Run completed"));
      addFields(item, [
        ["Terminal state", record.terminal_state],
        ["Stage", record.stage],
        ["Outcome", record.outcome],
        ["Failure reason", record.failure_reason]
      ]);
    });
    addPanel(grid, "Limitations", (run.limitations || []).map(function (value) { return { value: value }; }), function (record, item) {
      item.appendChild(element("span", "", record.value));
    });
    detail.appendChild(grid);
  }

  function applyHash() {
    var params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    var runId = params.get("run");
    var elementId = params.get("element");
    if (runId && runs.some(function (run) { return run.run_id === runId; })) state.runId = runId;
    if (elementId) state.elementId = elementId;
  }

  function render() {
    renderRunOptions();
    var run = currentRun();
    if (run && state.elementId && !state.viewportId) {
      (run.snapshots || []).some(function (snapshot) {
        if ((snapshot.elements || []).some(function (item) { return item.id === state.elementId; })) {
          state.viewportId = snapshot.id;
          return true;
        }
        return false;
      });
    }
    renderTimeline(run);
    renderRun(run);
  }

  fillSelect(versionFilter, uniqueValues("version_id"), "All versions");
  fillSelect(policyFilter, uniqueValues("policy"), "All policies");
  [runFilter, versionFilter, policyFilter].forEach(function (control) {
    control.addEventListener("input", render);
    control.addEventListener("change", render);
  });
  runSelect.addEventListener("change", function () {
    state.runId = runSelect.value;
    state.eventIndex = 0;
    state.viewportId = "";
    state.elementId = "";
    render();
  });
  document.getElementById("timeline-prev").addEventListener("click", function () {
    if (state.eventIndex > 0) { state.eventIndex -= 1; render(); }
  });
  document.getElementById("timeline-next").addEventListener("click", function () {
    var run = currentRun();
    if (run && state.eventIndex < (run.timeline || []).length - 1) { state.eventIndex += 1; render(); }
  });
  window.addEventListener("hashchange", function () { applyHash(); render(); });
  applyHash();
  render();
}());
