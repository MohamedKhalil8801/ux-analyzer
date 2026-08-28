(function () {
  "use strict";

  var dataNode = document.getElementById("report-data");
  if (!dataNode) return;
  var data = JSON.parse(dataNode.textContent || "{}");
  var runs = data.runs || [];
  var state = {
    runId: runs.length ? runs[0].run_id : "",
    scenarioId: "",
    eventIndex: 0,
    viewportId: "",
    elementId: "",
    saliencyKey: "",
    evidenceId: "",
    comparisonKey: "",
    comparisonDuration: "1s",
    playing: false,
    timer: null
  };

  var scenarioSelect = document.getElementById("scenario-select");
  var runSelect = document.getElementById("run-select");
  var playPause = document.getElementById("play-pause");
  var playLabel = document.getElementById("play-label");
  var progress = document.getElementById("playback-progress");
  var position = document.getElementById("playback-position");
  var statusBanner = document.getElementById("run-status-banner");
  var viewportStage = document.getElementById("viewport-stage");
  var viewportMeta = document.getElementById("viewport-meta");
  var eventKind = document.getElementById("event-kind");
  var eventCard = document.getElementById("current-event-card");
  var elementPanel = document.getElementById("selected-element-evidence");
  var elementDetail = document.getElementById("element-detail");
  var elementState = document.getElementById("element-state");
  var timelineList = document.getElementById("timeline-list");
  var timelineCount = document.getElementById("timeline-count");
  var saliencyTabs = document.getElementById("saliency-tabs");
  var saliencyDetail = document.getElementById("saliency-detail");
  var saliencyRuntimeLabel = document.getElementById("saliency-runtime-label");
  var comparisonSelect = document.getElementById("provider-comparison-select");
  var comparisonDurationSelect = document.getElementById("provider-comparison-duration");
  var comparisonOutput = document.getElementById("provider-comparison-output");
  var evidenceContext = document.getElementById("evidence-context");
  var evidenceDetail = document.getElementById("evidence-detail");
  var evidenceDetailSummary = document.getElementById("evidence-detail-summary");
  var evidenceDetailFields = document.getElementById("evidence-detail-fields");
  var providerComparisons = data.provider_comparisons || [];

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function exact(value) {
    if (value === undefined || value === null || value === "") return "unavailable";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  }

  function percentage(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "unavailable";
    return String(value * 100) + "%";
  }

  function titleCase(value) {
    return String(value || "record").replace(/[-_]/g, " ").replace(/\b\w/g, function (letter) {
      return letter.toUpperCase();
    });
  }

  function safeJson(value) {
    if (value === undefined || value === null) return "unavailable";
    try { return JSON.stringify(value, null, 2); }
    catch (_error) { return "unavailable: invalid recorded summary"; }
  }

  function currentRun() {
    return runs.find(function (run) { return run.run_id === state.runId; }) || null;
  }

  function currentEvent(run) {
    return run && run.timeline ? run.timeline[state.eventIndex] || null : null;
  }

  function elementRecord(run, elementId) {
    var snapshots = run && run.snapshots || [];
    for (var snapshotIndex = 0; snapshotIndex < snapshots.length; snapshotIndex += 1) {
      var match = (snapshots[snapshotIndex].elements || []).find(function (item) {
        return item.id === elementId;
      });
      if (match) return match;
    }
    return null;
  }

  function elementLabel(run, elementId) {
    var match = elementRecord(run, elementId);
    return match ? match.label : "the selected element";
  }

  function runLabel(run) {
    if (!run) return "Run unavailable";
    return run.scenario_label + " | " + run.version_label + " | " + run.persona_label;
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

  function addValues(parent, label, values, unavailableReason) {
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    var list = element("ul", "compact-list");
    (values || []).forEach(function (value) { list.appendChild(element("li", "", value)); });
    if (!list.childNodes.length) {
      list.appendChild(element("li", "empty", unavailableReason || "Unavailable: no recorded values."));
    }
    block.appendChild(list);
    parent.appendChild(block);
  }

  function regionLabel(run, regionId) {
    var snapshots = run && run.snapshots || [];
    for (var snapshotIndex = 0; snapshotIndex < snapshots.length; snapshotIndex += 1) {
      var match = (snapshots[snapshotIndex].regions || []).find(function (item) {
        return item.id === regionId;
      });
      if (match) return match.label || "Website section";
    }
    return "Website section";
  }

  function mappingKeyLabel(run, label, key) {
    if (label === "Element probabilities") return elementLabel(run, key);
    if (label === "Region probabilities") return regionLabel(run, key);
    return titleCase(key);
  }

  function addMapping(parent, label, mapping, run) {
    var entries = Object.keys(mapping || {}).sort();
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    if (!entries.length) {
      block.appendChild(element("p", "empty", "Unavailable: no recorded values."));
    } else {
      var list = element("dl", "field-list");
      entries.forEach(function (key) { addField(list, mappingKeyLabel(run, label, key), mapping[key]); });
      block.appendChild(list);
    }
    parent.appendChild(block);
  }

  function addJson(parent, label, value) {
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    block.appendChild(element("pre", "json-summary", safeJson(value)));
    parent.appendChild(block);
  }

  function actionText(action, run) {
    if (!action) return "Unavailable: no recorded action.";
    var text = titleCase(action.kind || "action");
    if (action.element_id) text += " on " + elementLabel(run, action.element_id);
    if (action.direction) text += " " + action.direction;
    if (action.duration_seconds !== undefined) text += " for " + action.duration_seconds + " seconds";
    return text;
  }

  function eventCategory(kind) {
    if (kind === "viewport-captured" || kind === "observation-recorded") return "Page observation";
    if (kind.indexOf("prominence") !== -1 || kind.indexOf("scent") !== -1) return "Element review";
    if (kind === "attention-selection-recorded") return "Next step selected";
    if (kind === "model-call-recorded" || kind === "decision-recorded" || kind === "agent-claim") return "Website review";
    if (kind === "action-proposed" || kind === "action-executed" || kind === "action-rejected") return "Interaction";
    if (kind === "verification-recorded") return "Task result";
    if (kind === "run-terminated") return "Task completed";
    if (kind.indexOf("failure") !== -1) return "Recorded issue";
    return "Recorded step";
  }

  function eventTitle(kind) {
    if (kind === "viewport-captured") return "Page viewed";
    if (kind === "observation-recorded") return "Page inspected";
    if (kind.indexOf("prominence") !== -1 || kind.indexOf("scent") !== -1) return "Elements reviewed";
    if (kind === "attention-selection-recorded") return "Next interaction selected";
    if (kind === "model-call-recorded" || kind === "decision-recorded" || kind === "agent-claim") return "Website review recorded";
    if (kind === "action-executed") return "Interaction completed";
    if (kind === "action-proposed") return "Next interaction planned";
    if (kind === "action-rejected") return "Interaction not completed";
    if (kind === "verification-recorded") return "Task result checked";
    if (kind === "run-terminated") return "Task completed";
    if (kind.indexOf("failure") !== -1) return "Issue recorded";
    return "Recorded step";
  }

  function eventSummary(record, run) {
    if (!record) return "Unavailable";
    if (record.reason) return String(record.reason);
    if (record.action) return actionText(record.action, run);
    if (record.outcome) return "Outcome: " + record.outcome;
    if (record.viewport_id) return "Website view recorded";
    if (record.selected_ids && record.selected_ids.length) return "Selected: " + record.selected_ids.map(function (id) { return elementLabel(run, id); }).join(", ");
    return eventCategory(record.kind);
  }

  function memoryLabel(item, run) {
    var key = String(item && item.key || "");
    var value = String(item && item.value || "");
    if (key.indexOf("-element-") !== -1) {
      var label = elementLabel(run, key);
      return label === "the selected element" ? (value || "Recorded element") : label;
    }
    return key + (value ? ": " + value : "");
  }

  function scenarioValues() {
    return runs.map(function (run) { return run.scenario_id; }).filter(function (value, index, values) {
      return value && values.indexOf(value) === index;
    }).sort();
  }

  function fillScenarioOptions() {
    scenarioValues().forEach(function (value) {
      var matching = runs.find(function (run) { return run.scenario_id === value; });
      var option = document.createElement("option");
      option.value = value;
      option.textContent = matching ? matching.scenario_label : value;
      scenarioSelect.appendChild(option);
    });
  }

  function visibleRuns() {
    return runs.filter(function (run) {
      return !state.scenarioId || run.scenario_id === state.scenarioId;
    });
  }

  function renderRunOptions() {
    var visible = visibleRuns();
    while (runSelect.firstChild) runSelect.removeChild(runSelect.firstChild);
    visible.forEach(function (run) {
      var option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = runLabel(run) + " | " + run.outcome;
      runSelect.appendChild(option);
    });
    if (!visible.some(function (run) { return run.run_id === state.runId; })) {
      state.runId = visible.length ? visible[0].run_id : "";
      state.eventIndex = 0;
      state.viewportId = "";
      state.elementId = "";
      state.saliencyKey = "";
    }
    runSelect.value = state.runId;
    scenarioSelect.value = state.scenarioId;
  }

  function setPlaying(playing) {
    state.playing = Boolean(playing);
    playPause.setAttribute("aria-pressed", state.playing ? "true" : "false");
    playLabel.textContent = state.playing ? "Pause" : "Play";
    playPause.querySelector(".play-icon").textContent = state.playing ? "Ⅱ" : "▶";
    elementState.textContent = state.playing ? "Playing" : "Paused";
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    if (state.playing) {
      state.timer = window.setInterval(function () {
        var run = currentRun();
        var lastIndex = run && run.timeline ? run.timeline.length - 1 : -1;
        if (state.eventIndex >= lastIndex) {
          setPlaying(false);
          return;
        }
        setEventIndex(state.eventIndex + 1, true);
      }, 1100);
    }
  }

  function recordedTime(event) {
    if (!event) return "time unavailable";
    return event.timestamp || event.recorded_at || event.time || "time unavailable";
  }

  function updateUrl(run, event, sectionId, push) {
    if (!run) return;
    var url = new URL(window.location.href);
    url.searchParams.set("run", run.run_id);
    if (event && event.event_id) url.searchParams.set("event", event.event_id);
    else url.searchParams.delete("event");
    if (state.elementId) url.searchParams.set("element", state.elementId);
    else url.searchParams.delete("element");
    if (state.evidenceId) url.searchParams.set("evidence", state.evidenceId);
    else url.searchParams.delete("evidence");
    if (sectionId) url.hash = sectionId;
    window.history[push ? "pushState" : "replaceState"](
      { reportState: true },
      "",
      url.href
    );
  }

  function setEventIndex(index, keepPlaying) {
    var run = currentRun();
    var events = run ? run.timeline || [] : [];
    state.eventIndex = Math.max(0, Math.min(Number(index) || 0, Math.max(events.length - 1, 0)));
    var snapshot = snapshotAt(run, state.eventIndex);
    if (!snapshot || !findElement(snapshot, state.elementId)) {
      state.elementId = "";
    }
    state.viewportId = snapshot ? snapshot.id : "";
    if (!keepPlaying) setPlaying(false);
    renderWorkspace();
  }

  function selectRun(runId, scrollWorkspace) {
    var run = runs.find(function (item) { return item.run_id === runId; });
    if (!run) return;
    if (run.run_page && !(run.timeline || []).length) {
      window.location.href = run.run_page + "?run=" + encodeURIComponent(run.run_id) + "#playback-workspace";
      return;
    }
    setPlaying(false);
    state.runId = run.run_id;
    state.scenarioId = run.scenario_id;
    state.eventIndex = 0;
    state.viewportId = "";
    state.elementId = "";
    state.evidenceId = "";
    clearEvidenceDestination();
    renderRunOptions();
    renderWorkspace();
    if (scrollWorkspace) {
      showViewForNode(document.getElementById("playback-workspace"));
      document.getElementById("playback-workspace").scrollIntoView({ block: "start" });
    }
  }

  function jumpToSequence(runId, sequence) {
    var run = runs.find(function (item) { return item.run_id === runId; });
    if (!run) return;
    if (!(run.timeline || []).length) {
      if (run.run_page) window.location.href = run.run_page + "?run=" + encodeURIComponent(run.run_id) + "&event=event-" + encodeURIComponent(sequence) + "#playback-workspace";
      return;
    }
    selectRun(runId, false);
    var index = run.timeline.findIndex(function (record) {
      return Number(record.sequence) === Number(sequence);
    });
    setEventIndex(index >= 0 ? index : 0, false);
    showViewForNode(document.getElementById("playback-workspace"));
    document.getElementById("playback-workspace").scrollIntoView({ block: "start" });
  }

  function findSnapshot(run, viewportId) {
    return run && (run.snapshots || []).find(function (snapshot) { return snapshot.id === viewportId; }) || null;
  }

  function eventIndexForEvidence(run, target) {
    if (!run) return null;
    var events = run.timeline || [];
    if (target.sequence !== undefined && target.sequence !== null) {
      var bySequence = events.findIndex(function (record) {
        return Number(record.sequence) === Number(target.sequence);
      });
      if (bySequence >= 0) return bySequence;
    }
    if (target.event_id) {
      var byId = events.findIndex(function (record) {
        return record.event_id === target.event_id;
      });
      if (byId >= 0) return byId;
    }
    if (target.viewport_id) {
      var byViewport = events.findIndex(function (record) {
        return record.viewport_id === target.viewport_id ||
          (record.snapshot && record.snapshot.id === target.viewport_id) ||
          (record.observation && record.observation.viewport_id === target.viewport_id);
      });
      if (byViewport >= 0) return byViewport;
    }
    if (target.namespace) {
      var group = (run.saliency || []).find(function (item) {
        return item.artifact_namespace === target.namespace || item.viewport_id === target.namespace;
      });
      if (group) {
        var sourceEvent = (group.source_event_ids || [])[0];
        if (sourceEvent) {
          var sourceIndex = events.findIndex(function (record) {
            return record.event_id === sourceEvent;
          });
          if (sourceIndex >= 0) return sourceIndex;
        }
      }
    }
    return null;
  }

  function evidenceControl(evidenceId) {
    var controls = document.querySelectorAll(".evidence-ref");
    for (var index = 0; index < controls.length; index += 1) {
      if (controls[index].dataset.evidenceId === evidenceId) return controls[index];
    }
    return null;
  }

  function resolveEvidence(ref) {
    if (typeof ref === "string") {
      var control = evidenceControl(ref);
      if (!control) return { evidenceId: ref, target: {} };
      var controlTarget = {};
      try { controlTarget = JSON.parse(control.dataset.evidenceTarget || "{}"); }
      catch (_error) { controlTarget = {}; }
      return { evidenceId: ref, target: controlTarget };
    }
    if (!ref || typeof ref !== "object") return { evidenceId: "", target: {} };
    if (ref.target) {
      return {
        evidenceId: ref.evidence_id || ref.evidenceId || "",
        target: ref.target
      };
    }
    return {
      evidenceId: ref.evidence_id || ref.evidenceId || "",
      target: ref
    };
  }

  function evidenceLink(run, target, evidenceId) {
    var params = new URLSearchParams();
    if (run && run.run_id) params.set("run", run.run_id);
    if (target.event_id) params.set("event", target.event_id);
    else if (target.sequence !== undefined && target.sequence !== null) {
      params.set("event", "event-" + target.sequence);
    }
    if (target.element_id) params.set("element", target.element_id);
    if (evidenceId) params.set("evidence", evidenceId);
    return "?" + params.toString() + "#playback-workspace";
  }

  function normalizeEvidenceTarget(run, target, evidenceId) {
    var normalized = Object.assign({}, target || {});
    if (normalized.kind !== "heatmap" && normalized.kind !== "native-map" && normalized.kind !== "saliency-metadata") {
      return normalized;
    }
    var evidenceParts = String(evidenceId || "").split(":");
    if (!normalized.duration && evidenceParts.length >= 4 && evidenceParts[0] === "heatmap") {
      normalized.duration = evidenceParts[evidenceParts.length - 1];
    }
    var group = (run && run.saliency || []).find(function (item) {
      return item.viewport_id === normalized.viewport_id ||
        item.artifact_namespace === normalized.namespace;
    });
    if (group && (!normalized.namespace || normalized.namespace === normalized.viewport_id)) {
      normalized.namespace = group.artifact_namespace || group.viewport_id;
    }
    return normalized;
  }

  function clearEvidenceDestination() {
    document.querySelectorAll('[data-viewing-evidence="true"]').forEach(function (node) {
      node.removeAttribute("data-viewing-evidence");
    });
    if (evidenceDetail) {
      evidenceDetail.hidden = true;
      delete evidenceDetail.dataset.evidenceId;
    }
    if (evidenceContext) {
      evidenceContext.hidden = true;
      evidenceContext.textContent = "";
    }
  }

  function showEvidenceContext(evidenceId) {
    if (!evidenceContext || !evidenceId) return;
    evidenceContext.hidden = false;
    evidenceContext.textContent = "Evidence opened in the workspace below.";
  }

  function showEvidenceDetail(evidenceId, target) {
    if (!evidenceDetail) return null;
    evidenceDetail.hidden = false;
    evidenceDetail.dataset.evidenceId = evidenceId;
    evidenceDetailSummary.textContent = target.summary || "Recorded evidence detail.";
    while (evidenceDetailFields.firstChild) evidenceDetailFields.removeChild(evidenceDetailFields.firstChild);
    var values = target.detail && typeof target.detail === "object" ? target.detail : {};
    Object.keys(values).sort().forEach(function (key) {
      addField(evidenceDetailFields, titleCase(key), safeJson(values[key]));
    });
    return evidenceDetail;
  }

  function metricDestination(runId, metricId) {
    return document.querySelector(
      'tr[data-run-id="' + CSS.escape(runId) + '"] [data-metric="' + CSS.escape(metricId) + '"]'
    );
  }

  function focusEvidenceDestination(target, evidenceId, options) {
    var destination = null;
    if (target.kind === "metric") destination = metricDestination(target.run_id, target.metric_id);
    else if (target.kind === "evidence-detail") destination = showEvidenceDetail(evidenceId, target);
    else if (target.kind === "event" || target.kind === "replay") destination = eventCard;
    else if (target.kind === "viewport" || target.kind === "screenshot") destination = viewportStage;
    else if (target.duration) destination = saliencyTabs.querySelector('[aria-selected="true"]');
    else if (target.element_id) destination = elementPanel;
    else destination = eventCard;
    if (!destination) return;
    destination.dataset.viewingEvidence = "true";
    if (!destination.hasAttribute("tabindex")) destination.setAttribute("tabindex", "-1");
    destination.scrollIntoView({ block: "center" });
    destination.focus({ preventScroll: true });
  }

  function evidenceSection(target) {
    if (target.kind === "metric") return "#comparison-overview";
    if (target.kind === "evidence-detail") return "#evidence-detail";
    return "#playback-workspace";
  }

  function openEvidence(ref, options) {
    options = options || {};
    var previousUrl = window.location.href;
    var resolved = resolveEvidence(ref);
    var evidenceId = resolved.evidenceId;
    var target = resolved.target || {};
    if (!evidenceId && target.evidence_id) evidenceId = target.evidence_id;
    var runId = target.run_id || target.runId;
    var run = runs.find(function (item) { return item.run_id === runId; });
    if (!run) return false;
    target = normalizeEvidenceTarget(run, target, evidenceId);

    state.evidenceId = evidenceId;
    state.runId = run.run_id;
    state.scenarioId = run.scenario_id;
    var targetEventIndex = eventIndexForEvidence(run, target);
    if (targetEventIndex !== null) state.eventIndex = targetEventIndex;
    state.viewportId = "";
    state.elementId = "";
    if (target.namespace && target.duration) {
      state.saliencyKey = target.namespace + ":" + target.duration;
    }
    renderRunOptions();

    if (run.run_page && !(run.timeline || []).length) {
      window.location.href = run.run_page + evidenceLink(run, target, evidenceId);
      return true;
    }

    clearEvidenceDestination();
    showEvidenceContext(evidenceId);
    setEventIndex(state.eventIndex, false);
    var snapshot = snapshotAt(run, state.eventIndex);
    if (target.viewport_id && snapshot && snapshot.id !== target.viewport_id) {
      var viewportIndex = eventIndexForEvidence(run, { viewport_id: target.viewport_id });
      if (viewportIndex !== null) setEventIndex(viewportIndex, false);
      snapshot = snapshotAt(run, state.eventIndex);
    }
    if (target.element_id && snapshot && findElement(snapshot, target.element_id)) {
      state.viewportId = snapshot.id;
      state.elementId = target.element_id;
      renderWorkspace();
    }
    var section = evidenceSection(target);
    if (options.pushHistory !== false) {
      window.history.replaceState({ reportState: true }, "", previousUrl);
    }
    updateUrl(run, currentEvent(run), section, options.pushHistory !== false);
    focusEvidenceDestination(target, evidenceId, options);
    return true;
  }

  function snapshotAt(run, eventIndex) {
    if (!run) return null;
    var viewportId = "";
    (run.timeline || []).slice(0, eventIndex + 1).forEach(function (record) {
      if (record.kind === "viewport-captured" && record.viewport_id) viewportId = record.viewport_id;
    });
    return viewportId ? findSnapshot(run, viewportId) : null;
  }

  function findElement(snapshot, elementId) {
    return snapshot && (snapshot.elements || []).find(function (item) { return item.id === elementId; }) || null;
  }

  function noticedState(run, eventIndex) {
    var noticed = {};
    var inspected = {};
    (run.timeline || []).slice(0, eventIndex + 1).forEach(function (record) {
      if (record.kind === "observation-recorded" && record.observation) {
        (record.observation.newly_revealed_elements || []).forEach(function (item) { noticed[item.id] = true; });
      }
      if (record.action && record.action.element_id && String(record.action.kind || "").indexOf("inspect") !== -1) {
        inspected[record.action.element_id] = true;
      }
    });
    return { noticed: noticed, inspected: inspected };
  }

  function scoreAt(run, viewportId, elementId, sequence) {
    var records = (run.prominence || []).filter(function (record) {
      return record.sequence <= sequence && (!record.viewport_id || record.viewport_id === viewportId);
    });
    for (var index = records.length - 1; index >= 0; index -= 1) {
      var score = (records[index].scores || []).find(function (item) { return item.element_id === elementId; });
      if (score) return score;
    }
    return null;
  }

  function scentAt(run, viewportId, elementId, sequence) {
    return (run.scent_records || []).filter(function (record) {
      return record.sequence <= sequence && (!record.viewport_id || record.viewport_id === viewportId);
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

  function linkedRecords(run, elementId) {
    var decisions = (run.decisions || []).filter(function (record) { return record.action && record.action.element_id === elementId; });
    var actions = (run.actions || []).filter(function (record) { return record.action && record.action.element_id === elementId; });
    var findings = (run.findings || []).filter(function (record) { return (record.element_ids || []).indexOf(elementId) !== -1; });
    var eventIds = decisions.concat(actions).map(function (record) { return "event-" + record.sequence; });
    var findingEvidence = findings.reduce(function (ids, finding) { return ids.concat(finding.evidence_ids || []); }, []);
    var evidence = (run.evidence || []).filter(function (record) {
      return (record.source_event_ids || []).some(function (id) { return eventIds.indexOf(id) !== -1; }) || findingEvidence.indexOf(record.evidence_id) !== -1;
    });
    return { decisions: decisions, actions: actions, findings: findings, evidence: evidence };
  }

  function linkedEvidenceText(record) {
    var description = String(record && record.description || "");
    var normalized = description.toLowerCase();
    if (normalized.indexOf("unique element ids") !== -1) return "Recorded elements inspected during discovery.";
    if (normalized.indexOf("unique region ids") !== -1) return "Website sections observed during discovery.";
    if (normalized.indexOf("executed scroll") !== -1) return "Scrolls used while finding the target.";
    if (normalized.indexOf("failed or non-target") !== -1) return "Wrong or unsuccessful interactions recorded.";
    if (normalized.indexOf("executed back") !== -1) return "Back actions used during discovery.";
    if (normalized.indexOf("model review unavailable") !== -1) return "Recorded fallback signal attached to this element.";
    return description;
  }

  function renderElementDetail(run, snapshot, event) {
    while (elementDetail.firstChild) elementDetail.removeChild(elementDetail.firstChild);
    var selected = findElement(snapshot, state.elementId);
    if (!selected) {
      delete elementPanel.dataset.selectedElementId;
      elementDetail.appendChild(element("p", "empty", snapshot ? "Unavailable: no recorded element selected." : "Unavailable: current event has no recorded viewport."));
      return;
    }
    elementPanel.dataset.selectedElementId = selected.id;
    elementDetail.appendChild(element("strong", "selected-element-title", selected.label));
    var region = (snapshot.regions || []).find(function (item) { return item.id === selected.region_id; });
    var attention = noticedState(run, state.eventIndex);
    addFields(elementDetail, [
      ["Type", titleCase(selected.role)],
      ["Section", region ? region.label : "Website section unavailable"],
      ["Clickable", selected.actionable],
      ["Disabled", selected.disabled],
      ["Visible on screen", percentage(selected.visibility_fraction)],
      ["Blocked by other content", percentage(selected.occlusion_fraction)],
      ["Local contrast", selected.local_contrast],
      ["Noticed before this step", Boolean(attention.noticed[selected.id])],
      ["Inspected before this step", Boolean(attention.inspected[selected.id])]
    ]);
    var score = scoreAt(run, snapshot.id, selected.id, event ? event.sequence : 0);
    if (score) {
      elementDetail.appendChild(element("h4", "", "Prominence contributions"));
      addFields(elementDetail, [
        ["Prominence raw score", score.raw_score],
        ["Normalized probability", score.normalized_probability],
        ["First notice probability", score.first_notice_probability],
        ["Notice within budget probability", score.notice_within_budget_probability]
      ]);
      elementDetail.appendChild(featureTable(score));
    }
    var scentValues = scentAt(run, snapshot.id, selected.id, event ? event.sequence : 0);
    if (scentValues.length) addValues(elementDetail, "Scent", scentValues);
    var linked = linkedRecords(run, selected.id);
    if (linked.decisions.length) addValues(elementDetail, "Linked decisions", linked.decisions.map(function (record) { return "Step " + record.sequence + ": " + (record.reason || actionText(record.action, run)); }));
    if (linked.actions.length) addValues(elementDetail, "Linked actions and results", linked.actions.map(function (record) { return "Step " + record.sequence + ": " + actionText(record.action, run) + " | " + (record.succeeded ? "succeeded" : "failed") + (record.error ? " | " + record.error : ""); }));
    if (linked.findings.length) addValues(elementDetail, "Related findings", linked.findings.map(function (record) { return record.title + ": " + record.cause; }));
    if (linked.evidence.length) addValues(elementDetail, "Related measurements", linked.evidence.map(linkedEvidenceText));
  }

  function saliencyEntries(run) {
    var entries = [];
    (run && run.saliency || []).forEach(function (group) {
      (group.entries || []).forEach(function (entry) {
        entries.push({ group: group, entry: entry, key: group.artifact_namespace + ":" + entry.duration });
      });
    });
    return entries;
  }

  function appendComparisonRankings(parent, title, rankings) {
    var block = element("div", "comparison-block");
    block.appendChild(element("strong", "", title));
    if (!rankings || !rankings.length) {
      block.appendChild(element("p", "empty", "No ranking."));
      parent.appendChild(block);
      return;
    }
    var table = element("table", "feature-table comparison-ranking-table");
    var head = element("thead", "");
    var headRow = element("tr", "");
    ["Rank", "Element", "Role", "Score", "Probability"].forEach(function (label) {
      headRow.appendChild(element("th", "", label));
    });
    head.appendChild(headRow);
    table.appendChild(head);
    var body = element("tbody", "");
    rankings.forEach(function (item) {
      var row = element("tr", "");
      row.appendChild(element("td", "", item.rank));
      row.appendChild(element("td", "", item.label || "Unlabelled element"));
      row.appendChild(element("td", "", item.role));
      row.appendChild(element("td", "", exact(item.adjusted_score !== undefined ? item.adjusted_score : item.score)));
      row.appendChild(element("td", "", exact(item.normalized_probability)));
      body.appendChild(row);
    });
    table.appendChild(body);
    block.appendChild(table);
    parent.appendChild(block);
  }

  function appendComparisonHeatmap(parent, provider, duration) {
    var block = element("div", "comparison-block");
    block.appendChild(element("strong", "", "Exact generated heatmap"));
    var heatmap = (provider.heatmaps || []).find(function (item) {
      return item.duration === duration;
    });
    if (!heatmap || !heatmap.heatmap) {
      block.appendChild(element("p", "empty", provider.provider_id === "heuristic"
        ? "Heuristic has no pixel heatmap."
        : "No validated heatmap recorded."));
      parent.appendChild(block);
      return;
    }
    var image = document.createElement("img");
    image.className = "provider-heatmap exact-heatmap";
    image.src = heatmap.heatmap;
    image.alt = "Exact " + provider.provider_id + " heatmap for " + heatmap.duration;
    block.appendChild(image);
    var link = document.createElement("a");
    link.href = heatmap.heatmap;
    link.target = "_blank";
    link.textContent = "Open exact generated heatmap";
    block.appendChild(link);
    addFields(block, [
      ["Artifact", "Recorded heatmap available"],
      ["Captured page", "Recorded website view"],
      ["Inference timing", exact(heatmap.inference_duration_ms) + " ms"],
      ["Execution provider", heatmap.execution_provider],
      ["Cache status", heatmap.cache_state]
    ]);
    appendComparisonRankings(block, "Foveacast element ranking", heatmap.ranked_elements || []);
    parent.appendChild(block);
  }

  function appendComparisonPath(parent, provider) {
    var block = element("div", "comparison-block");
    block.appendChild(element("strong", "", "Recorded action path"));
    var path = provider.action_path || [];
    if (!path.length) {
      block.appendChild(element("p", "empty", "No recorded actions."));
      parent.appendChild(block);
      return;
    }
    var list = element("ol", "comparison-path");
    path.forEach(function (item) {
      var row = element("li", "");
      var button = element("button", "path-step");
      button.type = "button";
      button.textContent = "Step " + item.sequence + " | " + titleCase(item.kind) + " | " + item.element_label;
      button.addEventListener("click", function () {
        jumpToSequence(provider.run_id, item.sequence);
      });
      row.appendChild(button);
      var status = item.succeeded === true ? "succeeded" : item.succeeded === false ? "failed" : "recorded";
      row.appendChild(element("span", "path-status path-status-" + status, status));
      if (item.error) row.appendChild(element("span", "path-error", item.error));
      list.appendChild(row);
    });
    block.appendChild(list);
    parent.appendChild(block);
  }

  function renderProviderCard(parent, provider, duration) {
    var card = element("article", "provider-card");
    var heading = element("div", "provider-card-heading");
    var title = element("h3", "", titleCase(provider.provider_id));
    heading.appendChild(title);
    var status = element("span", "provider-status", provider.outcome);
    heading.appendChild(status);
    card.appendChild(heading);
    if (provider.run_page) {
      var runLink = document.createElement("a");
      runLink.href = provider.run_page + "?run=" + encodeURIComponent(provider.run_id) + "#playback-workspace";
      runLink.textContent = "Open run workspace";
      card.appendChild(runLink);
    }
    addFields(card, [
      ["Verified", provider.verified],
      ["Failure", provider.failure_reason || "None recorded"]
    ]);
    appendComparisonHeatmap(card, provider, duration);
    if (provider.provider_id === "heuristic") {
      var prominence = (provider.prominence || [])[0];
      appendComparisonRankings(card, "Heuristic prominence ranking", prominence ? prominence.rankings : []);
    }
    appendComparisonPath(card, provider);
    parent.appendChild(card);
  }

  function selectedProviderComparison() {
    return providerComparisons.find(function (item) { return item.key === state.comparisonKey; }) || null;
  }

  function fillComparisonDurationOptions(comparison) {
    if (!comparisonDurationSelect) return;
    var durations = [];
    (comparison ? comparison.providers || [] : []).forEach(function (provider) {
      (provider.heatmaps || []).forEach(function (item) {
        if (durations.indexOf(item.duration) === -1) durations.push(item.duration);
      });
    });
    durations.sort(function (left, right) { return ["1s", "3s", "7s"].indexOf(left) - ["1s", "3s", "7s"].indexOf(right); });
    while (comparisonDurationSelect.firstChild) comparisonDurationSelect.removeChild(comparisonDurationSelect.firstChild);
    durations.forEach(function (duration) {
      var option = document.createElement("option");
      option.value = duration;
      option.textContent = duration;
      comparisonDurationSelect.appendChild(option);
    });
    comparisonDurationSelect.disabled = !durations.length;
    if (durations.indexOf(state.comparisonDuration) === -1) state.comparisonDuration = durations[0] || "1s";
    comparisonDurationSelect.value = state.comparisonDuration;
  }

  function renderProviderComparison() {
    if (!comparisonOutput) return;
    while (comparisonOutput.firstChild) comparisonOutput.removeChild(comparisonOutput.firstChild);
    var comparison = selectedProviderComparison();
    if (!comparison) {
      comparisonOutput.appendChild(element("p", "empty", "No paired provider runs recorded."));
      return;
    }
    var heading = element("div", "comparison-cell-heading");
    heading.appendChild(element("h3", "", comparison.scenario_label + " | " + comparison.version_label));
    heading.appendChild(element("p", "muted", "Seed " + comparison.seed + " | " + comparison.policy));
    comparisonOutput.appendChild(heading);
    var grid = element("div", "provider-comparison-grid");
    (comparison.providers || []).forEach(function (provider) {
      renderProviderCard(grid, provider, state.comparisonDuration);
    });
    comparisonOutput.appendChild(grid);
  }

  function fillProviderComparisonOptions() {
    if (!comparisonSelect) return;
    while (comparisonSelect.firstChild) comparisonSelect.removeChild(comparisonSelect.firstChild);
    providerComparisons.forEach(function (comparison) {
      var option = document.createElement("option");
      option.value = comparison.key;
      option.textContent = comparison.scenario_label + " | " + comparison.version_label + " | seed " + comparison.seed;
      comparisonSelect.appendChild(option);
    });
    state.comparisonKey = providerComparisons.length ? providerComparisons[0].key : "";
    comparisonSelect.value = state.comparisonKey;
    fillComparisonDurationOptions(selectedProviderComparison());
    renderProviderComparison();
  }

  function appendRankedElements(parent, ranked, run, group) {
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", "Ranked elements"));
    if (!ranked || !ranked.length) {
      block.appendChild(element("p", "empty", "Unavailable: no element aggregates recorded."));
      parent.appendChild(block);
      return;
    }
    var table = element("table", "feature-table saliency-table");
    var head = element("thead", "");
    var row = element("tr", "");
    ["Rank", "Element", "Role", "Adjusted score", "Bounds"].forEach(function (label) {
      row.appendChild(element("th", "", label));
    });
    head.appendChild(row);
    table.appendChild(head);
    var body = element("tbody", "");
    ranked.forEach(function (item) {
      var itemRow = element("tr", "");
      itemRow.appendChild(element("td", "", item.rank));
      var elementCell = element("td", "");
      var inspect = element("button", "ranked-element-button", item.label);
      inspect.type = "button";
      inspect.setAttribute("aria-label", "Inspect " + item.label + " on the screenshot");
      var selectRankedElement = function () {
        var snapshot = findSnapshot(run, group && group.viewport_id);
        if (snapshot) selectElement(run, snapshot, currentEvent(run), item.element_id);
      };
      inspect.addEventListener("click", selectRankedElement);
      inspect.addEventListener("mouseenter", selectRankedElement);
      elementCell.appendChild(inspect);
      itemRow.appendChild(elementCell);
      itemRow.appendChild(element("td", "", item.role));
      itemRow.appendChild(element("td", "", exact(item.adjusted_score)));
      var bounds = item.bounds || {};
      itemRow.appendChild(element("td", "", bounds.x === undefined ? "unavailable" : "x=" + bounds.x + ", y=" + bounds.y + ", w=" + bounds.width + ", h=" + bounds.height));
      body.appendChild(itemRow);
    });
    table.appendChild(body);
    var tableWrap = element("div", "table-wrap saliency-table-wrap");
    tableWrap.appendChild(table);
    block.appendChild(tableWrap);
    parent.appendChild(block);
  }

  function appendAggregationComponents(parent, aggregates, run) {
    var block = element("details", "value-block");
    block.appendChild(element("summary", "", "Aggregation components"));
    var table = element("table", "feature-table saliency-table");
    var head = element("thead", "");
    var row = element("tr", "");
    ["Element", "Density", "P95 peak", "Mass share", "Raw", "Adjusted"].forEach(function (label) {
      row.appendChild(element("th", "", label));
    });
    head.appendChild(row);
    table.appendChild(head);
    var body = element("tbody", "");
    (aggregates || []).forEach(function (item) {
      var itemRow = element("tr", "");
      itemRow.appendChild(element("td", "", elementLabel(run, item.element_id)));
      itemRow.appendChild(element("td", "", exact(item.density)));
      itemRow.appendChild(element("td", "", exact(item.robust_peak)));
      itemRow.appendChild(element("td", "", exact(item.mass_share)));
      itemRow.appendChild(element("td", "", exact(item.raw_score)));
      itemRow.appendChild(element("td", "", exact(item.adjusted_score)));
      body.appendChild(itemRow);
    });
    table.appendChild(body);
    var tableWrap = element("div", "table-wrap saliency-table-wrap");
    tableWrap.appendChild(table);
    block.appendChild(tableWrap);
    parent.appendChild(block);
  }

  function renderSaliency(run) {
    while (saliencyTabs.firstChild) saliencyTabs.removeChild(saliencyTabs.firstChild);
    while (saliencyDetail.firstChild) saliencyDetail.removeChild(saliencyDetail.firstChild);
    var entries = saliencyEntries(run);
    var runtime = run && run.saliency_runtime || {};
    saliencyRuntimeLabel.textContent = runtime.total_inference_ms === undefined ? "Unavailable" : exact(runtime.total_inference_ms) + " ms inference";
    var unavailable = (run && run.saliency || []).find(function (group) { return group.replay_available === false; });
    if (unavailable) {
      saliencyDetail.appendChild(element("p", "saliency-warning", unavailable.replay_error || "Saliency replay unavailable."));
    }
    if (!entries.length) {
      saliencyDetail.appendChild(element("p", "empty", "Unavailable: no saliency replay artifacts recorded."));
      return;
    }
    if (!entries.some(function (item) { return item.key === state.saliencyKey; })) state.saliencyKey = entries[0].key;
    entries.forEach(function (item, index) {
      var tab = element("button", "saliency-tab", item.entry.duration);
      tab.type = "button";
      tab.id = "saliency-tab-" + index;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", item.key === state.saliencyKey ? "true" : "false");
      tab.setAttribute("aria-controls", "saliency-detail");
      tab.setAttribute("tabindex", item.key === state.saliencyKey ? "0" : "-1");
      tab.dataset.saliencyKey = item.key;
      tab.addEventListener("click", function () { state.saliencyKey = item.key; renderSaliency(currentRun()); });
      tab.addEventListener("keydown", function (event) {
        var nextIndex = null;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % entries.length;
        else if (event.key === "ArrowLeft") nextIndex = (index - 1 + entries.length) % entries.length;
        else if (event.key === "Home") nextIndex = 0;
        else if (event.key === "End") nextIndex = entries.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        state.saliencyKey = entries[nextIndex].key;
        renderSaliency(currentRun());
        saliencyTabs.querySelectorAll('[role="tab"]')[nextIndex].focus();
      });
      saliencyTabs.appendChild(tab);
    });
    var selected = entries.find(function (item) { return item.key === state.saliencyKey; }) || entries[0];
    var selectedTab = saliencyTabs.querySelector('[aria-selected="true"]');
    if (selectedTab) saliencyDetail.setAttribute("aria-labelledby", selectedTab.id);
    var group = selected.group;
    var entry = selected.entry;
    addFields(saliencyDetail, [
      ["Captured page", "Recorded website view"],
      ["Duration", entry.duration],
      ["Search stage", group.search_stage],
      ["Provider", entry.provider_id || group.provider_id],
      ["Execution provider", entry.execution_provider || group.execution_provider],
      ["Cache status", group.cache_state],
      ["Inference timing", exact(entry.inference_duration_ms) + " ms"],
      ["Model", entry.model_id + " / " + entry.model_version]
    ]);
    if (group.overlay_message) saliencyDetail.appendChild(element("p", "saliency-warning", group.overlay_message));
    if (entry.heatmap) {
      var image = document.createElement("img");
      image.className = "saliency-heatmap";
      image.src = entry.heatmap;
      image.alt = "Heatmap-only saliency artifact for " + entry.duration;
      saliencyDetail.appendChild(image);
    } else {
      saliencyDetail.appendChild(element("p", "empty", "Unavailable: heatmap-only artifact was not recorded."));
    }
    appendRankedElements(saliencyDetail, entry.ranked_elements, run, group);
    appendAggregationComponents(saliencyDetail, entry.aggregation_components, run);
    addValues(saliencyDetail, "Selected mixture", (group.selected_mixture || []).map(function (item) { return item[0] + ": " + item[1]; }));
    addValues(saliencyDetail, "Stage history", (group.stage_history || []).map(function (item) {
      return item.event_id + ": " + item.kind + " | " + item.search_stage + " | cache " + item.cache_state;
    }));
    addValues(saliencyDetail, "Search-stage timeline", (run.saliency_stage_timeline || []).map(function (item) { return "Step " + item.sequence + ": " + item.kind + " | " + (item.search_stage || "unavailable"); }));
    addValues(saliencyDetail, "Fallback warnings", (run.saliency_fallbacks || []).map(function (item) { return item.provider_id + " -> " + item.fallback_provider_id + ": " + item.reason; }), "Unavailable: no fallback recorded.");
  }

  function selectElement(run, snapshot, event, elementId) {
    setPlaying(false);
    state.viewportId = snapshot.id;
    state.elementId = elementId;
    viewportStage.querySelectorAll(".viewport-overlay").forEach(function (overlay) {
      overlay.setAttribute("aria-pressed", overlay.dataset.elementId === elementId ? "true" : "false");
    });
    renderElementDetail(run, snapshot, event);
    updateUrl(run, event, null, false);
    var evidencePane = document.getElementById("selected-element-evidence");
    if (evidencePane && window.matchMedia && window.matchMedia("(max-width: 760px)").matches) {
      evidencePane.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  }

  function fitFrame(frame, snapshot) {
    var width = Math.max(viewportStage.clientWidth, 1);
    var height = Math.max(viewportStage.clientHeight, 1);
    var scale = Math.min(width / snapshot.viewport.width, height / snapshot.viewport.height);
    frame.style.width = Math.max(1, Math.floor(snapshot.viewport.width * scale)) + "px";
    frame.style.height = Math.max(1, Math.floor(snapshot.viewport.height * scale)) + "px";
  }

  function renderViewport(run, snapshot, event) {
    while (viewportStage.firstChild) viewportStage.removeChild(viewportStage.firstChild);
    if (!snapshot) {
      viewportMeta.textContent = "Unavailable";
      viewportStage.appendChild(element("p", "empty", "Unavailable: no viewport was recorded by this event."));
      renderElementDetail(run, null, event);
      return;
    }
    viewportMeta.textContent = snapshot.viewport.width + " x " + snapshot.viewport.height + " website view";
    var frame = element("div", "viewport-frame");
    fitFrame(frame, snapshot);
    if (snapshot.screenshot) {
      var image = document.createElement("img");
      image.src = snapshot.screenshot;
      image.alt = "Recorded website screenshot";
      frame.appendChild(image);
    } else {
      frame.appendChild(element("span", "empty", snapshot.screenshot_redacted ? "Overlay unavailable due redaction. Heatmap-only artifact retained." : "Unavailable: screenshot artifact was not recorded."));
    }
    var attention = noticedState(run, state.eventIndex);
    (snapshot.elements || []).filter(function (item) { return item.visibility_fraction > 0; }).forEach(function (item) {
      var overlayClass = "viewport-overlay";
      if (state.elementId === item.id) overlayClass += " selected-evidence";
      if (attention.inspected[item.id]) overlayClass += " inspected";
      else if (attention.noticed[item.id]) overlayClass += " noticed";
      var overlay = element("button", overlayClass);
      overlay.type = "button";
      overlay.dataset.elementId = item.id;
      overlay.dataset.viewportId = snapshot.id;
      overlay.setAttribute("aria-label", "Show evidence for " + item.label);
      overlay.setAttribute("aria-pressed", state.elementId === item.id ? "true" : "false");
      overlay.style.left = (item.bounds.x / snapshot.viewport.width * 100) + "%";
      overlay.style.top = (item.bounds.y / snapshot.viewport.height * 100) + "%";
      overlay.style.width = (item.bounds.width / snapshot.viewport.width * 100) + "%";
      overlay.style.height = (item.bounds.height / snapshot.viewport.height * 100) + "%";
      overlay.appendChild(element("span", "overlay-label", item.label));
      var select = function () { selectElement(run, snapshot, event, item.id); };
      overlay.addEventListener("mouseenter", select);
      overlay.addEventListener("focus", select);
      overlay.addEventListener("click", select);
      frame.appendChild(overlay);
    });
    viewportStage.appendChild(frame);
    renderElementDetail(run, snapshot, event);
  }

  function promptVersion(run, role) {
    var versions = run.manifests && run.manifests.prompt_versions || {};
    if (versions[role]) return versions[role];
    var manifest = run.manifests && (run.manifests.provider_manifests || []).find(function (item) { return item.role === role; });
    return manifest ? manifest.prompt_version : null;
  }

  function renderModelCall(parent, run, record) {
    var usage = record.token_usage || {};
    addFields(parent, [
      ["Model role", record.role],
      ["Model", record.model],
      ["Endpoint origin", record.endpoint_origin],
      ["Latency ms", record.latency_ms],
      ["Attempts", record.attempts],
      ["Schema version", record.schema_version],
      ["Prompt version", promptVersion(run, record.role)],
      ["Prompt digest", record.prompt_digest],
      ["Prompt tokens", usage.prompt_tokens],
      ["Completion tokens", usage.completion_tokens],
      ["Total tokens", usage.total_tokens]
    ]);
    addJson(parent, "Sanitized request summary", record.request);
    addJson(parent, "Sanitized response summary", record.response);
    addValues(parent, "Retries", (record.retries || []).map(function (retry) { return "Attempt " + retry.attempt + ": " + retry.reason; }));
  }

  function renderEvent(run, record) {
    while (eventCard.firstChild) eventCard.removeChild(eventCard.firstChild);
    if (!record) {
      eventKind.textContent = "Unavailable";
      eventCard.appendChild(element("p", "empty", "Unavailable: no timeline events recorded."));
      return;
    }
    eventKind.textContent = eventCategory(record.kind);
    eventCard.appendChild(element("span", "event-category", eventCategory(record.kind)));
    eventCard.appendChild(element("strong", "event-title", "Step " + record.sequence + " | " + eventTitle(record.kind)));
    addFields(eventCard, [["Recorded time", recordedTime(record)]]);

    if (record.kind === "viewport-captured") {
      var snapshot = findSnapshot(run, record.viewport_id);
      addFields(eventCard, [["Visible elements", snapshot ? snapshot.elements.length : null], ["Website sections", snapshot ? snapshot.regions.length : null]]);
    } else if (record.kind === "observation-recorded" && record.observation) {
      addFields(eventCard, [["Website section", record.observation.region_context && record.observation.region_context.label]]);
      addValues(eventCard, "Noticed now", (record.observation.newly_revealed_elements || []).map(function (item) { return item.label; }));
      addValues(eventCard, "Remembered", (record.observation.remembered_elements || []).map(function (item) { return item.label; }));
    } else if (record.kind.indexOf("prominence") !== -1 || record.kind.indexOf("scent") !== -1) {
      addValues(eventCard, "Element scores", (record.scores || []).map(function (score) { return elementLabel(run, score.element_id) + ": " + exact(score.raw_score !== undefined ? score.raw_score : score.score); }));
    } else if (record.kind === "attention-selection-recorded") {
      addFields(eventCard, [["Selection mode", record.selection_mode]]);
      addValues(eventCard, "Selected elements", (record.selected_ids || []).map(function (id) { return elementLabel(run, id); }));
      addValues(eventCard, "Recovery choices", (record.recovery_selected_ids || []).map(function (id) { return elementLabel(run, id); }));
      addMapping(eventCard, "Element probabilities", record.element_probabilities, run);
      addMapping(eventCard, "Region probabilities", record.region_probabilities, run);
    } else if (record.kind === "model-call-recorded" && record.record) {
      renderModelCall(eventCard, run, record.record);
    } else if (record.kind === "model-failure") {
      addFields(eventCard, [["Model role", record.role], ["Failure", record.reason]]);
      addJson(eventCard, "Sanitized response summary", record.response_summary);
    } else if (record.kind === "repeated-fixture-input" || record.kind === "fixture-input-completed") {
      addFields(eventCard, [["Element", elementLabel(run, record.element_id)], ["Reason", record.reason]]);
    } else if (record.kind === "repeated-action-detected" || record.kind === "repeated-action-cycle" || record.kind === "no-progress-recovery" || record.kind === "no-progress-detected") {
      addFields(eventCard, [["Action", record.action ? actionText(record.action, run) : null], ["Count", record.count], ["Cycle length", record.cycle_length], ["Reason", record.reason]]);
    } else if (record.kind === "model-call-budget-exhausted") {
      addFields(eventCard, [["Model calls", record.model_calls], ["Limit", record.limit], ["Reason", record.reason]]);
    } else if (record.kind === "verification-recorded" && record.verification) {
      addFields(eventCard, [["Verified", record.verification.verified], ["Details", record.verification.details]]);
    } else if (record.kind === "run-terminated") {
      addFields(eventCard, [["Terminal outcome", record.outcome], ["Valid UX sample", run.ux_sample_valid], ["Invalid sample reason", run.ux_sample_invalid_reason], ["Stage", run.stage], ["Terminal reason", run.terminal_reason], ["Evaluation failure", run.evaluation_failure_reason]]);
    } else {
      addFields(eventCard, [
        ["Action", record.action ? actionText(record.action, run) : null],
        ["Reason", record.reason],
        ["Result", record.succeeded === undefined ? null : record.succeeded ? "succeeded" : "failed"],
        ["Failure", record.error || record.message],
        ["Claimed success", record.claimed_success]
      ]);
    }
    addValues(eventCard, "Memory / state", (run.memory || []).map(function (item) { return memoryLabel(item, run) + " | strength " + item.strength; }), "Unavailable: no public memory state recorded.");
  }

  function renderTimeline(run) {
    while (timelineList.firstChild) timelineList.removeChild(timelineList.firstChild);
    var events = run ? run.timeline || [] : [];
    timelineCount.textContent = events.length + (events.length === 1 ? " event" : " events");
    events.forEach(function (record, index) {
      var item = document.createElement("li");
      var button = element("button", "timeline-event");
      button.type = "button";
      button.dataset.eventKind = record.kind;
      button.dataset.eventId = record.event_id || "";
      button.setAttribute("aria-current", index === state.eventIndex ? "true" : "false");
      button.appendChild(element("span", "timeline-step", "Step " + record.sequence));
      button.appendChild(element("span", "timeline-kind", eventTitle(record.kind)));
      button.appendChild(element("span", "timeline-summary", eventSummary(record, run)));
      button.addEventListener("click", function () { setEventIndex(index, false); });
      item.appendChild(button);
      timelineList.appendChild(item);
    });
    var current = timelineList.querySelector('[aria-current="true"]');
    if (current) current.scrollIntoView({ block: "nearest", inline: "center" });
  }

  function renderStatus(run) {
    statusBanner.className = "run-status-banner " + (run ? run.status_class : "status-untrusted");
    if (!run) {
      statusBanner.textContent = "Run unavailable: no run matches current selector.";
      return;
    }
    var parts = [runLabel(run), "Outcome: " + titleCase(run.outcome)];
    if (run.stage) parts.push("Stage: " + titleCase(run.stage));
    parts.push(run.verified ? "Result verified" : "Result not verified");
    if (run.user_effort) parts.push("Estimated task time " + Number(run.user_effort.estimated_task_seconds || 0).toFixed(1) + " seconds");
    if (run.evaluation_failure_reason) {
      parts.push(run.evaluation_failure_reason.indexOf("saliency-replay-unavailable") !== -1
        ? "Attention replay unavailable for this run"
        : "Evaluation check unavailable for this run");
    } else if (run.terminal_reason) {
      parts.push("Completion note: " + titleCase(run.terminal_reason));
    }
    else if (run.failure_reason) parts.push("Why: " + run.failure_reason);
    statusBanner.textContent = parts.join(" | ");
  }

  function renderRows(run) {
    document.querySelectorAll(".run-row[data-run-id]").forEach(function (row) {
      row.setAttribute("aria-current", run && row.dataset.runId === run.run_id ? "true" : "false");
    });
  }

  function renderWorkspace() {
    var run = currentRun();
    var events = run ? run.timeline || [] : [];
    state.eventIndex = Math.min(state.eventIndex, Math.max(events.length - 1, 0));
    var record = currentEvent(run);
    var snapshot = snapshotAt(run, state.eventIndex);
    state.viewportId = snapshot ? snapshot.id : "";
    renderStatus(run);
    renderRows(run);
    renderTimeline(run);
    renderEvent(run, record);
    renderViewport(run, snapshot, record);
    renderSaliency(run);
    progress.max = String(Math.max(events.length - 1, 0));
    progress.value = String(state.eventIndex);
    progress.disabled = !events.length;
    position.textContent = events.length ? "Event " + (state.eventIndex + 1) + " / " + events.length + " | " + recordedTime(record) : "Event 0 / 0 | time unavailable";
    runSelect.value = state.runId;
    updateUrl(run, record, null, false);
  }

  function applyUrlState() {
    var params = new URL(window.location.href).searchParams;
    var runId = params.get("run");
    var eventId = params.get("event");
    var elementId = params.get("element");
    var evidenceId = params.get("evidence");
    state.eventIndex = 0;
    state.elementId = "";
    if (runId && runs.some(function (run) { return run.run_id === runId; })) {
      state.runId = runId;
      var selectedRun = currentRun();
      state.scenarioId = selectedRun ? selectedRun.scenario_id : "";
    }
    var run = currentRun();
    if (eventId && run) {
      var index = (run.timeline || []).findIndex(function (record) { return record.event_id === eventId; });
      if (index >= 0) state.eventIndex = index;
    }
    if (elementId) state.elementId = elementId;
    state.evidenceId = evidenceId || "";
  }

  function restoreUrlState() {
    setPlaying(false);
    applyUrlState();
    renderRunOptions();
    renderWorkspace();
    if (window.location.hash.indexOf("#view-") === 0) {
      clearEvidenceDestination();
      return;
    }
    if (state.evidenceId) openEvidence(state.evidenceId, { pushHistory: false });
    else clearEvidenceDestination();
  }

  fillScenarioOptions();
  fillProviderComparisonOptions();
  applyUrlState();
  renderRunOptions();
  renderWorkspace();
  window.openEvidence = openEvidence;
  var initialEvidenceId = new URL(window.location.href).searchParams.get("evidence");
  if (initialEvidenceId) {
    window.setTimeout(function () {
      openEvidence(initialEvidenceId, { pushHistory: false });
    }, 0);
  }

  scenarioSelect.addEventListener("change", function () {
    state.scenarioId = scenarioSelect.value;
    var visible = visibleRuns();
    state.runId = visible.length ? visible[0].run_id : "";
    state.eventIndex = 0;
    state.viewportId = "";
    state.elementId = "";
    renderRunOptions();
    renderWorkspace();
  });
  if (comparisonSelect) {
    comparisonSelect.addEventListener("change", function () {
      state.comparisonKey = comparisonSelect.value;
      fillComparisonDurationOptions(selectedProviderComparison());
      renderProviderComparison();
    });
  }
  if (comparisonDurationSelect) {
    comparisonDurationSelect.addEventListener("change", function () {
      state.comparisonDuration = comparisonDurationSelect.value;
      renderProviderComparison();
    });
  }
  runSelect.addEventListener("change", function () { selectRun(runSelect.value, false); });
  playPause.addEventListener("click", function () {
    var run = currentRun();
    if (run && run.run_page && !(run.timeline || []).length) {
      selectRun(run.run_id, true);
      return;
    }
    setPlaying(!state.playing);
  });
  document.getElementById("step-back").addEventListener("click", function () { setEventIndex(state.eventIndex - 1, false); });
  document.getElementById("step-forward").addEventListener("click", function () { setEventIndex(state.eventIndex + 1, false); });
  document.getElementById("restart-playback").addEventListener("click", function () { setEventIndex(0, false); });
  progress.addEventListener("input", function () { setEventIndex(Number(progress.value), false); });
  document.querySelectorAll("[data-open-run]").forEach(function (control) {
    control.addEventListener("click", function (event) {
      var run = runs.find(function (item) { return item.run_id === control.dataset.openRun; });
      if (!run || (run.run_page && !(run.timeline || []).length)) return;
      event.preventDefault();
      var previousUrl = window.location.href;
      selectRun(control.dataset.openRun, false);
      window.history.replaceState({ reportState: true }, "", previousUrl);
      updateUrl(currentRun(), currentEvent(currentRun()), "#playback-workspace", true);
      showViewForNode(document.getElementById("playback-workspace"));
      document.getElementById("playback-workspace").scrollIntoView({ block: "start" });
    });
  });
  document.querySelectorAll(".evidence-ref").forEach(function (control) {
    control.addEventListener("click", function () { openEvidence(control.dataset.evidenceId); });
  });
  window.addEventListener("popstate", restoreUrlState);
  window.addEventListener("resize", function () {
    var run = currentRun();
    renderViewport(run, snapshotAt(run, state.eventIndex), currentEvent(run));
  });

  function viewSectionOf(node) {
    while (node) {
      if (node.classList && node.classList.contains("view")) return node;
      node = node.parentElement;
    }
    return null;
  }
  function showViewForNode(node) {
    var view = viewSectionOf(node);
    if (!view) return false;
    if (!view.hidden) return true;
    document.querySelectorAll(".view").forEach(function (item) { item.hidden = item !== view; });
    syncViewRail(view.id);
    return true;
  }
  function syncViewRail(activeId) {
    document.querySelectorAll(".view-rail a").forEach(function (tab) {
      if (tab.getAttribute("href") === "#" + activeId) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    });
  }
  document.querySelectorAll(".view-rail a").forEach(function (tab) {
    tab.addEventListener("click", function () {
      var target = document.getElementById(decodeURIComponent(tab.getAttribute("href").slice(1)));
      if (target) showViewForNode(target);
    });
  });
  function routeView(scrollTarget) {
    var hash = window.location.hash;
    if (hash.indexOf("#view-") === 0) {
      var view = document.getElementById(hash.slice(1));
      if (view) {
        showViewForNode(view);
        if (scrollTarget) window.scrollTo({ top: 0, behavior: "auto" });
        return;
      }
    }
    var params = new URLSearchParams(window.location.search);
    if (params.get("run")) {
      var workspace = document.getElementById("playback-workspace");
      showViewForNode(workspace);
      return;
    }
    if (hash.length > 1) {
      var el = document.getElementById(decodeURIComponent(hash.slice(1)));
      if (el) {
        showViewForNode(el);
        if (scrollTarget) el.scrollIntoView({ block: "start", behavior: "auto" });
      }
    }
  }
  focusEvidenceDestination = (function (original) {
    return function (target, evidenceId, options) {
      var destination = null;
      if (target.kind === "metric") {
        destination = metricDestination(target.run_id, target.metric_id);
        if (!destination) {
          destination = document.querySelector('tr[data-run-id="' + CSS.escape(String(target.run_id || "")) + '"]');
        }
      } else if (target.kind === "evidence-detail") destination = evidenceDetail;
      else if (target.kind === "event" || target.kind === "replay") destination = eventCard;
      else if (target.kind === "viewport" || target.kind === "screenshot") destination = viewportStage;
      else if (target.duration) destination = saliencyTabs;
      else if (target.element_id) destination = elementPanel;
      else destination = eventCard;
      if (destination && !(options && options.pushHistory === false)) showViewForNode(destination);
      return original(target, evidenceId, options);
    };
  })(focusEvidenceDestination);
  window.addEventListener("hashchange", function () { routeView(true); });
  routeView(true);
}());
