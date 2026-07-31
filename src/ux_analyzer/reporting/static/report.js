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

  function addMapping(parent, label, mapping) {
    var entries = Object.keys(mapping || {}).sort();
    var block = element("div", "value-block");
    block.appendChild(element("strong", "", label));
    if (!entries.length) {
      block.appendChild(element("p", "empty", "Unavailable: no recorded values."));
    } else {
      var list = element("dl", "field-list");
      entries.forEach(function (key) { addField(list, titleCase(key), mapping[key]); });
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

  function actionText(action) {
    if (!action) return "Unavailable: no recorded action.";
    var text = titleCase(action.kind || "action");
    if (action.element_id) text += " on " + action.element_id;
    if (action.direction) text += " " + action.direction;
    if (action.duration_seconds !== undefined) text += " for " + action.duration_seconds + " seconds";
    return text;
  }

  function eventCategory(kind) {
    if (kind === "viewport-captured" || kind === "observation-recorded") return "Observation";
    if (kind.indexOf("prominence") !== -1) return "Prominence";
    if (kind.indexOf("scent") !== -1) return "Scent";
    if (kind === "attention-selection-recorded") return "Attention selection";
    if (kind === "model-call-recorded") return "Model request / response";
    if (kind === "decision-recorded" || kind === "action-proposed" || kind === "agent-claim") return "Model decision and reason";
    if (kind === "action-executed" || kind === "action-rejected") return "Action and result";
    if (kind === "verification-recorded") return "Verification";
    if (kind.indexOf("memory") !== -1 || kind.indexOf("state") !== -1) return "Memory / state";
    if (kind === "run-terminated" || kind.indexOf("failure") !== -1) return "Terminal / failure";
    return "Recorded event";
  }

  function eventSummary(record) {
    if (!record) return "Unavailable";
    if (record.reason) return String(record.reason);
    if (record.action) return actionText(record.action);
    if (record.outcome) return "Outcome: " + record.outcome;
    if (record.viewport_id) return "Viewport: " + record.viewport_id;
    if (record.selected_ids && record.selected_ids.length) return "Selected: " + record.selected_ids.join(", ");
    return eventCategory(record.kind);
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
      option.textContent = run.run_id + " | " + run.version_label + " | " + run.outcome;
      runSelect.appendChild(option);
    });
    if (!visible.some(function (run) { return run.run_id === state.runId; })) {
      state.runId = visible.length ? visible[0].run_id : "";
      state.eventIndex = 0;
      state.viewportId = "";
      state.elementId = "";
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

  function updateHash(run, event) {
    if (!run) return;
    var params = new URLSearchParams();
    params.set("run", run.run_id);
    if (event && event.event_id) params.set("event", event.event_id);
    if (state.elementId) params.set("element", state.elementId);
    window.history.replaceState(null, "", "#" + params.toString());
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
      window.location.href = run.run_page + "#run=" + encodeURIComponent(run.run_id);
      return;
    }
    setPlaying(false);
    state.runId = run.run_id;
    state.scenarioId = run.scenario_id;
    state.eventIndex = 0;
    state.viewportId = "";
    state.elementId = "";
    renderRunOptions();
    renderWorkspace();
    if (scrollWorkspace) {
      document.getElementById("playback-workspace").scrollIntoView({ block: "start" });
    }
  }

  function findSnapshot(run, viewportId) {
    return run && (run.snapshots || []).find(function (snapshot) { return snapshot.id === viewportId; }) || null;
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
      ["Element ID", selected.id],
      ["Role", selected.role],
      ["Viewport", snapshot.id],
      ["Region", region ? region.label + " [" + region.id + "]" : selected.region_id],
      ["Actionable", selected.actionable],
      ["Disabled", selected.disabled],
      ["Bounds", "x=" + selected.bounds.x + ", y=" + selected.bounds.y + ", width=" + selected.bounds.width + ", height=" + selected.bounds.height],
      ["Visibility fraction", selected.visibility_fraction],
      ["Occlusion fraction", selected.occlusion_fraction],
      ["Local contrast", selected.local_contrast],
      ["Noticed by this step", Boolean(attention.noticed[selected.id])],
      ["Inspected by this step", Boolean(attention.inspected[selected.id])]
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
    } else {
      elementDetail.appendChild(element("p", "empty", "Unavailable: no prominence record for this element at current step."));
    }
    addValues(elementDetail, "Scent", scentAt(run, snapshot.id, selected.id, event ? event.sequence : 0), "Unavailable: no scent record at current step.");
    var linked = linkedRecords(run, selected.id);
    addValues(elementDetail, "Linked decisions", linked.decisions.map(function (record) { return "Step " + record.sequence + ": " + (record.reason || actionText(record.action)); }));
    addValues(elementDetail, "Linked actions and results", linked.actions.map(function (record) { return "Step " + record.sequence + ": " + actionText(record.action) + " | " + (record.succeeded ? "succeeded" : "failed") + (record.error ? " | " + record.error : ""); }));
    addValues(elementDetail, "Linked findings", linked.findings.map(function (record) { return record.title + ": " + record.cause; }));
    addValues(elementDetail, "Linked evidence", linked.evidence.map(function (record) { return record.evidence_id + ": " + record.description; }));
  }

  function selectElement(run, snapshot, event, elementId) {
    setPlaying(false);
    state.viewportId = snapshot.id;
    state.elementId = elementId;
    viewportStage.querySelectorAll(".viewport-overlay").forEach(function (overlay) {
      overlay.setAttribute("aria-pressed", overlay.dataset.elementId === elementId ? "true" : "false");
    });
    renderElementDetail(run, snapshot, event);
    updateHash(run, event);
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
    viewportMeta.textContent = snapshot.id + " | " + snapshot.viewport.width + " x " + snapshot.viewport.height;
    var frame = element("div", "viewport-frame");
    fitFrame(frame, snapshot);
    if (snapshot.screenshot) {
      var image = document.createElement("img");
      image.src = snapshot.screenshot;
      image.alt = "Recorded screenshot for " + snapshot.id;
      frame.appendChild(image);
    } else {
      frame.appendChild(element("span", "empty", "Unavailable: screenshot artifact was not recorded."));
    }
    var attention = noticedState(run, state.eventIndex);
    (snapshot.elements || []).filter(function (item) { return item.visibility_fraction > 0; }).forEach(function (item) {
      var overlayClass = "viewport-overlay";
      if (attention.inspected[item.id]) overlayClass += " inspected";
      else if (attention.noticed[item.id]) overlayClass += " noticed";
      var overlay = element("button", overlayClass);
      overlay.type = "button";
      overlay.dataset.elementId = item.id;
      overlay.dataset.viewportId = snapshot.id;
      overlay.setAttribute("aria-label", "Inspect recorded evidence for " + item.label);
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
    eventKind.textContent = titleCase(record.kind);
    eventCard.appendChild(element("span", "event-category", eventCategory(record.kind)));
    eventCard.appendChild(element("strong", "event-title", "Step " + record.sequence + " | " + titleCase(record.kind)));
    addFields(eventCard, [["Event ID", record.event_id], ["Recorded time", recordedTime(record)]]);

    if (record.kind === "viewport-captured") {
      var snapshot = findSnapshot(run, record.viewport_id);
      addFields(eventCard, [["Viewport", record.viewport_id], ["Elements", snapshot ? snapshot.elements.length : null], ["Regions", snapshot ? snapshot.regions.length : null]]);
    } else if (record.kind === "observation-recorded" && record.observation) {
      addFields(eventCard, [["Viewport", record.observation.viewport_id], ["Region", record.observation.region_context && record.observation.region_context.label]]);
      addValues(eventCard, "Noticed now", (record.observation.newly_revealed_elements || []).map(function (item) { return item.label + " [" + item.id + "]"; }));
      addValues(eventCard, "Remembered", (record.observation.remembered_elements || []).map(function (item) { return item.label + " [" + item.id + "]"; }));
    } else if (record.kind.indexOf("prominence") !== -1 || record.kind.indexOf("scent") !== -1) {
      addValues(eventCard, "Element scores", (record.scores || []).map(function (score) { return score.element_id + ": " + exact(score.raw_score !== undefined ? score.raw_score : score.score); }));
    } else if (record.kind === "attention-selection-recorded") {
      addFields(eventCard, [["Viewport", record.viewport_id], ["Mode", record.selection_mode], ["Region", record.region_id]]);
      addValues(eventCard, "Selected elements", record.selected_ids || []);
      addMapping(eventCard, "Element probabilities", record.element_probabilities);
      addMapping(eventCard, "Region probabilities", record.region_probabilities);
    } else if (record.kind === "model-call-recorded" && record.record) {
      renderModelCall(eventCard, run, record.record);
    } else if (record.kind === "verification-recorded" && record.verification) {
      addFields(eventCard, [["Verified", record.verification.verified], ["Details", record.verification.details]]);
      addValues(eventCard, "Evidence IDs", record.verification.evidence_ids || []);
    } else if (record.kind === "run-terminated") {
      addFields(eventCard, [["Terminal outcome", record.outcome], ["Stage", run.stage], ["Terminal reason", run.terminal_reason], ["Evaluation failure", run.evaluation_failure_reason]]);
    } else {
      addFields(eventCard, [
        ["Action", record.action ? actionText(record.action) : null],
        ["Reason", record.reason],
        ["Result", record.succeeded === undefined ? null : record.succeeded ? "succeeded" : "failed"],
        ["Failure", record.error || record.message],
        ["Claimed success", record.claimed_success]
      ]);
    }
    addValues(eventCard, "Memory / state", (run.memory || []).map(function (item) { return item.key + ": " + item.value + " | strength " + item.strength; }), "Unavailable: no public memory state recorded.");
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
      button.appendChild(element("span", "timeline-kind", titleCase(record.kind)));
      button.appendChild(element("span", "timeline-summary", eventSummary(record)));
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
    var parts = [
      run.run_id,
      "outcome " + run.outcome,
      "stage " + run.stage,
      run.verified ? "verified" : "not verified",
      "terminal state " + run.terminal_state
    ];
    if (run.terminal_reason) parts.push("terminal reason: " + run.terminal_reason);
    if (run.evaluation_failure_reason) parts.push("evaluation failure: " + run.evaluation_failure_reason);
    if (!run.trusted) parts.push("evidence untrusted");
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
    progress.max = String(Math.max(events.length - 1, 0));
    progress.value = String(state.eventIndex);
    progress.disabled = !events.length;
    position.textContent = events.length ? "Step " + (state.eventIndex + 1) + " / " + events.length + " | " + recordedTime(record) : "Step 0 / 0 | time unavailable";
    runSelect.value = state.runId;
    updateHash(run, record);
  }

  function applyHash() {
    var params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    var runId = params.get("run");
    var eventId = params.get("event");
    var elementId = params.get("element");
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
  }

  fillScenarioOptions();
  applyHash();
  renderRunOptions();
  renderWorkspace();

  scenarioSelect.addEventListener("change", function () {
    state.scenarioId = scenarioSelect.value;
    var visible = visibleRuns();
    selectRun(visible.length ? visible[0].run_id : "", false);
  });
  runSelect.addEventListener("change", function () { selectRun(runSelect.value, false); });
  playPause.addEventListener("click", function () { setPlaying(!state.playing); });
  document.getElementById("step-back").addEventListener("click", function () { setEventIndex(state.eventIndex - 1, false); });
  document.getElementById("step-forward").addEventListener("click", function () { setEventIndex(state.eventIndex + 1, false); });
  document.getElementById("restart-playback").addEventListener("click", function () { setEventIndex(0, false); });
  progress.addEventListener("input", function () { setEventIndex(Number(progress.value), false); });
  document.querySelectorAll(".run-row[data-run-id]").forEach(function (row) {
    var open = function () { selectRun(row.dataset.runId, true); };
    row.addEventListener("click", open);
    row.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
    });
  });
  window.addEventListener("hashchange", function () {
    setPlaying(false);
    applyHash();
    renderRunOptions();
    renderWorkspace();
  });
  window.addEventListener("resize", function () {
    var run = currentRun();
    renderViewport(run, snapshotAt(run, state.eventIndex), currentEvent(run));
  });
}());
