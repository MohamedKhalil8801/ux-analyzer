(function () {
  "use strict";

  var dataNode = document.getElementById("report-data");
  if (!dataNode) return;
  var data = JSON.parse(dataNode.textContent || "{}");
  var runs = data.runs || [];
  var comparisons = data.provider_comparisons || [];
  var scenarioSelect = document.getElementById("scenario-select");
  var runSelect = document.getElementById("run-select");
  var playButton = document.getElementById("play-pause");
  var statusBanner = document.getElementById("run-status-banner");
  var comparisonSelect = document.getElementById("provider-comparison-select");
  var durationSelect = document.getElementById("provider-comparison-duration");
  var comparisonOutput = document.getElementById("provider-comparison-output");

  function node(tag, className, text) {
    var value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = String(text);
    return value;
  }

  function runLabel(run) {
    return run.scenario_label + " | " + run.version_label + " | " + run.persona_label;
  }

  function selectedRun() {
    return runs.find(function (run) { return run.run_id === runSelect.value; }) || null;
  }

  function runUrl(run, extra) {
    if (!run || !run.run_page) return "";
    var params = new URLSearchParams(extra || {});
    params.set("run", run.run_id);
    return run.run_page + "?" + params.toString() + "#playback-workspace";
  }

  function openRun(run, extra) {
    var url = runUrl(run, extra);
    if (url) window.location.href = url;
  }

  function visibleRuns() {
    return runs.filter(function (run) {
      return !scenarioSelect.value || run.scenario_id === scenarioSelect.value;
    });
  }

  function renderRunOptions() {
    var current = runSelect.value;
    while (runSelect.firstChild) runSelect.removeChild(runSelect.firstChild);
    visibleRuns().forEach(function (run) {
      var option = node("option", "", runLabel(run) + " | " + run.outcome);
      option.value = run.run_id;
      runSelect.appendChild(option);
    });
    if (Array.from(runSelect.options).some(function (option) { return option.value === current; })) {
      runSelect.value = current;
    }
    var run = selectedRun();
    statusBanner.className = "run-status-banner " + (run ? run.status_class : "status-untrusted");
    statusBanner.textContent = run
      ? runLabel(run) + " | Outcome: " + String(run.outcome || "unavailable").replace(/-/g, " ") + " | Open the replay to inspect the recorded timeline."
      : "No replay matches the selected scenario.";
  }

  function fillRunControls() {
    var scenarios = [];
    runs.forEach(function (run) {
      if (run.scenario_id && scenarios.indexOf(run.scenario_id) === -1) scenarios.push(run.scenario_id);
    });
    scenarios.forEach(function (scenarioId) {
      var run = runs.find(function (item) { return item.scenario_id === scenarioId; });
      var option = node("option", "", run ? run.scenario_label : scenarioId);
      option.value = scenarioId;
      scenarioSelect.appendChild(option);
    });
    renderRunOptions();
    scenarioSelect.addEventListener("change", renderRunOptions);
    runSelect.addEventListener("change", function () { openRun(selectedRun()); });
    playButton.addEventListener("click", function () { openRun(selectedRun()); });
  }

  function openEvidence(control) {
    var target;
    try { target = JSON.parse(control.dataset.evidenceTarget || "{}"); }
    catch (_error) { return; }
    if (!target.run_page) return;
    var extra = {};
    if (target.event_id) extra.event = target.event_id;
    else if (target.sequence !== undefined) extra.event = "event-" + target.sequence;
    if (target.element_id) extra.element = target.element_id;
    if (control.dataset.evidenceId) extra.evidence = control.dataset.evidenceId;
    window.location.href = target.run_page + "?" + new URLSearchParams(Object.assign({ run: target.run_id }, extra)).toString() + "#playback-workspace";
  }

  function comparison() {
    return comparisons.find(function (item) { return item.key === comparisonSelect.value; }) || null;
  }

  function renderDurations() {
    if (!durationSelect) return;
    var values = [];
    var selected = comparison();
    (selected ? selected.providers || [] : []).forEach(function (provider) {
      (provider.heatmaps || []).forEach(function (heatmap) {
        if (values.indexOf(heatmap.duration) === -1) values.push(heatmap.duration);
      });
    });
    while (durationSelect.firstChild) durationSelect.removeChild(durationSelect.firstChild);
    values.forEach(function (duration) {
      var option = node("option", "", duration);
      option.value = duration;
      durationSelect.appendChild(option);
    });
    durationSelect.disabled = !values.length;
  }

  function renderComparison() {
    if (!comparisonOutput) return;
    while (comparisonOutput.firstChild) comparisonOutput.removeChild(comparisonOutput.firstChild);
    var selected = comparison();
    if (!selected) {
      comparisonOutput.appendChild(node("p", "empty", "No paired methods were recorded."));
      return;
    }
    comparisonOutput.appendChild(node("h3", "", selected.scenario_label + " | " + selected.version_label));
    var grid = node("div", "provider-comparison-grid");
    (selected.providers || []).forEach(function (provider) {
      var card = node("article", "provider-card");
      var heading = node("div", "provider-card-heading");
      heading.appendChild(node("h3", "", String(provider.provider_id || "method").replace(/-/g, " ")));
      heading.appendChild(node("span", "provider-status", provider.outcome));
      card.appendChild(heading);
      var heatmap = (provider.heatmaps || []).find(function (item) {
        return item.duration === durationSelect.value;
      });
      if (heatmap && heatmap.heatmap) {
        var image = node("img", "provider-heatmap exact-heatmap");
        image.src = heatmap.heatmap;
        image.alt = "Recorded attention heatmap";
        card.appendChild(image);
      }
      var path = node("ol", "comparison-path");
      (provider.action_path || []).forEach(function (step) {
        var item = node("li", "");
        var button = node("button", "path-step", "Step " + step.sequence + " | " + step.element_label);
        button.type = "button";
        button.addEventListener("click", function () {
          var run = runs.find(function (candidate) { return candidate.run_id === provider.run_id; });
          openRun(run, { event: "event-" + step.sequence });
        });
        item.appendChild(button);
        path.appendChild(item);
      });
      card.appendChild(path);
      grid.appendChild(card);
    });
    comparisonOutput.appendChild(grid);
  }

  function fillComparisons() {
    if (!comparisonSelect) return;
    comparisons.forEach(function (item) {
      var option = node("option", "", item.scenario_label + " | " + item.version_label);
      option.value = item.key;
      comparisonSelect.appendChild(option);
    });
    renderDurations();
    renderComparison();
    comparisonSelect.addEventListener("change", function () {
      renderDurations();
      renderComparison();
    });
    durationSelect.addEventListener("change", renderComparison);
  }

  fillRunControls();
  fillComparisons();
  document.querySelectorAll(".evidence-ref").forEach(function (control) {
    control.addEventListener("click", function () { openEvidence(control); });
  });
}());
