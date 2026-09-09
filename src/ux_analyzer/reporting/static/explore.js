(function () {
  "use strict";
  var apiBase = "/__explore/api";
  var STORAGE_KEY = "uxa-explore-curation";
  var state = {
    suggestions: [],
    corpus_summary: null,
    personas: { existing: [], suggested: [] },
    accepted: {},
    edits: {},
    added: [],
    personaSelection: { mode: "existing" },
    autoAccept: false,
    suggestionsSignature: "",
    loadFailed: false
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  // Persist volatile curation state so refresh does not destroy edits.
  // The stored signature pins the blob to one suggestion set; restoreCuration
  // refuses to re-apply blobs signed for any other set.
  function persistCuration() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
        signature: state.suggestionsSignature,
        accepted: state.accepted,
        edits: state.edits,
        added: state.added,
        personaSelection: state.personaSelection
      }));
    } catch (ignored) { /* storage unavailable */ }
  }

  // Re-apply persisted curation only where it still matches the freshly
  // fetched suggestion set: acceptance/edits survive only for ids the server
  // still offers, added scenarios only while their start_url is known to the
  // corpus. A blob signed for a different suggestion set is ignored
  // wholesale. Anything stale is dropped and announced, so old curation can
  // never resurface and fail server-side validation mid-flow.
  function restoreCuration() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var saved = JSON.parse(raw);
      if (!saved || typeof saved !== "object") return;
      if (saved.signature !== state.suggestionsSignature) {
        clearPersistedCuration();
        announceStatus("Saved curation ignored: the suggestion set changed since it was saved.");
        return;
      }
      var suggestionIds = {};
      state.suggestions.forEach(function (s) { suggestionIds[s.id] = true; });
      var knownUrls = {};
      ((state.corpus_summary && state.corpus_summary.urls) || []).forEach(function (u) {
        if (u) knownUrls[u] = true;
      });
      var dropped = 0;
      var keptAdded = [];
      if (Array.isArray(saved.added)) {
        saved.added.forEach(function (item) {
          if (item && item.id && knownUrls[item.start_url]) keptAdded.push(item);
          else dropped += 1;
        });
      }
      var keptAddedIds = {};
      keptAdded.forEach(function (item) { keptAddedIds[item.id] = true; });
      state.added = keptAdded;
      state.edits = {};
      if (saved.edits && typeof saved.edits === "object" && !Array.isArray(saved.edits)) {
        Object.keys(saved.edits).forEach(function (id) {
          if (suggestionIds[id]) state.edits[id] = saved.edits[id];
          else dropped += 1;
        });
      }
      // state.accepted was pre-seeded with all fetched suggestion ids; only
      // overlay saved values whose ids are still real, so phantom keys can
      // never enter the map.
      if (saved.accepted && typeof saved.accepted === "object" && !Array.isArray(saved.accepted)) {
        Object.keys(saved.accepted).forEach(function (id) {
          if (suggestionIds[id] || keptAddedIds[id]) state.accepted[id] = !!saved.accepted[id];
          else dropped += 1;
        });
      }
      if (saved.personaSelection && typeof saved.personaSelection === "object") {
        state.personaSelection = saved.personaSelection;
        if (!state.personaSelection.mode) {
          state.personaSelection.mode = "existing";
        }
        if (
          state.personaSelection.mode !== "custom" &&
          state.personaSelection.custom_persona
        ) {
          // Stale blobs from earlier sessions can carry custom_persona outside
          // custom mode; the server rejects that shape, so drop it here too.
          delete state.personaSelection.custom_persona;
        }
      }
      if (dropped > 0) {
        announceStatus(dropped + (dropped === 1 ? " stale entry discarded" : " stale entries discarded"));
      }
    } catch (ignored) { /* corrupt or unavailable storage: start fresh */ }
  }

  function clearPersistedCuration() {
    try { window.localStorage.removeItem(STORAGE_KEY); } catch (ignored) { /* ignore */ }
  }

  function focusKeySelector(key) {
    if (window.CSS && typeof CSS.escape === "function") {
      return '[data-focus-key="' + CSS.escape(key) + '"]';
    }
    return '[data-focus-key="' + key.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"]';
  }

  // Rebuild `container` via build() while restoring focus (and caret) to the
  // element marked with data-focus-key that was focused before the rebuild.
  function renderPreservingFocus(container, build) {
    var active = document.activeElement;
    var key = active && active.getAttribute ? active.getAttribute("data-focus-key") : null;
    var caret = active && typeof active.selectionStart === "number" ? active.selectionStart : null;
    build();
    if (!key) return;
    var next = container.querySelector(focusKeySelector(key));
    if (!next) return;
    next.focus();
    if (caret !== null && typeof next.setSelectionRange === "function") {
      try { next.setSelectionRange(caret, caret); } catch (ignored) { /* non-text input */ }
    }
  }

  // Single live region (#validation-error) carries everything the page must
  // say aloud. Validation errors interrupt (alert/assertive); status notes
  // such as stale-curation cleanup and save success speak politely
  // (status/polite) by retuning that one element — never a second region.
  var liveRegionEl = null;
  var validationAnnouncement = "";
  var statusAnnouncement = "";

  function getLiveRegion() {
    if (!liveRegionEl) liveRegionEl = document.getElementById("validation-error");
    return liveRegionEl;
  }

  function applyAnnouncement() {
    var node = getLiveRegion();
    if (!node) return;
    // Active validation problems take precedence over sticky status notes.
    var message = validationAnnouncement || statusAnnouncement;
    var isError = !!validationAnnouncement;
    node.textContent = message;
    if (message) node.classList.remove("hidden");
    else node.classList.add("hidden");
    node.setAttribute("role", isError ? "alert" : "status");
    node.setAttribute("aria-live", isError ? "assertive" : "polite");
  }

  function announceValidation(message) {
    validationAnnouncement = message;
    applyAnnouncement();
  }

  // Sticky informational announcement: survives subsequent empty validation
  // sweeps (renderSuggestions re-runs updateValidation on every rebuild)
  // until replaced by another status note or superseded by a real problem.
  function announceStatus(message) {
    statusAnnouncement = message;
    applyAnnouncement();
  }

  // True only while an actual validation problem is announced. Callers must
  // NOT sniff region visibility: a sticky status note also keeps the single
  // live region visible without being a failure.
  function hasActiveValidationError() {
    return !!validationAnnouncement;
  }

  // Primary actions stay disabled until the suggestion set has loaded.
  function setReviewActionsEnabled(enabled) {
    var saveBtn = document.getElementById("save-continue");
    if (saveBtn && saveBtn.textContent !== "Saved") {
      saveBtn.disabled = !enabled;
      if (enabled) saveBtn.textContent = "Save & Continue";
    }
    var acceptAll = document.getElementById("accept-all");
    if (acceptAll) acceptAll.disabled = !enabled;
  }

  // Honest degradation: replace the dead panels with an explicit error panel
  // explaining retry/abort; Save & Continue stays disabled until load succeeds.
  function renderLoadFailure(message) {
    state.loadFailed = true;
    setReviewActionsEnabled(false);
    var title = document.getElementById("suggestion-title");
    if (title) title.textContent = "Failed to load suggestions";
    var container = document.getElementById("suggestion-list");
    if (container) {
      container.innerHTML = "";
      var panel = el("div", "load-error");
      panel.id = "suggestions-error";
      panel.appendChild(el("h3", "load-error-title", "Suggestions could not be loaded"));
      panel.appendChild(el("p", "", "The scenario list failed to load (" + message + "). Curation is unavailable: retry the load, or close this window to abort."));
      var retryBtn = el("button", "btn btn-small btn-primary", "Retry load");
      retryBtn.type = "button";
      retryBtn.id = "retry-load";
      retryBtn.addEventListener("click", function () {
        retryBtn.disabled = true;
        retryBtn.textContent = "Retrying…";
        fetchSuggestions();
      });
      panel.appendChild(retryBtn);
      container.appendChild(panel);
    }
    announceValidation("Failed to load suggestions: " + message);
  }

  function fetchSuggestions() {
    return fetch(apiBase + "/suggestions", { headers: { Accept: "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("suggestions fetch failed"); return r.json(); })
      .then(function (data) {
        state.suggestions = data.suggestions || [];
        state.corpus_summary = data.corpus_summary || {};
        state.personas = data.personas || { existing: [], suggested: [] };
        state.autoAccept = !!data.auto_accept_flag;
        // init accepted all checked by default
        state.suggestions.forEach(function (s) { state.accepted[s.id] = true; });
        restoreCuration();
        state.loadFailed = false;
        setReviewActionsEnabled(true);
        if (state.suggestions.length === 0) {
          announceStatus(
            "No scenarios were suggested — the crawl captured only " +
            ((state.corpus_summary && state.corpus_summary.pages_count) || 0) +
            " page(s) with sparse visible labels. Add a custom scenario below or try a deeper crawl (--depth 1)."
          );
        } else {
          announceValidation("");
        }
        render();
        populateCustomUrls((state.corpus_summary && state.corpus_summary.urls) || []);
      })
      .catch(function (e) {
        renderLoadFailure(e && e.message ? e.message : "unknown error");
      });
  }

  function populateCustomUrls(urls) {
    var select = document.getElementById("custom-url");
    if (!select) return;
    select.innerHTML = "";
    (urls || []).forEach(function (u) {
      if (!u) return;
      var opt = document.createElement("option");
      opt.value = u;
      opt.textContent = u;
      select.appendChild(opt);
    });
  }

  function renderCrawlSummary() {
    var container = document.getElementById("crawl-summary");
    if (!container) return;
    container.innerHTML = "";
    var summary = state.corpus_summary;
    if (!summary || !summary.urls) {
      container.appendChild(el("p", "muted", "No crawl data."));
      return;
    }
    var count = summary.pages_count || summary.urls.length;
    var depth = summary.depth || 0;
    container.appendChild(el("div", "muted", "Pages: " + count + " / " + count + " | Depth: " + depth));
    var urlList = el("div", "");
    (summary.urls || []).forEach(function (u) {
      var a = el("a", "crawl-url", u);
      a.href = u;
      a.target = "_blank";
      a.rel = "noopener";
      urlList.appendChild(a);
    });
    container.appendChild(urlList);
  }

  function evaluationTargetHasLabel(et) {
    if (!et) return false;
    if (typeof et.label === "string" && et.label.trim()) return true;
    var labels = et.labels_by_version || {};
    return Object.keys(labels).some(function (k) { return String(labels[k] || "").trim(); });
  }

  function renderPersonaPanel() {
    var container = document.getElementById("persona-panel");
    if (!container) return;
    renderPreservingFocus(container, function () {
      container.innerHTML = "";
      var personas = state.personas;
      var existing = personas.existing || [];
      var suggested = personas.suggested || [];
      // Normalize selection so the rendered radio checked-state always matches state.
      if (!state.personaSelection.mode) {
        state.personaSelection.mode = existing.length ? "existing" : suggested.length ? "suggested" : "custom";
      }
      if (state.personaSelection.mode === "existing") {
        var currentExisting = (state.personaSelection.persona_ids || [])[0];
        if (!existing.some(function (p) { return p.id === currentExisting; })) {
          state.personaSelection.persona_ids = existing.length ? [existing[0].id] : [];
        }
      } else if (state.personaSelection.mode === "suggested") {
        var currentSuggested = (state.personaSelection.persona_ids || [])[0];
        if (!suggested.some(function (p) { return p.id === currentSuggested; })) {
          state.personaSelection.persona_ids = suggested.length ? [suggested[0].id] : [];
        }
      }
      function addOption(mode, label, value, checked) {
        var opt = el("label", "persona-option" + (checked ? " selected" : ""));
        var radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "persona-mode";
        radio.value = mode;
        radio.checked = !!checked;
        radio.setAttribute("data-focus-key", "persona:" + mode + ":" + value);
        radio.addEventListener("change", function () {
          state.personaSelection.mode = mode;
          if (mode === "existing" || mode === "suggested") {
            state.personaSelection.persona_ids = [value];
            delete state.personaSelection.custom_persona;
          }
          persistCuration();
          renderPersonaPanel();
        });
        opt.appendChild(radio);
        opt.appendChild(el("span", "", label + (value ? ": " + value : "")));
        container.appendChild(opt);
      }
      if (existing.length) {
        container.appendChild(el("h4", "eyebrow", "Existing personas"));
        existing.forEach(function (p) {
          var checked = state.personaSelection.mode === "existing" && (state.personaSelection.persona_ids || [])[0] === p.id;
          addOption("existing", p.name || p.id, p.id, checked);
        });
      }
      if (suggested.length) {
        container.appendChild(el("h4", "eyebrow", "Model-suggested personas"));
        suggested.forEach(function (p) {
          var checked = state.personaSelection.mode === "suggested" && (state.personaSelection.persona_ids || [])[0] === p.id;
          addOption("suggested", p.name || p.id, p.id, checked);
        });
      }
      // custom persona option
      var customChecked = state.personaSelection.mode === "custom";
      var customOpt = el("label", "persona-option" + (customChecked ? " selected" : ""));
      var customRadio = document.createElement("input");
      customRadio.type = "radio";
      customRadio.name = "persona-mode";
      customRadio.value = "custom";
      customRadio.checked = customChecked;
      customRadio.setAttribute("data-focus-key", "persona:custom:custom");
      customRadio.addEventListener("change", function () { state.personaSelection.mode = "custom"; persistCuration(); renderPersonaPanel(); });
      customOpt.appendChild(customRadio);
      customOpt.appendChild(el("span", "", "Create custom persona"));
      container.appendChild(customOpt);

      if (customChecked) {
        var form = el("div", "custom-persona-form");
        var defaults = state.personaSelection.custom_persona || { id: "custom-persona", name: "Custom Persona", working_memory_capacity: 4, initial_confidence: 0.6, initial_frustration: 0.2, abandonment_threshold: 0.8, attention_temperature: 1.0 };
        state.personaSelection.custom_persona = defaults;
        function textField(key, label) {
          var wrap = el("label", "", label);
          var input = document.createElement("input");
          input.type = "text";
          input.value = defaults[key];
          input.setAttribute("data-focus-key", "persona-custom:" + key);
          input.addEventListener("input", function () { defaults[key] = input.value; state.personaSelection.custom_persona = defaults; persistCuration(); });
          wrap.appendChild(input);
          form.appendChild(wrap);
        }
        function sliderField(key, label, min, max, step) {
          var wrap = el("label", "slider-row", label + " (" + defaults[key] + ")");
          var input = document.createElement("input");
          input.type = "range";
          input.min = String(min);
          input.max = String(max);
          input.step = String(step);
          input.value = String(defaults[key]);
          input.setAttribute("data-focus-key", "persona-custom:" + key);
          var valSpan = el("span", "slider-value", String(defaults[key]));
          input.addEventListener("input", function () {
            defaults[key] = parseFloat(input.value);
            wrap.firstChild.textContent = label + " (" + input.value + ")";
            valSpan.textContent = String(input.value);
            state.personaSelection.custom_persona = defaults;
            persistCuration();
          });
          wrap.appendChild(input);
          wrap.appendChild(valSpan);
          form.appendChild(wrap);
        }
        textField("id", "Persona ID");
        textField("name", "Persona name");
        sliderField("working_memory_capacity", "Working memory capacity", 1, 10, 1);
        sliderField("initial_confidence", "Initial confidence", 0, 1, 0.05);
        sliderField("initial_frustration", "Initial frustration", 0, 1, 0.05);
        sliderField("attention_temperature", "Attention temperature", 0.1, 2.0, 0.1);
        sliderField("abandonment_threshold", "Abandonment threshold", 0, 1, 0.05);
        container.appendChild(form);
      }
    });
  }

  function createCard(s) {
    var card = el("article", "suggestion-card");
    card.dataset.id = s.id;
    card.tabIndex = 0;
    var accepted = state.accepted[s.id] !== false;
    card.dataset.accepted = accepted ? "true" : "false";
    var header = el("div", "suggestion-card-header");
    var checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "suggestion-checkbox";
    checkbox.checked = accepted;
    checkbox.setAttribute("aria-label", "Accept " + s.id);
    checkbox.setAttribute("data-focus-key", s.id + ":checkbox");
    checkbox.addEventListener("change", function () {
      state.accepted[s.id] = checkbox.checked;
      card.dataset.accepted = checkbox.checked ? "true" : "false";
      persistCuration();
      updateTopCount();
    });
    header.appendChild(checkbox);
    var main = el("div", "suggestion-main");
    var goal = el("h3", "suggestion-goal", s.goal || "");
    main.appendChild(goal);
    main.appendChild(el("div", "suggestion-meta", (s.start_url || "") + " | verifier: " + ((s.verifier && s.verifier.text) || "")));
    if (s.rationale) main.appendChild(el("p", "suggestion-rationale", s.rationale));
    if (s.coverage && s.coverage.length) {
      var tags = el("div", "coverage-tags");
      s.coverage.forEach(function (c) { tags.appendChild(el("span", "tag", c)); });
      main.appendChild(tags);
    }
    header.appendChild(main);
    card.appendChild(header);

    // Keyboard support: Enter/Space on the focused card toggles acceptance.
    card.addEventListener("keydown", function (e) {
      if (e.target !== card) return;
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      var next = !(card.dataset.accepted === "true");
      state.accepted[s.id] = next;
      checkbox.checked = next;
      card.dataset.accepted = next ? "true" : "false";
      persistCuration();
      updateTopCount();
    });

    var actions = el("div", "card-actions");
    var editBtn = el("button", "btn btn-small", "Edit");
    editBtn.type = "button";
    editBtn.setAttribute("data-focus-key", s.id + ":edit");
    var dupBtn = el("button", "btn btn-small", "Duplicate");
    dupBtn.type = "button";
    dupBtn.setAttribute("data-focus-key", s.id + ":duplicate");
    var delBtn = el("button", "btn btn-small btn-danger", "Delete");
    delBtn.type = "button";
    delBtn.setAttribute("data-focus-key", s.id + ":delete");
    actions.appendChild(editBtn);
    actions.appendChild(dupBtn);
    actions.appendChild(delBtn);
    card.appendChild(actions);

    var editPanel = el("div", "edit-panel");
    function editTextField(labelText, initial, focusKey) {
      var label = el("label", "", labelText);
      var input = document.createElement("input");
      input.type = "text";
      input.value = initial;
      input.setAttribute("data-focus-key", focusKey);
      label.appendChild(input);
      editPanel.appendChild(label);
      return input;
    }
    var nameInput = editTextField("Name", s.name || "", s.id + ":name");
    // goal (textarea)
    var goalLabel = el("label", "", "Goal");
    var goalInput = document.createElement("textarea");
    goalInput.value = s.goal || "";
    goalInput.setAttribute("data-focus-key", s.id + ":goal");
    goalLabel.appendChild(goalInput);
    editPanel.appendChild(goalLabel);
    var verInput = editTextField("Verifier text", (s.verifier && s.verifier.text) || "", s.id + ":verifier-text");
    var roleInput = editTextField("Verifier role", (s.verifier && s.verifier.role) || "", s.id + ":verifier-role");
    roleInput.placeholder = "heading, button, etc.";
    // evaluation target
    var evalVal = "";
    if (s.evaluation_target) {
      if (typeof s.evaluation_target === "string") evalVal = s.evaluation_target;
      else if (s.evaluation_target.label) evalVal = s.evaluation_target.label;
      else if (s.evaluation_target.labels_by_version && s.evaluation_target.labels_by_version.live) evalVal = s.evaluation_target.labels_by_version.live;
      else if (s.evaluation_target.labels_by_version) evalVal = Object.values(s.evaluation_target.labels_by_version)[0] || "";
    }
    var evalInput = editTextField("Evaluation target label", evalVal, s.id + ":eval-label");
    var evalRoleInput = editTextField("Evaluation target role", (s.evaluation_target && s.evaluation_target.role) || "", s.id + ":eval-role");
    // budget sliders
    var budget = s.budget || { max_steps: 20, max_observations: 18, max_interactions: 8, timeout_seconds: null, stall_timeout_seconds: 90, max_model_calls: 32 };
    var budgetGrid = el("div", "budget-grid");
    function budgetSlider(key, label, min, max, step) {
      var wrap = el("label", "slider-row", label);
      var range = document.createElement("input");
      range.type = "range";
      range.min = String(min); range.max = String(max); range.step = String(step);
      range.value = String(budget[key] !== undefined ? budget[key] : min);
      range.setAttribute("data-focus-key", s.id + ":budget-" + key);
      var val = el("span", "slider-value", String(range.value));
      range.addEventListener("input", function () { val.textContent = range.value; budget[key] = parseInt(range.value, 10); });
      wrap.appendChild(range);
      wrap.appendChild(val);
      budgetGrid.appendChild(wrap);
    }
    budgetSlider("max_steps", "Max steps", 1, 50, 1);
    budgetSlider("max_observations", "Max observations", 1, 30, 1);
    budgetSlider("max_interactions", "Max interactions", 1, 30, 1);
    budgetSlider("max_model_calls", "Max model calls", 1, 64, 1);
    editPanel.appendChild(budgetGrid);
    var timeoutWrap = el("label", "", "Timeout seconds (empty = progress-based only)");
    var timeoutInput = document.createElement("input");
    timeoutInput.type = "number";
    timeoutInput.min = "10"; timeoutInput.max = "600"; timeoutInput.step = "10";
    timeoutInput.value = budget.timeout_seconds === null || budget.timeout_seconds === undefined ? "" : String(budget.timeout_seconds);
    timeoutInput.setAttribute("data-focus-key", s.id + ":timeout");
    timeoutWrap.appendChild(timeoutInput);
    editPanel.appendChild(timeoutWrap);

    var saveEditBtn = el("button", "btn btn-small btn-primary", "Save edit");
    saveEditBtn.type = "button";
    saveEditBtn.setAttribute("data-focus-key", s.id + ":save-edit");
    editPanel.appendChild(saveEditBtn);

    // Visual-only inline error; announcements go through the single live
    // region (#validation-error) via announceValidation.
    var editError = el("div", "inline-error hidden", "");
    editPanel.appendChild(editError);

    card.appendChild(editPanel);

    editBtn.addEventListener("click", function () {
      var isOpen = editPanel.hasAttribute("open");
      if (isOpen) editPanel.removeAttribute("open"); else editPanel.setAttribute("open", "");
    });

    saveEditBtn.addEventListener("click", function () {
      var newGoal = goalInput.value.trim();
      var newVerifier = verInput.value.trim();
      var newEvalLabel = evalInput.value.trim();
      var errors = [];
      if (!newGoal) { goalInput.classList.add("invalid"); errors.push("Goal must not be empty"); } else { goalInput.classList.remove("invalid"); }
      if (!newVerifier) { verInput.classList.add("invalid"); errors.push("Verifier text must not be empty"); } else { verInput.classList.remove("invalid"); }
      var origLabels = (s.evaluation_target && s.evaluation_target.labels_by_version) || {};
      var hasOrigLabels = Object.keys(origLabels).some(function (k) { return String(origLabels[k] || "").trim(); });
      if (!newEvalLabel && !hasOrigLabels) { evalInput.classList.add("invalid"); errors.push("Evaluation target label must not be empty"); } else { evalInput.classList.remove("invalid"); }
      if (errors.length) {
        var msg = errors.join("; ");
        editError.textContent = msg;
        editError.classList.remove("hidden");
        announceValidation(msg);
        return;
      }
      editError.classList.add("hidden");
      var evaluationTarget = newEvalLabel
        ? { label: newEvalLabel, role: evalRoleInput.value.trim() || null }
        : { labels_by_version: origLabels, role: evalRoleInput.value.trim() || null };
      // persist edit
      var edited = {
        id: s.id,
        name: nameInput.value.trim() || s.name,
        goal: newGoal,
        start_url: s.start_url,
        verifier: { type: "visible-result", text: newVerifier, role: roleInput.value.trim() || null, all_of: (s.verifier && s.verifier.all_of) || [] },
        evaluation_target: evaluationTarget,
        budget: { max_steps: parseInt(budget.max_steps, 10), max_observations: parseInt(budget.max_observations, 10), max_interactions: parseInt(budget.max_interactions, 10), max_model_calls: parseInt(budget.max_model_calls, 10), timeout_seconds: timeoutInput.value.trim() === "" ? null : parseInt(timeoutInput.value, 10) || null, stall_timeout_seconds: budget.stall_timeout_seconds || 90 },
        rationale: s.rationale || "",
        coverage: s.coverage || []
      };
      state.edits[s.id] = edited;
      // update card display
      goal.textContent = edited.goal;
      main.querySelector(".suggestion-meta").textContent = edited.start_url + " | verifier: " + edited.verifier.text;
      persistCuration();
      updateValidation();
      // close panel
      editPanel.removeAttribute("open");
    });

    dupBtn.addEventListener("click", function () {
      var newId = s.id + "-copy";
      var counter = 1;
      var existingIds = state.suggestions.map(function (x) { return x.id; }).concat(state.added.map(function (x) { return x.id; })).concat(Object.keys(state.edits));
      while (existingIds.indexOf(newId) !== -1) { counter += 1; newId = s.id + "-copy" + counter; }
      var dup = JSON.parse(JSON.stringify(state.edits[s.id] || s));
      dup.id = newId;
      dup.name = (dup.name || s.name) + " (copy)";
      state.added.push(dup);
      state.accepted[newId] = true;
      persistCuration();
      renderSuggestions();
    });

    delBtn.addEventListener("click", function () {
      // remove from accepted/edits/added/suggestions
      if (state.added.some(function (x) { return x.id === s.id; })) {
        state.added = state.added.filter(function (x) { return x.id !== s.id; });
        delete state.accepted[s.id];
        delete state.edits[s.id];
        persistCuration();
        renderSuggestions();
        return;
      }
      // for original suggestions, uncheck and mark as not accepted (or remove edit)
      delete state.edits[s.id];
      state.accepted[s.id] = false;
      checkbox.checked = false;
      card.dataset.accepted = "false";
      persistCuration();
      updateTopCount();
      updateValidation();
    });

    // validation on input
    function clearInvalidOnInput(input) {
      input.addEventListener("input", function () {
        if (input.value.trim()) input.classList.remove("invalid");
      });
    }
    clearInvalidOnInput(goalInput);
    clearInvalidOnInput(verInput);
    clearInvalidOnInput(evalInput);

    return card;
  }

  function renderSuggestions() {
    var container = document.getElementById("suggestion-list");
    if (!container) return;
    renderPreservingFocus(container, function () {
      container.innerHTML = "";
      var all = state.suggestions.concat(state.added);
      all.forEach(function (s) {
        var display = state.edits[s.id] || s;
        container.appendChild(createCard(display));
      });
      applyRovingTabindex();
    });
    updateTopCount();
    updateValidation();
  }

  // Roving tabindex across suggestion cards: exactly one card is tabbable.
  function applyRovingTabindex() {
    var cards = suggestionCards();
    if (!cards.length) return;
    var current = cards.indexOf(document.activeElement);
    var active = current >= 0 ? cards[current] : cards[0];
    cards.forEach(function (c) { c.tabIndex = c === active ? 0 : -1; });
  }

  function suggestionCards() {
    return Array.prototype.slice.call(
      document.querySelectorAll("#suggestion-list .suggestion-card")
    );
  }

  function handleCardNavKeydown(e) {
    if (e.target === e.currentTarget) return; // container-level delegation
    var target = e.target;
    if (!target.classList || !target.classList.contains("suggestion-card")) return;
    var cards = suggestionCards();
    var idx = cards.indexOf(target);
    if (idx < 0) return;
    var nextIdx = null;
    if (e.key === "ArrowDown") nextIdx = Math.min(cards.length - 1, idx + 1);
    else if (e.key === "ArrowUp") nextIdx = Math.max(0, idx - 1);
    else if (e.key === "Home") nextIdx = 0;
    else if (e.key === "End") nextIdx = cards.length - 1;
    if (nextIdx === null || nextIdx === idx) return;
    e.preventDefault();
    cards.forEach(function (c) { c.tabIndex = c === cards[nextIdx] ? 0 : -1; });
    cards[nextIdx].focus();
  }

  // Canonical selection count, single source: truthy-accepted ORIGINAL
  // suggestions plus the added scenarios. Added ids are excluded from the
  // accepted side (they are counted via `added` itself), so a custom
  // scenario is never counted twice.
  function curatedSelectionCount() {
    var addedIds = {};
    state.added.forEach(function (s) { addedIds[s.id] = true; });
    var originalsSelected = Object.keys(state.accepted).filter(function (id) {
      return state.accepted[id] && !addedIds[id];
    }).length;
    return originalsSelected + state.added.length;
  }

  function updateTopCount() {
    var countEl = document.getElementById("scenario-count");
    if (countEl) {
      var acceptedCount = curatedSelectionCount();
      countEl.textContent = acceptedCount + " scenario" + (acceptedCount !== 1 ? "s" : "") + " selected";
    }
    var badge = document.getElementById("auto-accept-badge");
    // Class toggle only: inline style writes lose to `.hidden{display:none !important}`.
    if (badge) badge.classList.toggle("hidden", !state.autoAccept);
  }

  function updateValidation() {
    var seen = {};
    var dup = null;
    var emptyVerifier = false;
    var emptyGoal = false;
    var emptyEval = false;
    var all = state.suggestions.concat(state.added);
    all.forEach(function (s) {
      var cur = state.edits[s.id] || s;
      if (!cur.goal || !String(cur.goal).trim()) emptyGoal = true;
      if (!cur.verifier || !cur.verifier.text || !String(cur.verifier.text).trim()) emptyVerifier = true;
      if (!evaluationTargetHasLabel(cur.evaluation_target)) emptyEval = true;
      if (seen[cur.id]) dup = cur.id; else seen[cur.id] = true;
    });
    var msgs = [];
    if (emptyGoal) msgs.push("Goal must not be empty");
    if (emptyVerifier) msgs.push("Verifier text must not be empty");
    if (emptyEval) msgs.push("Evaluation target label must not be empty");
    if (dup) msgs.push("Duplicate scenario id: " + dup);
    announceValidation(msgs.join(" | "));
  }

  // Pure payload builder: every scenario id appears exactly once across
  // accepted_ids / edited / added. Custom scenarios live only in `added`;
  // edited originals are claimed via `edited` (server replaces them);
  // an edit of an added scenario is folded back into `added`.
  function buildCuratePayload(curState) {
    var s = curState || {};
    var acceptedMap = s.accepted || {};
    var editsMap = s.edits || {};
    var addedList = Array.isArray(s.added) ? s.added : [];
    var addedIds = {};
    addedList.forEach(function (item) {
      if (item && item.id !== undefined && item.id !== null) {
        addedIds[String(item.id)] = true;
      }
    });
    var claimed = {};
    var acceptedIds = [];
    Object.keys(acceptedMap).forEach(function (id) {
      if (!acceptedMap[id]) return;
      var key = String(id);
      if (addedIds[key] || editsMap[key] || claimed[key]) return;
      claimed[key] = true;
      acceptedIds.push(key);
    });
    var edited = [];
    Object.keys(editsMap).forEach(function (id) {
      var key = String(id);
      if (addedIds[key] || acceptedMap[key] === false || claimed[key]) return;
      claimed[key] = true;
      edited.push(editsMap[id]);
    });
    var added = [];
    addedList.forEach(function (item) {
      if (!item) return;
      var key = String(item.id);
      if (claimed[key]) return;
      claimed[key] = true;
      added.push(editsMap[key] || item);
    });
    var selection = s.personaSelection || { mode: "existing" };
    var personaSelection;
    if (selection.mode === "custom" && selection.custom_persona) {
      personaSelection = { mode: "custom", custom_persona: selection.custom_persona };
    } else {
      personaSelection = {
        mode: selection.mode === "suggested" ? "suggested" : "existing",
        persona_ids: Array.isArray(selection.persona_ids) ? selection.persona_ids : []
      };
    }
    return {
      accepted_ids: acceptedIds,
      edited: edited,
      added: added,
      persona_selection: personaSelection,
      auto_accept_flag: !!s.autoAccept,
      suggestions_signature: typeof s.suggestionsSignature === "string" ? s.suggestionsSignature : null
    };
  }

  function render() {
    renderCrawlSummary();
    renderSuggestions();
    renderPersonaPanel();
    updateTopCount();
    var title = document.getElementById("suggestion-title");
    if (title) title.textContent = "Suggested Scenarios (" + state.suggestions.length + ")";
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Chain-of-custody: echo the server-signed suggestion-set signature.
    var sigMeta = document.querySelector('meta[name="uxa-suggestions-signature"]');
    state.suggestionsSignature = sigMeta ? (sigMeta.getAttribute("content") || "") : "";
    fetchSuggestions();
    var listContainer = document.getElementById("suggestion-list");
    if (listContainer) listContainer.addEventListener("keydown", handleCardNavKeydown);
    var acceptAllBtn = document.getElementById("accept-all");
    if (acceptAllBtn) acceptAllBtn.addEventListener("click", function () {
      state.suggestions.forEach(function (s) { state.accepted[s.id] = true; });
      state.added.forEach(function (s) { state.accepted[s.id] = true; });
      document.querySelectorAll(".suggestion-checkbox").forEach(function (cb) { cb.checked = true; cb.closest(".suggestion-card").dataset.accepted = "true"; });
      persistCuration();
      updateTopCount();
    });
    var saveBtn = document.getElementById("save-continue");
    if (saveBtn) saveBtn.addEventListener("click", function () {
      updateValidation();
      // Guard on announcement state, not region visibility: a sticky status
      // note (e.g. stale-entry discard) keeps the region visibly populated
      // without being a validation failure.
      if (hasActiveValidationError()) return;
      var payload = buildCuratePayload(state);
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving...";
      fetch(apiBase + "/curate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, status: r.status, body: j }; }); })
        .then(function (res) {
          if (!res.ok) {
            var detail = res.body.detail;
            var message = typeof detail === "string" ? detail : Array.isArray(detail)
              ? detail.map(function (d) { return (d.loc || []).join(".") + ": " + d.msg; }).join("; ")
              : "Validation failed";
            announceValidation(message);
            saveBtn.disabled = false;
            saveBtn.textContent = "Save & Continue";
            return;
          }
          // success: show curated preview; announce politely through the
          // shared live region (the visual banner alone is never spoken).
          var curatedCount = res.body.curated ? res.body.curated.length : 0;
          var preview = document.getElementById("fragment-preview");
          var curatedInfo = document.getElementById("curated-info");
          if (preview) { preview.textContent = res.body.fragment_yaml || JSON.stringify(res.body.curated, null, 2); preview.classList.remove("hidden"); }
          if (curatedInfo) { curatedInfo.textContent = "Curated " + curatedCount + " scenarios"; curatedInfo.classList.remove("hidden"); }
          var banner = document.getElementById("success-banner");
          var successText = "Curated " + curatedCount + " scenarios successfully. You may close this window.";
          if (banner) { banner.textContent = successText; banner.classList.remove("hidden"); }
          announceStatus(successText);
          saveBtn.textContent = "Saved";
          var heading = document.getElementById("preview-heading");
          if (heading) heading.classList.remove("hidden");
        })
        .catch(function (e) {
          announceValidation("Save failed: " + e.message);
          saveBtn.disabled = false;
          saveBtn.textContent = "Save & Continue";
        });
    });
    var addCustomBtn = document.getElementById("add-custom-btn");
    var addCustomForm = document.getElementById("add-custom-form");
    if (addCustomBtn && addCustomForm) {
      var detailsEl = addCustomForm.closest("details");
      addCustomBtn.addEventListener("click", function () {
        if (!detailsEl) return;
        detailsEl.open = !detailsEl.open;
      });
      var targetInput = document.getElementById("custom-target");
      if (targetInput) targetInput.addEventListener("input", function () { if (targetInput.value.trim()) targetInput.classList.remove("invalid"); });
      var customSave = document.getElementById("custom-save");
      if (customSave) customSave.addEventListener("click", function () {
        var idInput = document.getElementById("custom-id");
        var nameInput = document.getElementById("custom-name");
        var goalInput = document.getElementById("custom-goal");
        var urlSelect = document.getElementById("custom-url");
        var verifierInput = document.getElementById("custom-verifier");
        var targetField = document.getElementById("custom-target");
        var err = document.getElementById("custom-error");
        var id = idInput.value.trim();
        var goal = goalInput.value.trim();
        var verifier = verifierInput.value.trim();
        var target = targetField.value.trim();
        var msgs = [];
        if (!id) msgs.push("ID required");
        if (!goal) msgs.push("Goal must not be empty");
        if (!verifier) msgs.push("Verifier text must not be empty");
        if (!target) { targetField.classList.add("invalid"); msgs.push("Evaluation target label must not be empty"); }
        var existingIds = state.suggestions.map(function (s) { return s.id; }).concat(state.added.map(function (s) { return s.id; })).concat(Object.keys(state.edits));
        if (existingIds.indexOf(id) !== -1) msgs.push("Duplicate scenario id: " + id);
        if (msgs.length) {
          var msg = msgs.join("; ");
          err.textContent = msg;
          err.classList.remove("hidden");
          announceValidation(msg);
          return;
        }
        err.classList.add("hidden");
        var newScenario = {
          id: id,
          name: nameInput.value.trim() || id,
          goal: goal,
          start_url: urlSelect.value,
          verifier: { type: "visible-result", text: verifier, role: null, all_of: [] },
          evaluation_target: { label: target },
          budget: { max_steps: 20, max_observations: 18, max_interactions: 8, timeout_seconds: null, stall_timeout_seconds: 90, max_model_calls: 32 },
          rationale: "Custom scenario",
          coverage: ["custom"]
        };
        state.added.push(newScenario);
        state.accepted[id] = true;
        persistCuration();
        renderSuggestions();
        if (detailsEl) detailsEl.open = false;
        idInput.value = ""; nameInput.value = ""; goalInput.value = ""; verifierInput.value = ""; targetField.value = "";
      });
    }

    var resetBtn = document.getElementById("reset-curation");
    if (resetBtn) resetBtn.addEventListener("click", function () {
      clearPersistedCuration();
      window.location.reload();
    });
  });

  // Pure builder exposed for contract tests (no DOM access required).
  if (typeof window !== "undefined") {
    window.buildCuratePayload = buildCuratePayload;
  }
})();
