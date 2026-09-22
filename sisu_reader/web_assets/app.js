(function () {
  "use strict";

  var API = {
    status: "/api/status",
    session: "/api/session",
    reset: "/api/session/reset",
    retry: "/api/retry",
    ask: "/api/ask",
    job: "/api/jobs/"
  };
  var SESSION_KEY = "sisu-reader-session-v1";
  var TOKEN = /^[A-Za-z0-9_-]{24,160}$/;
  var STAGES = ["authorizing", "exact_search", "screening", "adaptive_wave", "reading", "evidence_review", "synthesizing", "claim_review", "binding_citations"];
  var STAGE_LABELS = {
    authorizing: "Checking access",
    exact_search: "Finding exact matches",
    screening: "Choosing documents",
    reading: "Reading sources",
    adaptive_wave: "Searching for more evidence",
    evidence_review: "Checking evidence gaps",
    synthesizing: "Writing the answer",
    claim_review: "Checking claims against sources",
    binding_citations: "Adding citations"
  };

  var elements = {
    engineCard: document.getElementById("engine-card"),
    statusLabel: document.getElementById("status-label"),
    statusDetail: document.getElementById("status-detail"),
    engineAction: document.getElementById("engine-action"),
    engineMessage: document.getElementById("engine-message"),
    engineCommand: document.getElementById("engine-command"),
    copyCommand: document.getElementById("copy-command"),
    retryEngine: document.getElementById("retry-engine"),
    banner: document.getElementById("engine-banner"),
    bannerTitle: document.getElementById("banner-title"),
    bannerText: document.getElementById("banner-text"),
    bannerRetry: document.getElementById("banner-retry"),
    pipeline: document.getElementById("pipeline"),
    conversation: document.getElementById("conversation"),
    welcome: document.getElementById("welcome"),
    messages: document.getElementById("messages"),
    form: document.getElementById("composer"),
    input: document.getElementById("question"),
    send: document.getElementById("send"),
    newConversation: document.getElementById("new-conversation"),
    mobileNew: document.getElementById("mobile-new"),
    mobileDetails: document.getElementById("mobile-details"),
    inspector: document.getElementById("inspector"),
    inspectorClose: document.getElementById("inspector-close"),
    backdrop: document.getElementById("backdrop"),
    sourcesTab: document.getElementById("sources-tab"),
    coverageTab: document.getElementById("coverage-tab"),
    runTab: document.getElementById("run-tab"),
    sourcesPanel: document.getElementById("sources-panel"),
    coveragePanel: document.getElementById("coverage-panel"),
    runPanel: document.getElementById("run-panel"),
    sourceCount: document.getElementById("source-count"),
    sourcesEmpty: document.getElementById("sources-empty"),
    sourceList: document.getElementById("source-list"),
    coverageEmpty: document.getElementById("coverage-empty"),
    coverageContent: document.getElementById("coverage-content"),
    runEmpty: document.getElementById("run-empty"),
    runContent: document.getElementById("run-content"),
    toasts: document.getElementById("toasts")
  };

  var state = {
    session: "",
    engineReady: false,
    engineError: null,
    busy: false,
    activeJob: "",
    lastJob: "",
    lastAnswer: null,
    statusTimer: null,
    traceToken: 0,
    inspectorReturnFocus: null
  };

  function node(tag, className, text) {
    var item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined && text !== null) item.textContent = String(text);
    return item;
  }

  function openSource(sourceId) {
    openInspector("sources");
    window.setTimeout(function () {
      var target = elements.sourceList.querySelector('[data-source-id="' + sourceId + '"]');
      if (!target) return;
      target.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("source-highlight");
      window.setTimeout(function () { target.classList.remove("source-highlight"); }, 1400);
    }, 80);
  }

  function appendInline(parent, value, sourceHandler, sourceIds) {
    var text = String(value || "");
    var pattern = /(\*\*[^*\n]+\*\*|\[S\d+\])/g;
    var cursor = 0;
    var match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      var token = match[0];
      if (token.slice(0, 2) === "**") {
        parent.appendChild(node("strong", "", token.slice(2, -2)));
      } else {
        var sourceId = token.slice(1, -1);
        if (sourceIds && !sourceIds[sourceId]) {
          parent.appendChild(document.createTextNode(token));
          cursor = pattern.lastIndex;
          continue;
        }
        var citation = node("button", "source-ref", sourceId.slice(1));
        citation.type = "button";
        citation.setAttribute("aria-label", "Open source " + sourceId.slice(1));
        citation.addEventListener("click", function (event) {
          if (typeof sourceHandler === "function") sourceHandler(event.currentTarget.dataset.sourceId);
          else openSource(event.currentTarget.dataset.sourceId);
        });
        citation.dataset.sourceId = sourceId;
        parent.appendChild(citation);
      }
      cursor = pattern.lastIndex;
    }
    if (cursor < text.length) parent.appendChild(document.createTextNode(text.slice(cursor)));
  }

  function renderAnswerText(value, sourceHandler, sourceIds) {
    var body = node("div", "answer-body");
    var lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
    var paragraph = [];
    var list = null;
    var listType = "";

    function flushParagraph() {
      if (!paragraph.length) return;
      var text = paragraph.join(" ").trim();
      paragraph = [];
      if (!text) return;
      var heading = text.match(/^(#{1,4})\s+(.+)$/);
      var item = node(heading ? "h" + Math.min(4, heading[1].length + 1) : "p");
      appendInline(item, heading ? heading[2] : text, sourceHandler, sourceIds);
      body.appendChild(item);
    }

    function closeList() {
      list = null;
      listType = "";
    }

    lines.forEach(function (rawLine) {
      var bullet = rawLine.match(/^(\s*)([*+-]|\d+[.)])\s+(.+)$/);
      if (bullet) {
        flushParagraph();
        var ordered = /^\d/.test(bullet[2]);
        var desired = ordered ? "ol" : "ul";
        if (!list || listType !== desired) {
          list = node(desired, "answer-list");
          listType = desired;
          body.appendChild(list);
        }
        var entry = node("li", bullet[1].length >= 2 ? "nested" : "");
        appendInline(entry, bullet[3].trim(), sourceHandler, sourceIds);
        list.appendChild(entry);
        return;
      }
      if (!rawLine.trim()) {
        flushParagraph();
        closeList();
        return;
      }
      closeList();
      paragraph.push(rawLine.trim());
    });
    flushParagraph();
    if (!body.childNodes.length) body.appendChild(node("p", "", "The engine returned no answer text."));
    return body;
  }

  function importantWarning(value) {
    return /technical error|without a valid|not paired|output limit|unavailable|failed/i.test(String(value || ""));
  }

  function requestId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return "web_" + window.crypto.randomUUID();
    }
    return "web_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2);
  }

  function ApiError(message, status, code, action, command) {
    this.name = "ApiError";
    this.message = message || "The local service returned an error.";
    this.status = status || 0;
    this.code = code || "";
    this.action = action || "";
    this.command = command || "";
  }
  ApiError.prototype = Object.create(Error.prototype);

  async function requestJson(url, options) {
    var settings = options || {};
    var headers = { "Accept": "application/json" };
    if (state.session) headers["X-SISU-Session"] = state.session;
    if (settings.body !== undefined) headers["Content-Type"] = "application/json";
    var response;
    try {
      response = await window.fetch(url, {
        method: settings.method || "GET",
        headers: headers,
        body: settings.body === undefined ? undefined : JSON.stringify(settings.body),
        credentials: "same-origin",
        cache: "no-store"
      });
    } catch (_error) {
      throw new ApiError("The local SISU service is unavailable.", 0, "network_error", "Start the UI server and retry.", "sisu-reader ui");
    }
    var payload = {};
    try { payload = await response.json(); } catch (_error) { payload = {}; }
    if (!response.ok) {
      var raw = payload && payload.error && typeof payload.error === "object" ? payload.error : {};
      throw new ApiError(raw.message || "The local request failed.", response.status, raw.code, raw.action, raw.command);
    }
    return payload;
  }

  function toast(message) {
    var item = node("div", "toast", message);
    elements.toasts.appendChild(item);
    window.setTimeout(function () { item.remove(); }, 4200);
  }

  function saveSession(token) {
    state.session = TOKEN.test(String(token || "")) ? String(token) : "";
    if (state.session) window.sessionStorage.setItem(SESSION_KEY, state.session);
    else window.sessionStorage.removeItem(SESSION_KEY);
  }

  async function createSession() {
    var payload = await requestJson(API.session, { method: "POST", body: {} });
    saveSession(payload.session_id);
    if (!state.session) throw new ApiError("The local service returned an invalid session.", 500, "invalid_session");
    return state.session;
  }

  async function ensureSession(force) {
    if (!force && TOKEN.test(state.session)) return state.session;
    return createSession();
  }

  function display(value) {
    if (typeof value === "string") return value;
    try { return JSON.stringify(value, null, 2); } catch (_error) { return String(value); }
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "";
    if (seconds < 10) return seconds.toFixed(1) + "s";
    if (seconds < 60) return Math.round(seconds) + "s";
    return Math.floor(seconds / 60) + "m " + Math.round(seconds % 60) + "s";
  }

  function autosize() {
    elements.input.style.height = "auto";
    elements.input.style.height = Math.min(elements.input.scrollHeight, 170) + "px";
  }

  function setBusy(value) {
    state.busy = Boolean(value);
    elements.send.disabled = state.busy;
    elements.send.textContent = state.busy ? "Thinking…" : "Ask ↑";
  }

  function renderPipeline(stage, status) {
    var current = String(stage || "");
    // The compact sidebar groups the additional research/checking steps.
    // Match named stages, never positions in two differently sized lists.
    var compactStage = {
      adaptive_wave: "exact_search",
      evidence_review: "reading",
      claim_review: "synthesizing"
    };
    var visible = Object.prototype.hasOwnProperty.call(compactStage, current) ? compactStage[current] : current;
    var index = STAGES.indexOf(visible);
    elements.pipeline.querySelectorAll("li[data-stage]").forEach(function (item) {
      var itemStage = item.getAttribute("data-stage");
      var position = STAGES.indexOf(itemStage);
      item.classList.toggle("active", status === "running" && itemStage === visible && index >= 0);
      item.classList.toggle("done", status === "complete" || (index >= 0 && position >= 0 && position < index));
    });
  }

  function errorText(error) {
    var parts = [error && error.message ? error.message : "The local request failed."];
    if (error && error.action) parts.push(error.action);
    if (error && error.command) parts.push(error.command);
    return parts.join("\n");
  }

  function setEngineStatus(engine) {
    var status = engine && typeof engine === "object" ? engine : {};
    var mode = String(status.state || "loading").toLowerCase();
    state.engineReady = Boolean(status.ready);
    state.engineError = status.error && typeof status.error === "object" ? status.error : null;
    elements.engineCard.classList.remove("ready", "failed");
    elements.engineAction.hidden = true;
    elements.banner.hidden = true;

    if (state.engineReady) {
      elements.engineCard.classList.add("ready");
      elements.statusLabel.textContent = "Ready · " + String(status.model || "local model");
      var detail = status.detail && typeof status.detail === "object" ? status.detail : {};
      var documents = detail.documents;
      if (documents === undefined) documents = detail.document_count;
      var units = detail.sections;
      if (units === undefined) units = detail.chunks;
      var summary = "Private local engine is ready";
      if (documents !== undefined) summary = String(documents) + " documents";
      if (units !== undefined) summary += " · " + String(units) + " readable units";
      elements.statusDetail.textContent = summary;
      return;
    }

    if (mode === "failed") {
      elements.engineCard.classList.add("failed");
      var problem = state.engineError || {};
      elements.statusLabel.textContent = "Local engine needs attention";
      elements.statusDetail.textContent = problem.message || "The engine did not start.";
      elements.engineMessage.textContent = problem.action || "Fix the local setup, then retry.";
      elements.engineCommand.textContent = problem.command || "sisu-reader doctor";
      elements.engineAction.hidden = false;
      elements.bannerTitle.textContent = problem.message || "The local engine did not start.";
      elements.bannerText.textContent = problem.action || "Your question field remains usable while you fix it.";
      elements.banner.hidden = false;
      return;
    }

    elements.statusLabel.textContent = "Starting the private engine…";
    elements.statusDetail.textContent = "You can type while the index and model open locally.";
    elements.bannerTitle.textContent = "Starting locally.";
    elements.bannerText.textContent = "The page is interactive; answers become available when startup finishes.";
    elements.banner.hidden = false;
  }

  function scheduleStatus(delay) {
    if (state.statusTimer !== null) window.clearTimeout(state.statusTimer);
    state.statusTimer = window.setTimeout(function () {
      state.statusTimer = null;
      checkStatus();
    }, delay);
  }

  async function checkStatus() {
    try {
      var payload = await requestJson(API.status);
      setEngineStatus(payload.engine || {});
      if (!state.engineReady && (!payload.engine || payload.engine.state !== "failed")) scheduleStatus(1100);
      else scheduleStatus(7000);
    } catch (_error) {
      setEngineStatus({
        state: "failed",
        ready: false,
        error: {
          message: "The local UI service is unavailable.",
          action: "Start the server and reload this page.",
          command: "sisu-reader ui"
        }
      });
      scheduleStatus(3000);
    }
  }

  async function retryEngine() {
    elements.statusLabel.textContent = "Retrying the private engine…";
    elements.engineAction.hidden = true;
    try {
      var payload = await requestJson(API.retry, { method: "POST", body: {} });
      setEngineStatus(payload.engine || {});
      scheduleStatus(700);
    } catch (error) {
      toast(errorText(error));
      checkStatus();
    }
  }

  async function copyCommand() {
    var command = elements.engineCommand.textContent;
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      toast("Command copied.");
    } catch (_error) {
      var selection = window.getSelection();
      var range = document.createRange();
      range.selectNodeContents(elements.engineCommand);
      selection.removeAllRanges();
      selection.addRange(range);
      toast("Command selected. Press Ctrl+C to copy it.");
    }
  }

  function addUserMessage(text) {
    elements.welcome.hidden = true;
    var article = node("article", "message user");
    article.appendChild(node("p", "message-label", "You"));
    article.appendChild(node("div", "bubble", text));
    elements.messages.appendChild(article);
    scrollLatest();
  }

  function addError(error) {
    elements.welcome.hidden = true;
    var article = node("article", "message assistant");
    article.appendChild(node("p", "message-label", "SISU Reader"));
    var box = node("div", "notice error", errorText(error));
    article.appendChild(box);
    elements.messages.appendChild(article);
    scrollLatest();
  }

  function addWorking() {
    var article = node("article", "message assistant working");
    article.appendChild(node("p", "message-label", "SISU Reader"));
    var bubble = node("div", "bubble");
    var title = node("div", "working-title", "Researching locally");
    var dots = node("span", "dots");
    dots.append(node("i"), node("i"), node("i"));
    title.appendChild(dots);
    var stage = node("div", "working-stage", "Waiting for the local worker…");
    bubble.append(title, stage);
    article.appendChild(bubble);
    elements.messages.appendChild(article);
    scrollLatest();
    var started = performance.now();
    var timer = window.setInterval(function () {
      var seconds = (performance.now() - started) / 1000;
      var base = stage.dataset.label || "Working locally";
      stage.textContent = base + " · " + formatDuration(seconds);
    }, 1000);
    return {
      element: article,
      update: function (job) {
        var current = String(job.stage || "");
        var label = STAGE_LABELS[current] || (job.status === "queued" ? "Waiting for the local worker" : "Working locally");
        if (job.status === "queued" && typeof job.queue_position === "number") label += " · queue position " + job.queue_position;
        if (job.progress && job.progress.message) label += " · " + String(job.progress.message);
        stage.dataset.label = label;
        stage.textContent = label + " · " + formatDuration(Number(job.elapsed_s || 0));
        renderPipeline(current, String(job.status || "running"));
      },
      stop: function () { window.clearInterval(timer); }
    };
  }

  function normalizeAnswer(raw) {
    var answer = raw && typeof raw === "object" ? raw : {};
    return {
      status: String(answer.status || "error"),
      text: String(answer.text || ""),
      sources: Array.isArray(answer.sources) ? answer.sources : [],
      warnings: Array.isArray(answer.warnings) ? answer.warnings : [],
      timings: answer.timings && typeof answer.timings === "object" ? answer.timings : {},
      coverage: answer.coverage && typeof answer.coverage === "object" ? answer.coverage : {},
      tracePath: String(answer.trace_path || ""),
      traceAvailable: Boolean(answer.trace_available),
      debug: answer.debug && typeof answer.debug === "object" ? answer.debug : {}
    };
  }

  function answerElapsed(answer, job) {
    if (typeof answer.timings.total_s === "number") return answer.timings.total_s;
    if (typeof answer.timings.answer_ready_s === "number") return answer.timings.answer_ready_s;
    return job && typeof job.elapsed_s === "number" ? job.elapsed_s : null;
  }

  function showAnswerDetails(answer, job, tab) {
    state.lastAnswer = answer;
    state.lastJob = job && job.id ? String(job.id) : "";
    renderSources(answer.sources);
    renderCoverage(answer.coverage);
    renderRun(answer, job || {});
    openInspector(tab || "sources");
  }

  function addAssistantMessage(answer, job) {
    var article = node("article", "message assistant");
    article.appendChild(node("p", "message-label", "SISU"));
    var bubble = node("div", "bubble answer-bubble");
    var sourceIds = {};
    answer.sources.forEach(function (source) { sourceIds[String(source.id || "")] = true; });
    bubble.appendChild(renderAnswerText(answer.text, function (sourceId) {
      showAnswerDetails(answer, job, "sources");
      openSource(sourceId);
    }, sourceIds));
    article.appendChild(bubble);
    var actions = node("div", "answer-actions");
    if (answer.status === "partial") actions.appendChild(node("span", "answer-chip partial", "Partial"));
    if (answer.warnings.some(importantWarning)) actions.appendChild(node("span", "answer-chip caution", "Check details"));
    if (answer.sources.length) {
      var documentNames = {};
      answer.sources.forEach(function (source) {
        documentNames[String(source.title || source.document_id || "source")] = true;
      });
      var documentCount = Object.keys(documentNames).length;
      var sourceLabel = answer.sources.length + (answer.sources.length === 1 ? " citation" : " citations");
      if (documentCount > 1) sourceLabel += " · " + documentCount + " docs";
      var sourceButton = node("button", "detail-button source-button", sourceLabel);
      sourceButton.type = "button";
      sourceButton.addEventListener("click", function () { showAnswerDetails(answer, job, "sources"); });
      actions.appendChild(sourceButton);
    }
    var detailButton = node("button", "detail-button quiet", "Details");
    detailButton.type = "button";
    detailButton.addEventListener("click", function () {
      var tab = answer.warnings.some(importantWarning)
        ? "run"
        : (Object.keys(answer.coverage).length ? "coverage" : "run");
      showAnswerDetails(answer, job, tab);
    });
    actions.appendChild(detailButton);
    var elapsed = answerElapsed(answer, job);
    if (elapsed !== null) actions.appendChild(node("span", "answer-time", formatDuration(elapsed)));
    article.appendChild(actions);
    elements.messages.appendChild(article);
    state.lastAnswer = answer;
    state.lastJob = job && job.id ? String(job.id) : "";
    renderSources(answer.sources);
    renderCoverage(answer.coverage);
    renderRun(answer, job || {});
    renderPipeline("binding_citations", "complete");
    scrollLatest();
  }

  function renderSources(sources) {
    elements.sourceList.replaceChildren();
    elements.sourceCount.textContent = String(sources.length);
    elements.sourcesEmpty.hidden = sources.length > 0;
    sources.forEach(function (source, index) {
      var sourceId = String(source.id || "S" + (index + 1));
      var card = node("details", "source-card");
      card.dataset.sourceId = sourceId;
      var summary = node("summary");
      summary.appendChild(node("span", "source-number", sourceId.replace(/^S/, "")));
      var label = node("span", "source-summary");
      label.appendChild(node("strong", "", String(source.title || "Untitled source")));
      if (source.locator) label.appendChild(node("small", "", String(source.locator)));
      summary.appendChild(label);
      card.appendChild(summary);
      card.appendChild(node("blockquote", "source-quote", String(source.quote || "")));
      if (source.provenance) {
        card.appendChild(node("small", "source-provenance", String(source.provenance)));
      }
      if (source.uri) {
        if (/^https:\/\//i.test(String(source.uri))) {
          var link = node("a", "detail-button source-link", "Open video");
          link.href = String(source.uri);
          link.target = "_blank";
          link.rel = "noreferrer noopener";
          card.appendChild(link);
        } else {
          var pathDetails = node("details", "compact-details");
          pathDetails.appendChild(node("summary", "", "Local video path"));
          pathDetails.appendChild(node("code", "", String(source.uri)));
          card.appendChild(pathDetails);
        }
      }
      elements.sourceList.appendChild(card);
    });
  }

  function valueCount(value) {
    if (Array.isArray(value)) return value.length;
    if (typeof value === "number") return value;
    if (value && typeof value === "object") return Object.keys(value).length;
    return null;
  }

  function coverageValue(coverage, names) {
    for (var i = 0; i < names.length; i += 1) {
      if (coverage[names[i]] !== undefined) return coverage[names[i]];
    }
    return undefined;
  }

  function renderCoverage(coverage) {
    elements.coverageContent.replaceChildren();
    var keys = Object.keys(coverage || {});
    elements.coverageEmpty.hidden = keys.length > 0;
    if (!keys.length) return;
    var card = node("section", "coverage-card");
    card.appendChild(node("h3", "", "Coverage"));
    var grid = node("div", "coverage-grid");
    [
      ["Screened", coverageValue(coverage, ["manifests_screened", "documents_screened"])],
      ["Opened", coverageValue(coverage, ["documents_read", "documents_read_complete"])],
      ["Remaining", coverageValue(coverage, ["documents_remaining", "deferred_documents", "deferred"])],
      ["Evidence", coverageValue(coverage, ["evidence_cards"])]
    ].forEach(function (pair) {
      var count = valueCount(pair[1]);
      if (count === null) return;
      var stat = node("div", "coverage-stat");
      stat.append(node("strong", "", count), node("small", "", pair[0]));
      grid.appendChild(stat);
    });
    if (grid.childNodes.length) card.appendChild(grid);
    if (coverage.reference_checks_enabled) {
      var referenceNote = coverage.reference_incomplete
        ? "Some referenced conditions remain unresolved. Affected conclusions are provisional."
        : "Detected explicit references were checked in the supplied evidence.";
      card.appendChild(node("p", "", referenceNote));
    }
    var notes = [];
    ["incomplete_reasons", "extraction_gaps", "document_errors"].forEach(function (key) {
      var values = coverage[key];
      if (Array.isArray(values)) values.forEach(function (value) { if (value) notes.push(String(value)); });
    });
    if (notes.length) {
      var details = node("details", "compact-details");
      details.appendChild(node("summary", "", notes.length + (notes.length === 1 ? " note" : " notes")));
      var noteList = node("ul", "detail-notes");
      notes.forEach(function (value) { noteList.appendChild(node("li", "", value)); });
      details.appendChild(noteList);
      card.appendChild(details);
    }
    elements.coverageContent.appendChild(card);
  }

  function renderRun(answer, job) {
    state.traceToken += 1;
    elements.runContent.replaceChildren();
    elements.runEmpty.hidden = false;
    var hasDetails = Object.keys(answer.timings).length || answer.warnings.length || answer.traceAvailable || Object.keys(answer.debug).length;
    if (!hasDetails) return;
    elements.runEmpty.hidden = true;
    var summary = node("section", "run-card");
    summary.appendChild(node("h3", "", "Local run"));
    var list = node("dl", "key-values");
    [["Status", answer.status], ["Elapsed", formatDuration(answerElapsed(answer, job))]].forEach(function (pair) {
      var row = node("div");
      row.append(node("dt", "", pair[0]), node("dd", "", pair[1] || "—"));
      list.appendChild(row);
    });
    summary.appendChild(list);
    elements.runContent.appendChild(summary);

    var traceJobId = job && job.id ? String(job.id) : "";
    if (answer.traceAvailable) {
      var trace = node("section", "run-card");
      trace.appendChild(node("h3", "", "Private trace"));
      if (answer.traceAvailable && traceJobId) {
        var button = node("button", "detail-button", "Open trace");
        button.type = "button";
        button.addEventListener("click", function () { loadTrace(traceJobId, trace); });
        trace.appendChild(button);
      }
      elements.runContent.appendChild(trace);
    }
    if (answer.warnings.length) {
      var notes = node("section", "run-card");
      notes.appendChild(node("h3", "", "Notes"));
      var warnings = node("ul", "detail-notes");
      answer.warnings.forEach(function (warning) { warnings.appendChild(node("li", "", warning)); });
      notes.appendChild(warnings);
      elements.runContent.appendChild(notes);
    }
    if (Object.keys(answer.timings).length || Object.keys(answer.debug).length) {
      var advanced = node("details", "run-card compact-details");
      advanced.appendChild(node("summary", "", "Technical details"));
      var payload = {};
      if (Object.keys(answer.timings).length) payload.timings = answer.timings;
      if (Object.keys(answer.debug).length) payload.debug = answer.debug;
      advanced.appendChild(node("pre", "", display(payload)));
      elements.runContent.appendChild(advanced);
    }
  }

  function addRunObject(title, value) {
    var card = node("section", "run-card");
    card.appendChild(node("h3", "", title));
    card.appendChild(node("pre", "", display(value)));
    elements.runContent.appendChild(card);
  }

  async function loadTrace(jobId, card) {
    var token = state.traceToken + 1;
    state.traceToken = token;
    var loading = node("div", "notice", "Opening the saved local trace…");
    card.appendChild(loading);
    try {
      var payload = await requestJson(API.job + encodeURIComponent(jobId) + "/trace");
      if (token !== state.traceToken) return;
      loading.remove();
      var pre = node("pre", "", display(payload.trace || {}));
      card.appendChild(pre);
    } catch (error) {
      if (token !== state.traceToken) return;
      loading.className = "notice error";
      loading.textContent = errorText(error);
    }
  }

  function activateTab(name) {
    ["sources", "coverage", "run"].forEach(function (tab) {
      var selected = tab === name;
      elements[tab + "Tab"].setAttribute("aria-selected", String(selected));
      elements[tab + "Panel"].hidden = !selected;
    });
  }

  function openInspector(tab) {
    if (!elements.inspector.classList.contains("open")) {
      state.inspectorReturnFocus = document.activeElement;
    }
    elements.inspector.removeAttribute("inert");
    elements.inspector.setAttribute("aria-hidden", "false");
    activateTab(tab || "sources");
    elements.inspector.classList.add("open");
    elements.backdrop.hidden = !window.matchMedia("(max-width: 1180px)").matches;
  }

  function closeInspector() {
    var wasOpen = elements.inspector.classList.contains("open");
    elements.inspector.classList.remove("open");
    elements.inspector.setAttribute("aria-hidden", "true");
    elements.inspector.setAttribute("inert", "");
    elements.backdrop.hidden = true;
    if (
      wasOpen
      && state.inspectorReturnFocus
      && state.inspectorReturnFocus.isConnected
      && typeof state.inspectorReturnFocus.focus === "function"
    ) {
      state.inspectorReturnFocus.focus();
    }
    state.inspectorReturnFocus = null;
  }

  function scrollLatest() {
    window.requestAnimationFrame(function () {
      elements.conversation.scrollTop = elements.conversation.scrollHeight;
    });
  }

  function delay(milliseconds) {
    return new Promise(function (resolve) { window.setTimeout(resolve, milliseconds); });
  }

  async function pollJob(jobId, working) {
    var interval = 450;
    var transientFailures = 0;
    while (state.activeJob === jobId) {
      await delay(interval);
      try {
        var payload = await requestJson(API.job + encodeURIComponent(jobId));
        transientFailures = 0;
        var job = payload.job && typeof payload.job === "object" ? payload.job : null;
        if (!job) throw new ApiError("The service returned an invalid job.", 500, "invalid_job");
        working.update(job);
        if (job.status === "complete") return job;
        if (job.status === "failed") {
          var failure = job.error && typeof job.error === "object" ? job.error : {};
          throw new ApiError(failure.message || "The local answer failed.", 500, failure.code, failure.action, failure.command);
        }
        interval = Math.min(interval + 80, 1250);
      } catch (error) {
        if (error instanceof ApiError && error.code && error.code !== "network_error") throw error;
        transientFailures += 1;
        if (transientFailures >= 4) throw error;
        interval = Math.min(interval + 350, 1900);
      }
    }
    throw new ApiError("The local request was interrupted.", 0, "interrupted");
  }

  async function startJob(message, request, allowSessionRetry) {
    try {
      return await requestJson(API.ask, { method: "POST", body: { message: message, request_id: request } });
    } catch (error) {
      if (allowSessionRetry && error instanceof ApiError && error.code === "session_expired") {
        await ensureSession(true);
        return startJob(message, request, false);
      }
      throw error;
    }
  }

  async function submitQuestion(raw) {
    var message = String(raw || "").trim();
    if (!message || state.busy) return;
    if (!state.engineReady) {
      var problem = state.engineError || { message: "The private engine is still starting.", action: "Wait a moment and retry." };
      toast(errorText(problem));
      elements.input.focus();
      checkStatus();
      return;
    }
    setBusy(true);
    var working = null;
    try {
      await ensureSession(false);
      var payload = await startJob(message, requestId(), true);
      var job = payload.job && typeof payload.job === "object" ? payload.job : null;
      if (!job || !job.id) throw new ApiError("The local service returned an invalid job.", 500, "invalid_job");
      elements.input.value = "";
      autosize();
      addUserMessage(message);
      working = addWorking();
      working.update(job);
      state.activeJob = String(job.id);
      if (job.status !== "complete") job = await pollJob(state.activeJob, working);
      working.stop();
      working.element.remove();
      working = null;
      addAssistantMessage(normalizeAnswer(job.answer), job);
    } catch (error) {
      if (working) {
        working.stop();
        working.element.remove();
      }
      addError(error);
      checkStatus();
    } finally {
      state.activeJob = "";
      setBusy(false);
      elements.input.focus();
    }
  }

  async function newConversation() {
    if (state.busy) return;
    try {
      await ensureSession(true);
      state.lastAnswer = null;
      state.lastJob = "";
      elements.messages.replaceChildren();
      elements.welcome.hidden = false;
      renderSources([]);
      renderCoverage({});
      elements.runContent.replaceChildren();
      elements.runEmpty.hidden = false;
      renderPipeline("", "idle");
      closeInspector();
      elements.input.focus();
    } catch (error) {
      toast(errorText(error));
    }
  }

  elements.form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitQuestion(elements.input.value);
  });
  elements.input.addEventListener("input", autosize);
  elements.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });
  elements.newConversation.addEventListener("click", newConversation);
  elements.mobileNew.addEventListener("click", newConversation);
  elements.mobileDetails.addEventListener("click", function () { openInspector("sources"); });
  elements.retryEngine.addEventListener("click", retryEngine);
  elements.bannerRetry.addEventListener("click", retryEngine);
  elements.copyCommand.addEventListener("click", copyCommand);
  elements.sourcesTab.addEventListener("click", function () { activateTab("sources"); });
  elements.coverageTab.addEventListener("click", function () { activateTab("coverage"); });
  elements.runTab.addEventListener("click", function () { activateTab("run"); });
  elements.inspectorClose.addEventListener("click", closeInspector);
  elements.backdrop.addEventListener("click", closeInspector);
  document.querySelectorAll(".suggestions button").forEach(function (button) {
    button.addEventListener("click", function () {
      var question = button.dataset.question || button.textContent;
      elements.input.value = question;
      autosize();
      submitQuestion(question);
    });
  });
  window.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeInspector();
  });

  var stored = window.sessionStorage.getItem(SESSION_KEY) || "";
  saveSession(stored);
  if (!state.session) createSession().catch(function (error) { toast(errorText(error)); });
  renderSources([]);
  renderCoverage({});
  renderPipeline("", "idle");
  autosize();
  setBusy(false);
  checkStatus();
})();
