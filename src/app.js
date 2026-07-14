// Aria UI — vanilla JS, no framework. Talks to the Python sidecar HTTP API.
//
// Port discovery: in a bundled/dev Tauri build the Rust shell spawns the
// sidecar on an OS-picked port and exposes it via the `sidecar_port` command
// (and a `sidecar-ready` event). We resolve that before the first fetch. When
// running the UI in a plain browser (no Tauri), we fall back to the fixed dev
// port 8765 so `npm run dev` + `python app.py --port 8765` just works.
let API = (window.__ARIA_API__ || "http://127.0.0.1:8765");
// Resolves once the real sidecar port is known (Tauri only) — api() awaits
// this before every call so an early click can't race ahead of it. See boot().
let sidecarReadyPromise = null;
// True once we've ever successfully resolved the real sidecar port. Lets
// refreshStatus() tell "still starting up" (not confirmed yet) apart from
// "was working, now isn't" (confirmed, then a call failed) — the former
// should read as a calm "Starting…", not an alarming "Offline".
let sidecarConfirmed = false;
// Keep in lockstep with python-sidecar/app.py's APP_VERSION and
// src-tauri/tauri.conf.json's "version" — shown when update checks are
// disabled (e.g. a dev build with no bundled .update_token).
const APP_VERSION_FALLBACK = "0.1.0";

// ---- tiny helpers --------------------------------------------------------
const $ = (sel, el = document) => el.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    n.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return n;
};
// Explicit column widths (percentages summing to 100) so settings tables
// never grow wider than their container — with table-layout:auto, a long
// unbreakable string (e.g. an HF repo path) could force the whole table
// wider than the panel, silently pushing the last column(s) — often the
// action button — out of view with no way to reach them.
const colgroup = (...widths) => el("colgroup", {},
  ...widths.map(w => el("col", { style: `width:${w}%` })));
const esc = (s) => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
// Assistant replies render as real Markdown (lists, bold, fenced code, etc.)
// instead of literal asterisks/backticks. DOMPurify sanitizes marked's HTML
// output before it ever reaches innerHTML — model output can be indirectly
// steered by untrusted content (web search results, tool output) fed back
// into the prompt, so this isn't optional hardening.
const renderMarkdown = (text) => DOMPurify.sanitize(marked.parse(text ?? ""));
const fmtTime = (ms) => ms ? new Date(ms).toLocaleString() : "—";

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

async function api(path, method = "GET", body = null) {
  if (sidecarReadyPromise) await sidecarReadyPromise;
  try {
    const res = await fetch(API + path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : (method === "POST" ? "{}" : null),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  } catch (e) {
    toast("⚠ " + e.message);
    throw e;
  }
}

// ---- status ---------------------------------------------------------------
let lastStatus = null;
let wasBackendDown = false;
async function refreshStatus() {
  try {
    const s = await api("/status");
    lastStatus = s;
    $("#st-engine").textContent = s.engine || "—";
    $("#st-model").textContent = s.model || "not loaded";
    $("#st-adapter").textContent = s.active_adapter || "base";
    $("#st-device").textContent = (s.capabilities && s.capabilities.device) || "—";
    $("#status-dot").classList.remove("offline", "pending");
    $("#status-text").textContent = s.loaded || (s.engine || "").toLowerCase().includes("fake")
      ? "Ready" : "Online";
    // The backend just came back (fresh launch finishing its cold start, or
    // main.rs's auto-restart recovering from a crash) — a Settings panel
    // opened during the outage would otherwise sit frozen on "Load failed"
    // / "Couldn't check for updates" forever, since nothing else prompts it
    // to look again once the user has already stopped clicking things.
    if (wasBackendDown && $("#settings-overlay").classList.contains("open")) {
      switchSettingsView(currentSettingsView);
      checkForUpdates();
    }
    wasBackendDown = false;
    return s;
  } catch {
    lastStatus = null;
    wasBackendDown = true;
    if (sidecarConfirmed) {
      // We've talked to it successfully before — this failure means it's
      // genuinely gone (crashed, quit), not just still starting up.
      $("#status-dot").classList.remove("pending");
      $("#status-dot").classList.add("offline");
      $("#status-text").textContent = "Offline";
    } else {
      // Never confirmed yet — a frozen PyInstaller build can take well over
      // the polling window to finish self-extracting on a cold launch.
      // That's normal, not broken; don't show an alarming "Offline" for it,
      // but don't flash a false-positive green "healthy" dot either.
      $("#status-dot").classList.remove("offline");
      $("#status-dot").classList.add("pending");
      $("#status-text").textContent = "Starting…";
    }
    return null;
  }
}

// ==========================================================================
// VOICE PLAYBACK — gapless, barge-in capable
// ==========================================================================
let speakEnabled = false;
let speakingToken = 0;
let currentAudio = null;

function stopSpeaking() {
  speakingToken++;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  api("/voice/stop", "POST").catch(() => {});
  $("#stop-btn").style.display = "none";
}

function playSegment(seg) {
  return new Promise(resolve => {
    const mime = seg.format === "aiff" ? "audio/aiff" : "audio/wav";
    const a = new Audio("data:" + mime + ";base64," + seg.audio_b64);
    currentAudio = a;
    a.onended = resolve;
    a.onerror = resolve;
    a.play().catch(resolve);
  });
}

async function speakSegments(segments) {
  const token = ++speakingToken;
  $("#stop-btn").style.display = "";
  for (const seg of segments) {
    if (token !== speakingToken) return;
    await playSegment(seg);
  }
  if (token === speakingToken) $("#stop-btn").style.display = "none";
}

// ==========================================================================
// VOICE INPUT (mic button) — offline dictation, no cloud speech APIs
// ==========================================================================
// Chrome's MediaRecorder only emits webm/opus, and decoding that server-side
// would need a system ffmpeg install (mlx_audio shells out to it). Recording
// raw PCM and encoding a WAV file ourselves avoids that dependency entirely
// — miniaudio (mlx_audio's default reader) reads WAV natively.
let sttAvailable = false;
let micRecording = false;
let micStream = null, micCtx = null, micNode = null, micChunks = [];

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function downsampleTo16k(buffer, inRate) {
  const outRate = 16000;
  if (inRate === outRate) return buffer;
  const ratio = inRate / outRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    result[i] = buffer[Math.min(buffer.length - 1, Math.round(i * ratio))];
  }
  return result;
}

function encodeWav16k(float32) {
  const pcm = floatTo16BitPCM(float32);
  const rate = 16000;
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(8, "WAVE"); writeStr(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, pcm.length * 2, true);
  new Int16Array(buffer, 44).set(pcm);
  return buffer;
}

function bufToBase64(buf) {
  let binary = "";
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = micCtx.createMediaStreamSource(micStream);
  micNode = micCtx.createScriptProcessor(4096, 1, 1);
  micChunks = [];
  micNode.onaudioprocess = (e) => micChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(micNode);
  micNode.connect(micCtx.destination);
}

function stopMicAndEncode() {
  if (micNode) { micNode.disconnect(); micNode.onaudioprocess = null; }
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  const rate = micCtx ? micCtx.sampleRate : 44100;
  const total = micChunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of micChunks) { merged.set(c, off); off += c.length; }
  if (micCtx) micCtx.close();
  micChunks = [];
  return encodeWav16k(downsampleTo16k(merged, rate));
}

async function toggleMic() {
  const micBtn = $("#mic-btn");
  if (!micRecording) {
    try {
      await startMic();
      micRecording = true;
      micBtn.classList.add("recording");
      micBtn.title = micBtn.ariaLabel = "Stop recording";
    } catch {
      toast("Microphone access denied");
    }
    return;
  }
  micRecording = false;
  micBtn.classList.remove("recording");
  micBtn.title = micBtn.ariaLabel = "Voice input";
  const wavBuf = stopMicAndEncode();

  const ta = $("#composer-input");
  const prevPlaceholder = ta.placeholder;
  ta.placeholder = "Transcribing…";
  ta.disabled = true;
  try {
    const r = await api("/transcribe", "POST", { audio_b64: bufToBase64(wavBuf), mime: "audio/wav" });
    if (r.ok && r.text) {
      ta.value = (ta.value ? ta.value.trim() + " " : "") + r.text;
      autosize(ta);
    } else if (r.error) {
      toast(r.error);
    }
  } catch { /* api() already toasts the error */ }
  ta.disabled = false;
  ta.placeholder = prevPlaceholder;
  ta.focus();
}

async function refreshSttAvailability() {
  try {
    const s = await api("/stt");
    sttAvailable = !!s.available;
  } catch {
    sttAvailable = false;
  }
  const micBtn = $("#mic-btn");
  micBtn.disabled = !sttAvailable;
  micBtn.title = sttAvailable ? "Voice input" : "Voice input needs Apple Silicon (mlx-audio)";
}

// ==========================================================================
// ONBOARDING
// ==========================================================================
const OB_STEPS = ["welcome", "profile", "choose", "downloading", "ready"];
let chosenModel = "gemma-4-12b";

function obShow(step) {
  document.querySelectorAll(".ob-step").forEach(s => s.classList.toggle("active", s.dataset.step === step));
  document.querySelectorAll(".ob-dot").forEach(d => d.classList.toggle("active", d.dataset.dot === step));
  const card = document.querySelector(".ob-card");
  if (card) card.scrollTop = 0;
}

async function populateChoices() {
  const box = $("#ob-choices");
  box.innerHTML = "";
  let catalog = {
    "gemma-4-12b": { size_gb: 7.0, blurb: "Best quality — recommended for most Macs" },
    "gemma-4-e4b": { size_gb: 3.0, blurb: "Smaller — learns your style faster" },
    "gemma-4-e2b": { size_gb: 1.8, blurb: "Smallest & fastest" },
  };
  try {
    const m = await api("/models");
    catalog = Object.fromEntries(Object.entries(m.catalog).map(([id, info]) => [id, {
      size_gb: info.size_gb,
      blurb: id === "gemma-4-12b" ? "Best quality — recommended for most Macs"
        : id === "gemma-4-e4b" ? "Smaller — learns your style faster"
        : "Smallest & fastest",
    }]));
  } catch { /* sidecar not up yet — use defaults, download will retry */ }

  const niceName = { "gemma-4-12b": "Balanced (Recommended)", "gemma-4-e4b": "Personalizable", "gemma-4-e2b": "Lightweight" };
  const order = ["gemma-4-12b", "gemma-4-e4b", "gemma-4-e2b"];
  const entries = Object.entries(catalog).sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  entries.forEach(([id, info]) => {
    const row = el("div", { class: "ob-choice" + (id === chosenModel ? " selected" : "") },
      el("div", { class: "radio" }),
      el("div", { class: "meta" }, el("b", {}, niceName[id] || id), el("span", {}, info.blurb)),
      el("div", { class: "tag" }, info.size_gb + " GB"));
    row.addEventListener("click", () => {
      chosenModel = id;
      box.querySelectorAll(".ob-choice").forEach(c => c.classList.remove("selected"));
      row.classList.add("selected");
    });
    box.appendChild(row);
  });
}

const RING_CIRC = 2 * Math.PI * 42;
function setRingPercent(pct) {
  const ring = $("#ob-ring-progress");
  const clamped = Math.min(100, Math.max(0, pct));
  ring.style.strokeDashoffset = String(RING_CIRC * (1 - clamped / 100));
}
function fmtBytes(n) {
  if (!n) return "0 MB";
  const gb = n / 1024 ** 3;
  return gb >= 0.1 ? gb.toFixed(2) + " GB" : (n / 1024 ** 2).toFixed(0) + " MB";
}

// Polls real download progress (bytes on disk vs. repo total) until the
// backend reports done/error. Falls back to an indeterminate spinner + raw
// byte count when the total size isn't known yet (e.g. HF metadata lookup
// hasn't resolved).
function pollDownloadProgress(modelId) {
  const spin = $("#ob-ring-spin");
  const pctLabel = $("#ob-ring-pct");
  const bytesLabel = $("#ob-bytes");
  const statusLine = $("#ob-status-line");
  spin.classList.remove("hidden");
  pctLabel.textContent = "";
  setRingPercent(0);

  return new Promise((resolve, reject) => {
    const tick = async () => {
      let p;
      try {
        p = await api(`/models/download/progress?model_id=${encodeURIComponent(modelId)}`);
      } catch (e) { reject(e); return; }
      if (p.status === "loading") {
        spin.classList.remove("hidden");
        setRingPercent(100); pctLabel.textContent = "";
        statusLine.textContent = "Loading Aria into memory…";
        bytesLabel.textContent = "Almost there…";
      } else if (p.total_bytes > 0) {
        spin.classList.add("hidden");
        setRingPercent(p.percent || 0);
        pctLabel.textContent = Math.round(p.percent || 0) + "%";
        bytesLabel.textContent = `${fmtBytes(p.downloaded_bytes)} of ${fmtBytes(p.total_bytes)}`;
        statusLine.textContent = "Downloading Aria's model…";
      } else if (p.downloaded_bytes > 0) {
        bytesLabel.textContent = `${fmtBytes(p.downloaded_bytes)} downloaded…`;
      }
      if (p.status === "done") {
        spin.classList.add("hidden");
        setRingPercent(100); pctLabel.textContent = "100%";
        statusLine.textContent = "Ready!";
        resolve(); return;
      }
      if (p.status === "error") { reject(new Error(p.error || "Download failed")); return; }
      setTimeout(tick, 700);
    };
    tick();
  });
}

async function startDownload() {
  obShow("downloading");
  $("#ob-error-slot").innerHTML = "";
  $("#ob-bytes").textContent = "";
  $("#ob-status-line").textContent = "Starting download…";
  try {
    const r = await api("/models/download", "POST", { model_id: chosenModel });
    if (!r.ok) throw new Error(r.error || "Download failed");
    await pollDownloadProgress(chosenModel);
    await refreshStatus();
    finishOnboardingStep("ready");
  } catch (e) {
    $("#ob-error-slot").innerHTML = "";
    $("#ob-error-slot").appendChild(el("div", { class: "ob-error" },
      "Couldn't finish setup: " + e.message));
    const retry = el("button", { class: "ob-primary", style: "margin-top:14px" }, "Try again");
    retry.addEventListener("click", startDownload);
    const cont = el("button", { class: "ob-link" }, "Continue without downloading");
    cont.addEventListener("click", () => finishOnboardingStep("ready"));
    $("#ob-error-slot").append(retry, cont);
  }
}

function finishOnboardingStep(step) { obShow(step); }

// Each non-empty field becomes its own memory chunk (source "profile") so
// RAG retrieval can surface just the relevant fact instead of one big blob.
async function saveProfile() {
  const facts = [
    [$("#ob-name").value.trim(), n => `The user's name is ${n}.`],
    [$("#ob-country").value.trim(), c => `The user is based in ${c}.`],
    [$("#ob-job").value.trim(), j => `The user's job/role is ${j}.`],
    [$("#ob-prefs").value.trim(), p => `User preferences for how Aria should respond: ${p}`],
  ].filter(([v]) => v).map(([v, fmt]) => fmt(v));

  for (const text of facts) {
    try { await api("/memory", "POST", { text, source: "profile" }); } catch { /* best effort */ }
  }
  obShowAsync("choose");
}

function completeOnboarding() {
  localStorage.setItem("aria_onboarded", "1");
  $("#onboarding").classList.add("hidden");
  $("#app").classList.remove("hidden");
  refreshStatus();
  renderChat();
}

function wireOnboarding() {
  document.querySelectorAll("[data-next]").forEach(b =>
    b.addEventListener("click", () => obShowAsync(b.dataset.next)));
  $('[data-action="save-profile"]').addEventListener("click", saveProfile);
  $('[data-action="skip-profile"]').addEventListener("click", () => obShowAsync("choose"));
  $('[data-action="start-download"]').addEventListener("click", startDownload);
  $('[data-action="skip-download"]').addEventListener("click", () => finishOnboardingStep("ready"));
  $('[data-action="finish"]').addEventListener("click", completeOnboarding);
}
function obShowAsync(step) {
  obShow(step);
  if (step === "choose") populateChoices();
}

async function maybeSkipOnboarding() {
  const onboarded = localStorage.getItem("aria_onboarded") === "1";
  const s = await refreshStatus();
  const isFake = s && (s.engine || "").toLowerCase().includes("fake");
  if (onboarded || (s && (s.loaded || isFake))) {
    completeOnboarding();
    return true;
  }
  obShow("welcome");
  return false;
}

// ==========================================================================
// CHAT — the whole main-window experience
// ==========================================================================
let chatHistory = [];
let currentSessionId = null;

// Quick-save is deliberately available on every message, not just
// assistant replies with auto-detected facts — the heuristic auto-saver
// (extract_auto_facts) only catches a handful of self-disclosure patterns,
// so anything else the user wants remembered (an assistant's explanation, a
// fact buried in a longer message) needs an explicit, one-click way in.
function addSaveToMemoryButton(bubble, text) {
  const btn = el("button", { class: "msg-save-btn", title: "Save to memory" },
    el("svg", { viewBox: "0 0 20 20", fill: "none" },
      el("path", { d: "M5 3.5h10a1 1 0 0 1 1 1V17l-6-3.2L4 17V4.5a1 1 0 0 1 1-1z", stroke: "currentColor", "stroke-width": "1.5", "stroke-linejoin": "round" })));
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await api("/memory", "POST", { text, source: "chat" });
      toast("Saved to memory");
      btn.classList.add("saved");
    } catch { btn.disabled = false; }
  });
  bubble.appendChild(btn);
}

function renderMsg(m) {
  const d = m.role === "assistant"
    ? el("div", { class: "msg " + m.role, html: renderMarkdown(m.content) })
    : el("div", { class: "msg " + m.role }, m.content);
  if (m.role === "assistant" && m.used_context)
    d.appendChild(el("div", { class: "ctx" }, "✓ used your memory as context"));
  if (m.role === "assistant" && m.auto_saved && m.auto_saved.length)
    m.auto_saved.forEach(fact =>
      d.appendChild(el("div", { class: "ctx" }, `✓ saved to memory: ${fact}`)));
  if (m.role === "assistant" && m.used_skills && m.used_skills.length)
    d.appendChild(el("div", { class: "ctx" }, `✓ used skill: ${m.used_skills.join(", ")}`));
  if (m.role === "assistant" && m.used_tools && m.used_tools.length)
    d.appendChild(el("div", { class: "ctx" }, `✓ used tool: ${m.used_tools.join(", ")}`));
  if (m.role === "assistant" && m.used_search)
    d.appendChild(renderSearchSources(m.used_search));
  if (m.role === "assistant" && m.image_job)
    renderImageJob(d, m.image_job);
  if (m.content) addSaveToMemoryButton(d, m.content);
  return d;
}

// The <img>'s src points at a different origin (the sidecar's own
// 127.0.0.1:port, not the page's own origin) — plain `<a download>` on a
// cross-origin URL is spec'd to just navigate instead of downloading in
// strict WebKit, which is exactly the "can't download it" gap this fixes:
// fetch the bytes into a blob first, then download *that* (blob: URLs are
// always same-origin for download purposes regardless of where they came
// from).
function renderGeneratedImage(result, prompt) {
  const wrap = el("div", { class: "generated-image-wrap" });
  const img = el("img", { class: "generated-image", src: API + result.url, alt: prompt || "" });
  const dl = el("button", { class: "image-download-btn", title: "Download image" },
    el("svg", { viewBox: "0 0 20 20", fill: "none" },
      el("path", {
        d: "M10 3v9m0 0-3.5-3.5M10 12l3.5-3.5M4 15.5h12",
        stroke: "currentColor", "stroke-width": "1.5",
        "stroke-linecap": "round", "stroke-linejoin": "round",
      })));
  dl.addEventListener("click", async () => {
    dl.disabled = true;
    try {
      const resp = await fetch(API + result.url);
      const blob = await resp.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = blobUrl;
      a.download = result.filename || "aria-image.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(blobUrl);
    } catch {
      toast("Couldn't download image");
    } finally {
      dl.disabled = false;
    }
  });
  wrap.append(img, dl);
  return wrap;
}

// Generation runs in the background server-side (see image_generate() in
// app.py — first use downloads a ~4.3GB model, so this can never block the
// chat request itself). The placeholder polls until it either becomes a
// real <img> or a visible error, same UX shape as the model-download
// progress bar elsewhere in Settings.
function renderImageJob(container, imageJob) {
  if (!imageJob) return;
  if (!imageJob.ok) {
    container.appendChild(el("div", { class: "ctx error" }, "⚠ " + (imageJob.error || "Couldn't start image generation")));
    return;
  }
  // Reopening a past chat replays whatever final state was persisted for
  // this specific message (see chat_history.update_message_image_job) —
  // it must NOT poll /images/progress, which only ever reflects the single
  // most recent job process-wide, not this particular historical one.
  if (imageJob.status === "done" && imageJob.result) {
    container.appendChild(renderGeneratedImage(imageJob.result, imageJob.prompt));
    return;
  }
  if (imageJob.status === "error") {
    container.appendChild(el("div", { class: "ctx error" }, "⚠ " + (imageJob.error || "Image generation failed")));
    return;
  }
  const placeholder = el("div", { class: "generated-image-placeholder" },
    el("span", { class: "spinner" }), " Generating image…");
  container.appendChild(placeholder);
  const tick = async () => {
    let p;
    try { p = await api("/images/progress"); } catch { return; }
    if (p.status === "done" && p.result) {
      placeholder.replaceWith(renderGeneratedImage(p.result, imageJob.prompt));
      return;
    }
    if (p.status === "error") {
      placeholder.className = "ctx error";
      placeholder.textContent = "⚠ " + (p.error || "Image generation failed");
      return;
    }
    setTimeout(tick, 2000);
  };
  tick();
}

function renderSearchSources(search) {
  const wrap = el("div", { class: "ctx search-sources" },
    `✓ searched the web for "${search.query}"`);
  (search.results || []).forEach(r => {
    const link = el("a", { href: "#", title: r.url }, r.title || r.url);
    link.addEventListener("click", e => { e.preventDefault(); openUrl(r.url); });
    wrap.appendChild(el("div", { class: "search-source" }, link));
  });
  return wrap;
}

// Mirrors the onboarding welcome step's icon+bold+description row
// (index.html's `.ob-feature` blocks) for visual consistency with the rest
// of the app's first-run copy.
function emptyChatFeature(iconSvg, title, desc) {
  return el("div", { class: "empty-chat-feature" },
    el("span", { class: "ico", html: iconSvg }),
    el("div", {}, el("b", {}, title), el("span", { class: "desc" }, desc)));
}

function renderChat() {
  const msgs = $("#messages");
  msgs.innerHTML = "";
  if (chatHistory.length === 0) {
    msgs.appendChild(el("div", { class: "empty-chat" },
      el("div", { class: "mark" }, "◆"),
      el("h2", {}, "Hi, I'm Aria"),
      el("p", {}, "Ask me anything. I remember what you tell me, and everything stays on your Mac."),
      el("div", { class: "empty-chat-features" },
        emptyChatFeature(
          '<svg viewBox="0 0 20 20" fill="none"><path d="M10 3.5l1.1 3.4L14.5 8l-3.4 1.1L10 12.5l-1.1-3.4L5.5 8l3.4-1.1L10 3.5z" fill="currentColor"/><path d="M15.5 12l.6 1.9 1.9.6-1.9.6-.6 1.9-.6-1.9-1.9-.6 1.9-.6.6-1.9z" fill="currentColor"/></svg>',
          "Remembers what you tell it", "Say something once, and it's remembered."),
        emptyChatFeature(
          '<svg viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="2.6" stroke="currentColor" stroke-width="1.6"/><path d="M10 3.2v1.6M10 15.2v1.6M16.8 10h-1.6M4.8 10H3.2M14.9 5.1l-1.1 1.1M6.2 13.7l-1.1 1.1M14.9 14.9l-1.1-1.1M6.2 6.3 5.1 5.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
          "Can use tools", "Time, search, and more — you control what's on in Settings."),
        emptyChatFeature(
          '<svg viewBox="0 0 20 20" fill="none"><rect x="4.5" y="9" width="11" height="8" rx="2" stroke="currentColor" stroke-width="1.6"/><path d="M6.5 9V6.5a3.5 3.5 0 0 1 7 0V9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
          "Completely private", "Nothing you say ever leaves this computer."))));
  } else {
    chatHistory.forEach(m => msgs.appendChild(renderMsg(m)));
  }
  msgs.scrollTop = msgs.scrollHeight;

  if (lastStatus && !lastStatus.loaded && !(lastStatus.engine || "").toLowerCase().includes("fake")) {
    const banner = el("div", { class: "firstrun" },
      el("b", {}, "No model loaded yet. "),
      "Open Settings → Models to finish setup.",
      (() => {
        const b = el("button", { class: "btn small" }, "Open Settings");
        b.addEventListener("click", () => openSettings("models"));
        return b;
      })());
    msgs.prepend(banner);
  }
}

async function doSend() {
  const ta = $("#composer-input");
  const text = ta.value.trim();
  if (!text) return;
  ta.value = ""; autosize(ta);
  stopSpeaking();
  const msgs = $("#messages");
  if (msgs.querySelector(".empty-chat")) msgs.innerHTML = "";
  const um = { role: "user", content: text };
  chatHistory.push(um); msgs.appendChild(renderMsg(um));
  msgs.scrollTop = msgs.scrollHeight;
  $("#send-btn").disabled = true;

  if (speakEnabled) {
    // TTS needs the complete reply before it can synthesize audio, so this
    // path stays request/response rather than streamed.
    const thinking = el("div", { class: "msg assistant" }, el("span", { class: "spinner" }));
    msgs.appendChild(thinking); msgs.scrollTop = msgs.scrollHeight;
    try {
      const r = await api("/chat/speak", "POST", {
        messages: chatHistory, use_memory: true, use_tools: true,
        session_id: currentSessionId,
      });
      thinking.remove();
      currentSessionId = r.session_id || currentSessionId;
      const am = { role: "assistant", content: r.content, used_context: r.used_context, auto_saved: r.auto_saved, used_skills: r.used_skills, used_tools: r.used_tools, used_search: r.used_search, image_job: r.image_job };
      chatHistory.push(am); msgs.appendChild(renderMsg(am));
      msgs.scrollTop = msgs.scrollHeight;
      if (r.speech && r.speech.segments && r.speech.segments.length) speakSegments(r.speech.segments);
      refreshSessionList();
    } catch { thinking.remove(); }
    $("#send-btn").disabled = false;
    return;
  }

  // Streamed path — tokens render as they arrive instead of waiting for the
  // whole reply, which matters a lot on a multi-billion-parameter model.
  await streamChatInto(msgs);
  $("#send-btn").disabled = false;
}

async function streamChatInto(msgs) {
  const bubble = el("div", { class: "msg assistant" });
  const spinner = el("span", { class: "spinner" });
  bubble.appendChild(spinner);
  msgs.appendChild(bubble); msgs.scrollTop = msgs.scrollHeight;

  let textAcc = "", usedContext = false, autoSaved = [], usedSkills = [], usedTools = [], usedSearch = null, imageJob = null, gotFirstToken = false;
  try {
    if (sidecarReadyPromise) await sidecarReadyPromise;
    const res = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatHistory, use_memory: true, use_tools: true,
        session_id: currentSessionId,
      }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        if (evt.error) { toast("⚠ " + evt.error); continue; }
        if (evt.delta) {
          if (!gotFirstToken) { spinner.remove(); gotFirstToken = true; }
          textAcc += evt.delta;
          bubble.innerHTML = renderMarkdown(textAcc);
          msgs.scrollTop = msgs.scrollHeight;
        }
        if (evt.done) {
          usedContext = evt.used_context; autoSaved = evt.auto_saved || [];
          usedSkills = evt.used_skills || []; usedTools = evt.used_tools || [];
          usedSearch = evt.used_search || null;
          imageJob = evt.image_job || null;
          currentSessionId = evt.session_id || currentSessionId;
        }
      }
    }
  } catch (e) {
    toast("⚠ " + e.message);
  }

  if (!textAcc) { bubble.remove(); return; }
  bubble.innerHTML = renderMarkdown(textAcc);
  if (usedContext) bubble.appendChild(el("div", { class: "ctx" }, "✓ used your memory as context"));
  autoSaved.forEach(fact => bubble.appendChild(el("div", { class: "ctx" }, `✓ saved to memory: ${fact}`)));
  if (usedSkills.length) bubble.appendChild(el("div", { class: "ctx" }, `✓ used skill: ${usedSkills.join(", ")}`));
  if (usedTools.length) bubble.appendChild(el("div", { class: "ctx" }, `✓ used tool: ${usedTools.join(", ")}`));
  if (usedSearch) bubble.appendChild(renderSearchSources(usedSearch));
  renderImageJob(bubble, imageJob);
  addSaveToMemoryButton(bubble, textAcc);
  chatHistory.push({ role: "assistant", content: textAcc, used_context: usedContext, auto_saved: autoSaved, used_skills: usedSkills, used_tools: usedTools, used_search: usedSearch });
  refreshSessionList();
}

function autosize(ta) {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
}

function wireChat() {
  const ta = $("#composer-input");
  ta.addEventListener("input", () => autosize(ta));
  ta.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(); }
  });
  $("#send-btn").addEventListener("click", doSend);
  $("#stop-btn").addEventListener("click", stopSpeaking);
  $("#mic-btn").addEventListener("click", toggleMic);
  $("#speak-toggle").addEventListener("click", () => {
    speakEnabled = !speakEnabled;
    $("#speak-toggle").classList.toggle("on", speakEnabled);
    if (!speakEnabled) stopSpeaking();
    toast(speakEnabled ? "Aria will speak replies" : "Speaking off");
  });
}

// ==========================================================================
// CHAT HISTORY SIDEBAR — past sessions, persisted server-side (see
// chat_history.py). A session is created lazily on first message, not on
// "New chat" click — clicking New chat just clears the working area so nothing
// empty ever clutters the list.
// ==========================================================================
function relativeTime(ms) {
  const diff = Date.now() - ms;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return new Date(ms).toLocaleDateString();
}

async function refreshSessionList() {
  const list = $("#session-list");
  let sessions;
  try { sessions = await api("/sessions"); } catch { return; }
  list.innerHTML = "";
  if (!sessions.length) {
    list.appendChild(el("div", { class: "empty" }, "No past chats yet."));
    return;
  }
  sessions.forEach(s => {
    const item = el("div", {
      class: "session-item" + (s.id === currentSessionId ? " active" : ""),
      title: `${s.title} — ${relativeTime(s.updated_at)}`,
    });
    const title = el("span", { class: "session-title" }, s.title);
    title.addEventListener("click", () => loadSession(s.id));
    const del = el("button", { class: "session-delete", title: "Delete chat" }, "✕");
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      await api("/sessions/delete", "POST", { session_id: s.id });
      if (s.id === currentSessionId) startNewChat();
      refreshSessionList();
    });
    item.append(title, del);
    list.appendChild(item);
  });
}

async function loadSession(sessionId) {
  if (sessionId === currentSessionId) return;
  let msgs;
  try { msgs = await api("/sessions/messages?session_id=" + encodeURIComponent(sessionId)); }
  catch { toast("Couldn't load that chat"); return; }
  currentSessionId = sessionId;
  chatHistory = msgs.map(m => ({ role: m.role, content: m.content, image_job: m.image_job || null }));
  renderChat();
  refreshSessionList();
}

function startNewChat() {
  currentSessionId = null;
  chatHistory = [];
  renderChat();
  refreshSessionList();
}

function wireHistorySidebar() {
  $("#new-chat-btn").addEventListener("click", startNewChat);
  $("#sidebar-toggle").addEventListener("click", () => {
    $("#history-sidebar").classList.toggle("collapsed");
  });
  refreshSessionList();
}

// ==========================================================================
// SETTINGS SHEET — technical panels (Memory / Tools / Training / Adapters / Models)
// ==========================================================================
const settingsViews = {};

const MEMORY_UPLOAD_EXTS = [".txt", ".md", ".markdown", ".pdf"];

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1] || "");
    reader.onerror = () => reject(reader.error || new Error("couldn't read file"));
    reader.readAsDataURL(file);
  });
}

function buildMemoryUploadCard(root) {
  const card = el("div", { class: "card upload-card" });
  const zone = el("div", { class: "upload-zone" },
    el("div", {}, "Drop a .txt, .md, or .pdf file here, or click to browse"));
  const input = el("input", { type: "file", accept: MEMORY_UPLOAD_EXTS.join(","), style: "display:none" });

  async function handleFile(file) {
    if (!file) return;
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    if (!MEMORY_UPLOAD_EXTS.includes(ext)) {
      toast(`Unsupported file type: ${ext || file.name}`);
      return;
    }
    zone.classList.add("busy");
    zone.textContent = `Reading ${file.name}…`;
    try {
      const content_b64 = await readFileAsBase64(file);
      const r = await api("/memory/upload", "POST", { filename: file.name, content_b64 });
      toast(`Added ${r.chunks_added} chunk(s) from ${file.name}`);
      settingsViews.memory(root);
    } catch {
      zone.classList.remove("busy");
      zone.textContent = "Drop a .txt, .md, or .pdf file here, or click to browse";
    }
  }

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => handleFile(input.files[0]));
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("dragover");
    handleFile(e.dataTransfer.files[0]);
  });

  card.append(el("label", {}, "Upload a file"), zone, input);
  return card;
}

settingsViews.memory = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Memory", "Everything Aria knows about you. Facts live here — instant, editable, private."));

  const add = el("div", { class: "card" });
  const src = el("input", { placeholder: "source (e.g. notes.md)", value: "note" });
  const txt = el("textarea", { placeholder: "Paste a note, fact, or document to remember…" });
  const btn = el("button", { class: "btn" }, "Add to memory");
  btn.addEventListener("click", async () => {
    if (!txt.value.trim()) return;
    const r = await api("/memory", "POST", { text: txt.value, source: src.value || "note" });
    toast(`Added ${r.chunks_added} chunk(s)`); txt.value = "";
    settingsViews.memory(root);
  });
  add.append(el("label", {}, "Add memory"),
    el("div", { class: "grid" }, src, txt, el("div", {}, btn)));
  root.appendChild(add);

  root.appendChild(buildMemoryUploadCard(root));

  const list = await api("/memory");
  const card = el("div", { class: "card" });
  card.appendChild(el("label", {}, `Stored chunks (${list.length})`));
  if (list.length === 0) card.appendChild(el("div", { class: "empty" }, "No memory yet."));
  else {
    const tbl = el("table");
    tbl.appendChild(colgroup(14, 56, 10, 20));
    tbl.appendChild(el("tr", {}, el("th", {}, "Source"), el("th", {}, "Text"),
      el("th", {}, "Used"), el("th", {}, "")));
    list.forEach(c => {
      const del = el("button", { class: "btn danger" }, "Delete");
      del.addEventListener("click", async () => {
        await api("/memory/delete", "POST", { chunk_id: c.id });
        toast("Deleted"); settingsViews.memory(root);
      });
      tbl.appendChild(el("tr", {},
        el("td", {}, el("code", {}, c.source || "—")),
        el("td", { html: esc((c.text || "").slice(0, 140)) }),
        el("td", {}, String(c.use_count ?? 0)),
        el("td", {}, del)));
    });
    card.appendChild(tbl);
  }
  root.appendChild(card);
};

settingsViews.persona = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Persona",
    "The instructions Aria always follows — the instant, reversible way to change tone or behavior. For a permanent change instead, use the Training tab."));

  const p = await api("/persona");
  const card = el("div", { class: "card" });
  const badge = el("span", { class: "pill " + (p.is_custom ? "info" : "good") },
    p.is_custom ? "custom" : "default");
  const prompt = el("textarea", { style: "min-height:160px", maxlength: "11000" });
  prompt.value = p.prompt;
  const saveBtn = el("button", { class: "btn" }, "Save");
  const resetBtn = el("button", { class: "btn ghost" }, "Reset to default");
  resetBtn.disabled = !p.is_custom;

  saveBtn.addEventListener("click", async () => {
    if (!prompt.value.trim()) { toast("Prompt can't be empty"); return; }
    await api("/persona", "POST", { prompt: prompt.value });
    toast("Persona saved"); settingsViews.persona(root);
  });
  resetBtn.addEventListener("click", async () => {
    await api("/persona/reset", "POST");
    toast("Reset to default"); settingsViews.persona(root);
  });

  card.append(
    el("div", { class: "row", style: "justify-content:space-between;align-items:center" },
      el("label", { style: "margin:0" }, "System prompt"), badge),
    prompt,
    el("div", { class: "row", style: "margin-top:10px" }, saveBtn, resetBtn));
  root.appendChild(card);
};

settingsViews.skills = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Skills",
    "Reusable instructions Aria follows when you say their trigger phrase. Describe one in plain language and let Aria draft it, or write it yourself."));

  const draft = el("div", { class: "card" });
  const desc = el("textarea", { placeholder:
    "Describe the skill in your own words — e.g. \"When I paste rough meeting notes, turn them into a structured summary with action items.\"" });
  const draftBtn = el("button", { class: "btn ghost" }, "Draft with Aria");
  draft.append(el("label", {}, "Describe a skill"),
    el("div", { class: "grid" }, desc, el("div", {}, draftBtn)));
  root.appendChild(draft);

  const form = el("div", { class: "card" });
  const name = el("input", { placeholder: "Name (e.g. Meeting Notes)" });
  const trigger = el("input", { placeholder: "Trigger phrase (optional) — e.g. \"meeting notes\"" });
  const instructions = el("textarea", { placeholder: "Instructions Aria should follow when this skill is active…" });
  const saveBtn = el("button", { class: "btn" }, "Save skill");
  form.append(el("label", {}, "Skill details"),
    el("div", { class: "grid" }, name, trigger, instructions, el("div", {}, saveBtn)));
  root.appendChild(form);

  draftBtn.addEventListener("click", async () => {
    if (!desc.value.trim()) return;
    draftBtn.disabled = true; draftBtn.textContent = "Drafting…";
    try {
      const r = await api("/skills/draft", "POST", { description: desc.value });
      if (r.ok) {
        name.value = r.name; trigger.value = r.trigger; instructions.value = r.instructions;
        toast("Draft ready — review and save below");
      } else {
        toast(r.error || "Couldn't draft a skill from that");
      }
    } finally {
      draftBtn.disabled = false; draftBtn.textContent = "Draft with Aria";
    }
  });

  saveBtn.addEventListener("click", async () => {
    if (!name.value.trim() || !instructions.value.trim()) {
      toast("Name and instructions are required"); return;
    }
    await api("/skills", "POST", {
      name: name.value, instructions: instructions.value, trigger: trigger.value || null,
    });
    toast("Skill saved");
    settingsViews.skills(root);
  });

  const list = await api("/skills");
  const card = el("div", { class: "card" });
  card.appendChild(el("label", {}, `Saved skills (${list.length})`));
  if (list.length === 0) card.appendChild(el("div", { class: "empty" }, "No skills yet."));
  else {
    const tbl = el("table");
    tbl.appendChild(colgroup(18, 22, 42, 18));
    tbl.appendChild(el("tr", {}, el("th", {}, "Name"), el("th", {}, "Trigger"),
      el("th", {}, "Instructions"), el("th", {}, "")));
    list.forEach(s => {
      const del = el("button", { class: "btn danger" }, "Delete");
      del.addEventListener("click", async () => {
        await api("/skills/delete", "POST", { skill_id: s.id });
        toast("Deleted"); settingsViews.skills(root);
      });
      tbl.appendChild(el("tr", {},
        el("td", {}, s.name),
        el("td", { class: "muted" }, s.trigger ? el("code", {}, s.trigger) : "—"),
        el("td", { html: esc((s.instructions || "").slice(0, 160)) }),
        el("td", {}, del)));
    });
    card.appendChild(tbl);
  }
  root.appendChild(card);
};

settingsViews.tools = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Tools", "Functions the model can call. Every call is logged below. " +
    "Enabling a \"risky\" tool lets the assistant invoke it on its own mid-conversation, not just via manual testing."));

  const tools = await api("/tools");
  const tcard = el("div", { class: "card" });
  tcard.appendChild(el("label", {}, "Available tools"));
  const tbl = el("table");
  tbl.appendChild(colgroup(18, 46, 14, 22));
  tbl.appendChild(el("tr", {}, el("th", {}, "Tool"), el("th", {}, "Description"),
    el("th", {}, "Status"), el("th", {}, "")));
  tools.forEach(t => {
    const toggle = el("button", { class: "btn ghost" }, t.enabled ? "Disable" : "Enable");
    toggle.addEventListener("click", async () => {
      await api("/tools/toggle", "POST", { name: t.name, enabled: !t.enabled });
      toast(`${t.name} ${t.enabled ? "disabled" : "enabled"}`); settingsViews.tools(root);
    });
    tbl.appendChild(el("tr", {},
      el("td", {}, el("code", {}, t.name), t.dangerous ? el("span", { class: "pill bad", style: "margin-left:6px" }, "risky") : null),
      el("td", { class: "muted" }, t.description),
      el("td", {}, el("span", { class: "pill " + (t.enabled ? "good" : "bad") }, t.enabled ? "on" : "off")),
      el("td", {}, toggle)));
  });
  tcard.appendChild(tbl);
  root.appendChild(tcard);

  const calls = await api("/tools/calls");
  const ccard = el("div", { class: "card" });
  ccard.appendChild(el("label", {}, `Call log (${calls.length})`));
  if (calls.length === 0) ccard.appendChild(el("div", { class: "empty" }, "No tool calls yet."));
  else {
    const tbl2 = el("table");
    tbl2.appendChild(colgroup(20, 16, 34, 15, 15));
    tbl2.appendChild(el("tr", {}, el("th", {}, "When"), el("th", {}, "Tool"),
      el("th", {}, "Args"), el("th", {}, "Status"), el("th", {}, "ms")));
    calls.forEach(c => tbl2.appendChild(el("tr", {},
      el("td", { class: "muted" }, fmtTime(c.created_at)),
      el("td", {}, el("code", {}, c.tool_name)),
      el("td", { html: esc((c.arguments || "").slice(0, 60)) }),
      el("td", {}, el("span", { class: "pill " + statusPill(c.status) }, c.status)),
      el("td", {}, String(c.duration_ms ?? "")))));
    ccard.appendChild(tbl2);
  }
  root.appendChild(ccard);
};
function statusPill(s) { return s === "ok" ? "good" : s === "denied" ? "warn" : "bad"; }

settingsViews.training = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Training",
    "Periodic fine-tunes on your feedback. Style & behavior — not facts. A candidate must beat the current model before it's promoted."));

  const fb = await api("/status").then(s => s.feedback).catch(() => ({}));
  const tcard = el("div", { class: "card" });
  tcard.appendChild(el("div", { class: "row", style: "justify-content:space-between" },
    el("div", {},
      el("div", { html: `<b>${fb.pending ?? 0}</b> pending examples` }),
      el("div", { class: "muted", html: `${fb.total ?? 0} total · trains at ${fb.threshold ?? "—"}` })),
    (() => {
      const b = el("button", { class: "btn" }, "Train now");
      b.addEventListener("click", async () => {
        b.disabled = true; b.innerHTML = ""; b.appendChild(el("span", { class: "spinner" }));
        try {
          const r = await api("/train", "POST");
          toast(`Training: ${r.status}` + (r.reason ? ` — ${r.reason}` : ""));
        } finally { settingsViews.training(root); }
      });
      return b;
    })()));
  root.appendChild(tcard);

  const add = el("div", { class: "card" });
  const ins = el("textarea", { placeholder: "Instruction / prompt" });
  const pref = el("textarea", { placeholder: "Preferred answer (what Aria should have said)" });
  const ab = el("button", { class: "btn ghost" }, "Add feedback example");
  ab.addEventListener("click", async () => {
    if (!ins.value.trim() || !pref.value.trim()) return;
    await api("/feedback", "POST", { instruction: ins.value, preferred: pref.value });
    toast("Feedback saved"); ins.value = pref.value = ""; settingsViews.training(root);
  });
  add.append(el("label", {}, "Teach by example"), el("div", { class: "grid" }, ins, pref, el("div", {}, ab)));
  root.appendChild(add);

  const runs = await api("/training/runs");
  const rcard = el("div", { class: "card" });
  rcard.appendChild(el("label", {}, `Training runs (${runs.length})`));
  if (runs.length === 0) rcard.appendChild(el("div", { class: "empty" }, "No runs yet."));
  else {
    const tbl = el("table");
    tbl.appendChild(colgroup(26, 16, 16, 16, 26));
    tbl.appendChild(el("tr", {}, el("th", {}, "When"), el("th", {}, "Examples"),
      el("th", {}, "Cand."), el("th", {}, "Base"), el("th", {}, "Result")));
    runs.forEach(r => tbl.appendChild(el("tr", {},
      el("td", { class: "muted" }, fmtTime(r.created_at)),
      el("td", {}, `${r.n_train ?? "?"}+${r.n_held_out ?? "?"}`),
      el("td", {}, fmtScore(r.candidate_score)),
      el("td", {}, fmtScore(r.baseline_score)),
      el("td", {}, el("span", { class: "pill " + runPill(r.status) }, r.status)))));
    rcard.appendChild(tbl);
  }
  root.appendChild(rcard);
};
const fmtScore = (x) => (x == null ? "—" : Number(x).toFixed(3));
function runPill(s) {
  return s === "promoted" ? "good" : s === "rejected" ? "warn" :
    s === "failed" ? "bad" : "info";
}

settingsViews.adapters = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Adapters",
    "Every trained version of Aria's behavior. Exactly one is active. Roll back anytime."));
  const list = await api("/adapters");
  const card = el("div", { class: "card" });
  if (list.length === 0) card.appendChild(el("div", { class: "empty" },
    "No adapters yet. Train one from the Training tab."));
  else {
    const tbl = el("table");
    tbl.appendChild(colgroup(20, 18, 12, 20, 14, 16));
    tbl.appendChild(el("tr", {}, el("th", {}, "Version"), el("th", {}, "Base"),
      el("th", {}, "Score"), el("th", {}, "Created"), el("th", {}, "State"), el("th", {}, "")));
    list.forEach(a => {
      const active = a.is_active == 1;
      const act = el("button", { class: "btn ghost" }, active ? "Active" : "Activate");
      act.disabled = active;
      act.addEventListener("click", async () => {
        await api("/adapters/activate", "POST", { adapter_id: a.id });
        toast("Activated " + a.id); refreshStatus(); settingsViews.adapters(root);
      });
      tbl.appendChild(el("tr", {},
        el("td", {}, el("code", {}, a.id)),
        el("td", { class: "muted" }, a.base_model || "—"),
        el("td", {}, fmtScore(a.eval_score)),
        el("td", { class: "muted" }, fmtTime(a.created_at)),
        el("td", {}, active ? el("span", { class: "pill good" }, "active")
          : (a.promoted ? el("span", { class: "pill info" }, "promoted")
            : el("span", { class: "pill warn" }, "rejected"))),
        el("td", {}, act)));
    });
    card.appendChild(tbl);
  }
  root.appendChild(card);
};

settingsViews.models = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Models", "Download and manage local weights. All inference runs on-device."));
  const m = await api("/models");
  const card = el("div", { class: "card" });
  const tbl = el("table");
  tbl.appendChild(colgroup(16, 26, 11, 19, 14, 14));
  tbl.appendChild(el("tr", {}, el("th", {}, "Model"), el("th", {}, "HF repo"),
    el("th", {}, "Size"), el("th", {}, "Role"), el("th", {}, "State"), el("th", {}, "")));
  Object.entries(m.catalog).forEach(([id, info]) => {
    const have = m.downloaded.includes(id);
    const cur = m.current === id;
    const dl = el("button", { class: "btn ghost" }, "Download");
    dl.addEventListener("click", async () => {
      dl.disabled = true; dl.textContent = "Starting…";
      const r = await api("/models/download", "POST", { model_id: id });
      if (!r.ok) { toast(r.error || "Download failed"); settingsViews.models(root); return; }
      try {
        await new Promise((resolve, reject) => {
          const tick = async () => {
            let p;
            try { p = await api(`/models/download/progress?model_id=${encodeURIComponent(id)}`); }
            catch (e) { reject(e); return; }
            dl.textContent = p.total_bytes > 0
              ? Math.round(p.percent || 0) + "%"
              : fmtBytes(p.downloaded_bytes) + "…";
            if (p.status === "done") return resolve();
            if (p.status === "error") return reject(new Error(p.error || "Download failed"));
            setTimeout(tick, 700);
          };
          tick();
        });
        toast("Downloaded " + id);
      } catch (e) {
        toast(e.message);
      }
      settingsViews.models(root);
    });
    // Only one action button per row — a disabled "Downloaded" ghost button
    // next to "Load" was redundant (the State column's "on disk" pill
    // already says that) and squeezed the actions column too narrow.
    const actions = [];
    if (!have) {
      actions.push(dl);
    } else if (!cur) {
      const load = el("button", { class: "btn small" }, "Load");
      load.addEventListener("click", async () => {
        load.disabled = true; load.textContent = "Loading…";
        try {
          await api("/models/load", "POST", { model_id: id });
          toast("Loaded " + id);
          refreshStatus();
        } catch { /* api() already toasts */ }
        settingsViews.models(root);
      });
      actions.push(load);
    }
    tbl.appendChild(el("tr", {},
      el("td", {}, el("code", {}, id)),
      el("td", { class: "muted" }, info.repo),
      el("td", { class: "nowrap" }, info.size_gb + " GB"),
      el("td", { class: "muted" }, info.role),
      // "loaded" implies on-disk, so showing both pills was redundant and
      // the two together didn't fit this column at settings-sheet width —
      // one pill communicates the model's actual state either way.
      el("td", { class: "nowrap" }, cur ? el("span", { class: "pill good" }, "loaded")
        : have ? el("span", { class: "pill good" }, "on disk") : el("span", { class: "pill" }, "—")),
      el("td", {}, actions)));
  });
  card.appendChild(tbl);
  root.appendChild(card);

  root.appendChild(el("div", { class: "view-head", style: "margin-top:22px" },
    el("h1", {}, "Voice"),
    el("p", {}, "Natural offline speech, fully on-device.")));
  const vcard = el("div", { class: "card" });
  try {
    const v = await api("/voice");
    const row = el("div", { class: "voice-row" });

    const voiceSel = el("select", {});
    (v.voices || []).forEach(name => {
      const o = el("option", { value: name }, name);
      if (name === v.active_voice) o.selected = true;
      voiceSel.appendChild(o);
    });
    voiceSel.addEventListener("change", async () => {
      await api("/voice/set", "POST", { voice: voiceSel.value });
      toast("Voice → " + voiceSel.value);
    });

    const rate = el("input", { type: "range", min: "0.7", max: "1.4", step: "0.05", value: String(v.rate || 1) });
    const rateLbl = el("span", { class: "muted" }, "×" + (v.rate || 1).toFixed(2));
    rate.addEventListener("input", () => rateLbl.textContent = "×" + Number(rate.value).toFixed(2));
    rate.addEventListener("change", async () => {
      await api("/voice/set", "POST", { rate: Number(rate.value) });
      toast("Speaking rate " + Number(rate.value).toFixed(2) + "×");
    });

    const test = el("button", { class: "btn ghost", html:
      '<svg viewBox="0 0 20 20" fill="none" aria-hidden="true" style="width:12px;height:12px;vertical-align:-1px;margin-right:5px"><path d="M6.5 4.5v11l9-5.5-9-5.5z" fill="currentColor"/></svg>Test voice' });
    test.addEventListener("click", async () => {
      const r = await api("/speak", "POST", {
        text: "Hi, I'm Aria. I run entirely on your Mac, and I can talk with you naturally for as long as you like."
      });
      if (r.segments && r.segments.length) speakSegments(r.segments);
    });

    row.append(
      el("label", {}, "Voice", voiceSel),
      el("label", {}, "Speed", rate, rateLbl),
      test);
    vcard.appendChild(row);

    const badges = el("div", { class: "voice-badges" },
      el("span", { class: "pill " + (v.neural ? "good" : "") }, v.neural ? "neural" : "system voice"),
      el("span", { class: "pill good" }, v.offline ? "offline" : "online"),
      el("span", { class: "pill info" }, "backend: " + v.name),
      el("span", { class: "pill" }, (v.sample_rate || 0) + " Hz"));
    vcard.appendChild(badges);
    if (v.notes) vcard.appendChild(el("div", { class: "muted", style: "margin-top:8px" }, v.notes));
  } catch {
    vcard.appendChild(el("div", { class: "empty" }, "Voice engine unavailable."));
  }
  root.appendChild(vcard);
};

function headerBar(title, sub) {
  return el("div", { class: "view-head" },
    el("div", {}, el("h1", {}, title), sub ? el("p", {}, sub) : null),
    (() => { const b = el("button", { class: "btn ghost" }, "↻");
      b.addEventListener("click", () => { switchSettingsView(currentSettingsView); refreshStatus(); }); return b; })());
}

let currentSettingsView = "memory";
let settingsRetryTimer = null;
let settingsRetryCount = 0;
const SETTINGS_MAX_AUTO_RETRIES = 8; // ~24s at 3s apart — comfortably past a cold start

function switchSettingsView(name) {
  clearTimeout(settingsRetryTimer);
  if (name !== currentSettingsView) settingsRetryCount = 0;
  currentSettingsView = name;
  document.querySelectorAll(".settings-tab").forEach(b =>
    b.classList.toggle("active", b.dataset.view === name));
  const root = $("#settings-body");
  root.scrollTop = 0; root.scrollLeft = 0;
  root.innerHTML = "<div class='empty'><span class='spinner'></span></div>";
  (settingsViews[name] || settingsViews.memory)(root).then(() => {
    settingsRetryCount = 0;
  }).catch(e => {
    root.innerHTML = `<div class='empty'>Could not load. Is the sidecar running?<br><br><code>${esc(e.message)}</code></div>`;
    // Most failures here are just "asked during the ~20s cold-start window
    // before the sidecar was listening yet" — retry quietly a few times
    // instead of leaving the user staring at a dead panel they have to
    // remember to manually refresh once it's actually up.
    if (settingsRetryCount++ < SETTINGS_MAX_AUTO_RETRIES &&
        $("#settings-overlay").classList.contains("open") &&
        currentSettingsView === name) {
      settingsRetryTimer = setTimeout(() => switchSettingsView(name), 3000);
    }
  });
}

function openSettings(view) {
  $("#settings-overlay").classList.add("open");
  switchSettingsView(view || currentSettingsView);
  checkForUpdates();
}
function closeSettings() { $("#settings-overlay").classList.remove("open"); }

function wireSettings() {
  $("#settings-btn").addEventListener("click", () => openSettings());
  $("#settings-close").addEventListener("click", closeSettings);
  $("#settings-overlay").addEventListener("click", e => { if (e.target.id === "settings-overlay") closeSettings(); });
  document.querySelectorAll(".settings-tab").forEach(b =>
    b.addEventListener("click", () => switchSettingsView(b.dataset.view)));
  $("#update-check-btn").addEventListener("click", checkForUpdates);
  $("#relaunch-banner").addEventListener("click", relaunchIntoUpdate);
}

// ==========================================================================
// APP UPDATES — fully silent: check, download, and install all happen in
// the background (server-side, see update_auto_pull() in app.py) with
// nothing landing in ~/Downloads. The UI's only job is to notice when a
// build is ready on disk and offer a one-click "Relaunch to update" — the
// same shape as Claude desktop's own updater.
// ==========================================================================
let lastUpdateCheck = null;
let autoUpdatePollTimer = null;

function setUpdateBadge(visible) {
  $("#settings-update-badge")?.toggleAttribute("hidden", !visible);
}

function showRelaunchBanner(version) {
  $("#relaunch-version").textContent = `v${version}`;
  $("#relaunch-banner").hidden = false;
}

// Polls /update/auto/status while a pull is in flight, stopping once it
// lands on "ready" (show the relaunch banner) or "error" (fail silently —
// the next periodic checkForUpdatesSilently() will retry from scratch).
function pollAutoUpdateStatus() {
  clearTimeout(autoUpdatePollTimer);
  const tick = async () => {
    let s;
    try { s = await api("/update/auto/status"); } catch { return; }
    if (s.phase === "ready") {
      setUpdateBadge(true);
      showRelaunchBanner(s.version);
      return;
    }
    if (s.phase === "error") return;
    autoUpdatePollTimer = setTimeout(tick, 4000);
  };
  tick();
}

// Silent background check — records the result, lights the settings-gear
// badge, and (if an update exists) kicks the auto-pull pipeline so it's
// downloaded and installed well before the user ever notices.
async function checkForUpdatesSilently() {
  let r;
  try {
    r = await api("/update/auto", "POST");
  } catch {
    return;
  }
  lastUpdateCheck = r;
  if (r.ok && r.update_available) {
    setUpdateBadge(true);
    pollAutoUpdateStatus();
  }
}

// Manual "Check for Updates" in Settings — same auto-pull pipeline, just
// with visible status text instead of operating silently.
async function checkForUpdates() {
  const statusEl = $("#update-status");
  const actionEl = $("#update-action");
  const btn = $("#update-check-btn");
  btn.disabled = true; btn.textContent = "Checking…";
  actionEl.style.display = "none"; actionEl.innerHTML = "";

  let r;
  try {
    r = await api("/update/auto", "POST");
  } catch {
    statusEl.textContent = "Couldn't check for updates";
    btn.disabled = false; btn.textContent = "Check for Updates";
    return;
  }
  lastUpdateCheck = r;
  btn.disabled = false; btn.textContent = "Check for Updates";

  const currentVersion = r.current_version || APP_VERSION_FALLBACK;
  statusEl.classList.remove("available");
  statusEl.textContent = `Aria v${currentVersion}`;

  if (!r.enabled) return;
  if (!r.ok) {
    statusEl.textContent += " — update check failed: " + (r.error || "unknown error");
    return;
  }
  if (!r.update_available) {
    statusEl.textContent += " — up to date";
    setUpdateBadge(false);
    return;
  }

  setUpdateBadge(true);
  statusEl.classList.add("available");
  actionEl.style.display = "flex";
  const notes = r.notes ? el("div", { id: "update-notes" }, r.notes) : null;
  if (notes) actionEl.append(notes);

  const tick = async () => {
    let s;
    try { s = await api("/update/auto/status"); } catch { return; }
    if (s.phase === "ready") {
      statusEl.textContent = statusEl.textContent.replace(/ — .*/, "") +
        ` — v${s.version} ready`;
      actionEl.innerHTML = "";
      const relaunchBtn = el("button", { class: "btn small" }, "Relaunch now");
      relaunchBtn.addEventListener("click", relaunchIntoUpdate);
      actionEl.append(relaunchBtn);
      showRelaunchBanner(s.version);
      return;
    }
    if (s.phase === "error") {
      statusEl.textContent = statusEl.textContent.replace(/ — .*/, "") +
        ` — update failed: ${s.error || "unknown error"}`;
      return;
    }
    statusEl.textContent = statusEl.textContent.replace(/ — .*/, "") +
      ` — v${r.latest_version} ${s.phase}…`;
    setTimeout(tick, 1500);
  };
  tick();
}

async function openUrl(url) {
  const invoke = tauriInvoke();
  if (invoke) {
    try { await invoke("plugin:shell|open", { path: url }); return; } catch { /* fall through */ }
  }
  window.open(url, "_blank");
}

// The new build is already installed on disk (update_auto_pull did that in
// the background) — this just opens it fresh, then quits this now-outdated
// instance, the same handoff pattern the old manual-install flow used.
async function relaunchIntoUpdate() {
  const banner = $("#relaunch-banner");
  banner.disabled = true;
  let r;
  try {
    r = await api("/update/relaunch", "POST");
  } catch (e) {
    r = { ok: false, error: e.message };
  }
  if (!r.ok) {
    toast(r.error || "Couldn't relaunch — quit Aria (⌘Q) and reopen it to use the new version.");
    banner.disabled = false;
    return;
  }
  const invoke = tauriInvoke();
  if (!invoke) {
    toast("Update installed — restart Aria to use it.");
    return;
  }
  setTimeout(() => {
    invoke("quit_now").catch(() => {});
    // If quit_now worked, this whole page is torn down before this timer
    // fires. Still here after a few seconds means the restart itself
    // silently failed, so hand the user something actionable instead of a
    // permanently spinning banner.
    setTimeout(() => {
      banner.disabled = false;
      toast("Couldn't restart automatically — quit Aria (⌘Q) and reopen it.");
    }, 3000);
  }, 800);
}

// ==========================================================================
// BOOT — discover the sidecar port (Tauri), then start the UI
// ==========================================================================
function tauriInvoke() {
  const t = window.__TAURI__;
  if (t?.core?.invoke) return t.core.invoke;
  if (t?.invoke) return t.invoke;
  if (t?.tauri?.invoke) return t.tauri.invoke;
  return null;
}

// A cold PyInstaller onefile launch self-extracts before it can even start
// listening — comfortably under 20s on a warm disk cache, but slower disks
// or the very first launch after an install can take longer. This loop
// NEVER permanently gives up (confirmed live: a one-shot bounded version
// that stopped retrying after 60s left a fully healthy, responsive sidecar
// unreachable from the UI for 4+ minutes with no way to recover short of
// restarting the app). It keeps trying indefinitely in the background —
// fast for the first ~30s to catch a normal cold boot promptly, then backs
// off so a genuinely stuck case doesn't spin uselessly — and self-heals
// `API` + the visible status the moment it does succeed, however late.
async function resolveSidecarPortLoop() {
  let attempt = 0;
  while (!sidecarConfirmed) {
    // Re-checked every iteration, not just once up front — Tauri injects
    // `window.__TAURI__` into the webview asynchronously, and this script
    // can start running before that injection lands. A one-shot null check
    // here would exit permanently on that race, and no retry loop below it
    // could ever recover: confirmed live as the root cause of the app
    // staying on "Starting…" forever despite a fully healthy sidecar.
    const invoke = tauriInvoke();
    if (invoke) {
      try {
        const port = await invoke("sidecar_port");
        if (port) {
          API = `http://127.0.0.1:${port}`;
          sidecarConfirmed = true;
          refreshStatus();
          return;
        }
      } catch { /* command not ready */ }
    }
    attempt++;
    await new Promise(r => setTimeout(r, attempt < 150 ? 200 : 2000));
  }
}

async function boot() {
  const invoke = tauriInvoke();
  const portLoop = resolveSidecarPortLoop();  // fire-and-forget — keeps running regardless of the gate below
  // api() awaits this so a user opening Settings or sending a message the
  // instant the window appears can't race ahead of port resolution and
  // hit the hardcoded dev-fallback port. Capped at 5s so a slow or stuck
  // resolution can never block the app itself forever — the background
  // loop above keeps retrying and corrects things whenever it does land,
  // even well after this gate has already let calls through.
  sidecarReadyPromise = invoke
    ? Promise.race([portLoop, new Promise(r => setTimeout(r, 5000))])
    : null;

  wireOnboarding();
  wireChat();
  wireSettings();
  wireHistorySidebar();

  const ev = window.__TAURI__?.event;
  if (ev?.listen) {
    ev.listen("sidecar-ready", (e) => {
      if (e?.payload) { API = `http://127.0.0.1:${e.payload}`; sidecarConfirmed = true; refreshStatus(); }
    }).catch(() => {});
  }
  await sidecarReadyPromise;

  const skipped = await maybeSkipOnboarding();
  if (!skipped) await populateChoices();

  refreshSttAvailability();
  setInterval(refreshStatus, 8000);

  // Silent, non-blocking — a failed/offline check must never delay the app
  // being usable, so this runs after boot's own critical-path work is done.
  checkForUpdatesSilently();
  setInterval(checkForUpdatesSilently, 6 * 60 * 60 * 1000);
}

boot();
