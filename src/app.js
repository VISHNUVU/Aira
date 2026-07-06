// Aria UI — vanilla JS, no framework. Talks to the Python sidecar HTTP API.
//
// Port discovery: in a bundled/dev Tauri build the Rust shell spawns the
// sidecar on an OS-picked port and exposes it via the `sidecar_port` command
// (and a `sidecar-ready` event). We resolve that before the first fetch. When
// running the UI in a plain browser (no Tauri), we fall back to the fixed dev
// port 8765 so `npm run dev` + `python app.py --port 8765` just works.
let API = (window.__ARIA_API__ || "http://127.0.0.1:8765");

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
const esc = (s) => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const fmtTime = (ms) => ms ? new Date(ms).toLocaleString() : "—";

function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2200);
}

async function api(path, method = "GET", body = null) {
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
async function refreshStatus() {
  try {
    const s = await api("/status");
    lastStatus = s;
    $("#st-engine").textContent = s.engine || "—";
    $("#st-model").textContent = s.model || "not loaded";
    $("#st-adapter").textContent = s.active_adapter || "base";
    $("#st-device").textContent = (s.capabilities && s.capabilities.device) || "—";
    $("#status-dot").classList.remove("offline");
    $("#status-text").textContent = s.loaded || (s.engine || "").toLowerCase().includes("fake")
      ? "Ready" : "Online";
    return s;
  } catch {
    lastStatus = null;
    $("#status-dot").classList.add("offline");
    $("#status-text").textContent = "Offline";
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
      micBtn.title = "Stop recording";
    } catch {
      toast("Microphone access denied");
    }
    return;
  }
  micRecording = false;
  micBtn.classList.remove("recording");
  micBtn.title = "Voice input";
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
const OB_STEPS = ["welcome", "choose", "downloading", "ready"];
let chosenModel = "gemma-4-12b";

function obShow(step) {
  document.querySelectorAll(".ob-step").forEach(s => s.classList.toggle("active", s.dataset.step === step));
  document.querySelectorAll(".ob-dot").forEach(d => d.classList.toggle("active", d.dataset.dot === step));
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

function renderMsg(m) {
  const d = el("div", { class: "msg " + m.role }, m.content);
  if (m.role === "assistant" && m.used_context)
    d.appendChild(el("div", { class: "ctx" }, "✓ used your memory as context"));
  return d;
}

function renderChat() {
  const msgs = $("#messages");
  msgs.innerHTML = "";
  if (chatHistory.length === 0) {
    msgs.appendChild(el("div", { class: "empty-chat" },
      el("div", { class: "mark" }, "◆"),
      el("h2", {}, "Hi, I'm Aria"),
      el("p", {}, "Ask me anything. I remember what you tell me, and everything stays on your Mac.")));
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
      const r = await api("/chat/speak", "POST", { messages: chatHistory, use_memory: true, use_tools: true });
      thinking.remove();
      const am = { role: "assistant", content: r.content, used_context: r.used_context };
      chatHistory.push(am); msgs.appendChild(renderMsg(am));
      msgs.scrollTop = msgs.scrollHeight;
      if (r.speech && r.speech.segments && r.speech.segments.length) speakSegments(r.speech.segments);
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

  let textAcc = "", usedContext = false, gotFirstToken = false;
  try {
    const res = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chatHistory, use_memory: true, use_tools: true }),
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
          bubble.textContent = textAcc;
          msgs.scrollTop = msgs.scrollHeight;
        }
        if (evt.done) usedContext = evt.used_context;
      }
    }
  } catch (e) {
    toast("⚠ " + e.message);
  }

  if (!textAcc) { bubble.remove(); return; }
  bubble.textContent = textAcc;
  if (usedContext) bubble.appendChild(el("div", { class: "ctx" }, "✓ used your memory as context"));
  chatHistory.push({ role: "assistant", content: textAcc, used_context: usedContext });
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
// SETTINGS SHEET — technical panels (Memory / Tools / Training / Adapters / Models)
// ==========================================================================
const settingsViews = {};

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

  const list = await api("/memory");
  const card = el("div", { class: "card" });
  card.appendChild(el("label", {}, `Stored chunks (${list.length})`));
  if (list.length === 0) card.appendChild(el("div", { class: "empty" }, "No memory yet."));
  else {
    const tbl = el("table");
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

settingsViews.tools = async function (root) {
  root.innerHTML = "";
  root.appendChild(headerBar("Tools", "Functions the model can call. Every call is logged below."));

  const tools = await api("/tools");
  const tcard = el("div", { class: "card" });
  tcard.appendChild(el("label", {}, "Available tools"));
  const tbl = el("table");
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
  tbl.appendChild(el("tr", {}, el("th", {}, "Model"), el("th", {}, "HF repo"),
    el("th", {}, "Size"), el("th", {}, "Role"), el("th", {}, "State"), el("th", {}, "")));
  Object.entries(m.catalog).forEach(([id, info]) => {
    const have = m.downloaded.includes(id);
    const cur = m.current === id;
    const dl = el("button", { class: "btn ghost" }, have ? "Downloaded" : "Download");
    dl.disabled = have;
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
    const actions = [dl];
    if (have && !cur) {
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
      el("td", {}, el("code", {}, id), cur ? el("span", { class: "pill good", style: "margin-left:6px" }, "loaded") : null),
      el("td", { class: "muted" }, info.repo),
      el("td", {}, info.size_gb + " GB"),
      el("td", { class: "muted" }, info.role),
      el("td", {}, have ? el("span", { class: "pill good" }, "on disk") : el("span", { class: "pill" }, "—")),
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

    const test = el("button", { class: "btn ghost" }, "▶ Test voice");
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
function switchSettingsView(name) {
  currentSettingsView = name;
  document.querySelectorAll(".settings-tab").forEach(b =>
    b.classList.toggle("active", b.dataset.view === name));
  const root = $("#settings-body");
  root.innerHTML = "<div class='empty'><span class='spinner'></span></div>";
  (settingsViews[name] || settingsViews.memory)(root).catch(e => {
    root.innerHTML = `<div class='empty'>Could not load. Is the sidecar running?<br><br><code>${esc(e.message)}</code></div>`;
  });
}

function openSettings(view) {
  $("#settings-overlay").classList.add("open");
  switchSettingsView(view || currentSettingsView);
}
function closeSettings() { $("#settings-overlay").classList.remove("open"); }

function wireSettings() {
  $("#settings-btn").addEventListener("click", () => openSettings());
  $("#settings-close").addEventListener("click", closeSettings);
  $("#settings-overlay").addEventListener("click", e => { if (e.target.id === "settings-overlay") closeSettings(); });
  document.querySelectorAll(".settings-tab").forEach(b =>
    b.addEventListener("click", () => switchSettingsView(b.dataset.view)));
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

async function resolveSidecarPort() {
  const invoke = tauriInvoke();
  if (!invoke) return false;
  for (let i = 0; i < 100; i++) {
    try {
      const port = await invoke("sidecar_port");
      if (port) { API = `http://127.0.0.1:${port}`; return true; }
    } catch { /* command not ready */ }
    await new Promise(r => setTimeout(r, 200));
  }
  return false;
}

async function boot() {
  wireOnboarding();
  wireChat();
  wireSettings();

  const ev = window.__TAURI__?.event;
  if (ev?.listen) {
    ev.listen("sidecar-ready", (e) => {
      if (e?.payload) { API = `http://127.0.0.1:${e.payload}`; refreshStatus(); }
    }).catch(() => {});
  }
  await resolveSidecarPort();

  const skipped = await maybeSkipOnboarding();
  if (!skipped) await populateChoices();

  refreshSttAvailability();
  setInterval(refreshStatus, 8000);
}

boot();
