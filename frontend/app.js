/* SaafScan SPA — capture, analyse, verdict, history, report. No frameworks. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { stream: null, facing: "environment", lastVerdictId: null, busy: false };

/* ---------------------------------------------------------------- health -- */
async function ping() {
  const el = $("status");
  try {
    const r = await fetch("/health");
    const body = await r.json();
    el.textContent = body.status === "ok" ? "● ready" : "● degraded";
    el.dataset.state = body.status === "ok" ? "ok" : "down";
  } catch {
    el.textContent = "● offline";
    el.dataset.state = "down";
  }
}

/* ---------------------------------------------------------------- camera -- */
async function startCamera() {
  const hint = $("camera-state");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    hint.textContent = "unsupported — use upload";
    return;
  }
  try {
    stopCamera();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing },
      audio: false,
    });
    const video = $("video");
    video.srcObject = state.stream;
    await video.play();
    hint.textContent = "live";
    $("snap").disabled = false;
    $("flip").disabled = false;
  } catch (err) {
    // Expected over plain HTTP on LAN — secure-context requirement.
    hint.textContent = `unavailable (${err.name || "error"}) — use upload`;
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
}

function grabFrame() {
  const video = $("video");
  const canvas = $("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

/* --------------------------------------------------------------- analyse -- */
async function analyseBlob(blob, tag) {
  if (state.busy) return;
  state.busy = true;
  showVerdictPanel(true);
  setError("");
  $("v-chip").textContent = "analysing…";
  $("v-chip").dataset.tone = "warn";
  $("v-conf").textContent = "—";
  $("v-meter").style.width = "0%";
  const started = performance.now();

  try {
    const form = new FormData();
    form.append("file", blob, tag || "capture.jpg");
    const r = await fetch("/analyze", { method: "POST", body: form });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(typeof body.detail === "string" ? body.detail : `Upload rejected (HTTP ${r.status})`);
    }
    renderVerdict(body, (performance.now() - started) / 1000);
    await loadHistory();
  } catch (err) {
    $("v-chip").textContent = "rejected";
    $("v-chip").dataset.tone = "bad";
    $("v-conf").textContent = "no verdict";
    $("v-advice").textContent = "—";
    setError(err.message || String(err));
  } finally {
    state.busy = false;
  }
}

function renderVerdict(body, seconds) {
  const v = body.verdict;
  state.lastVerdictId = body.id;
  const tone = v.adulterant === "none" ? "ok" : v.adulterant === "inconclusive" ? "warn" : "bad";
  const chip = $("v-chip");
  chip.dataset.tone = tone;
  chip.textContent = {
    none: "PURE ✓",
    starch: "STARCH ADULTERATED",
    detergent: "DETERGENT ADULTERATED",
    both: "MULTIPLE ADULTERANTS",
    inconclusive: "INCONCLUSIVE",
  }[v.adulterant] || v.adulterant.toUpperCase();

  const pct = Math.round(v.confidence * 100);
  $("v-conf").textContent = `${pct}% confidence`;
  requestAnimationFrame(() => { $("v-meter").style.width = `${pct}%`; });
  $("v-sev").textContent = v.severity;
  $("v-advice").textContent = v.advice;
  $("v-evidence").textContent = JSON.stringify(v.evidence, null, 2);
  $("v-report").href = `/report/${body.id}`;
  $("v-latency").textContent =
    `verdict in ${seconds.toFixed(2)}s (server analysis ${Math.round(body.timing_ms)}ms) · id ${body.id}`;
}

function showVerdictPanel(show) {
  $("verdict-panel").classList.toggle("hidden", !show);
  if (show) $("verdict-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setError(msg) {
  const el = $("v-error");
  el.textContent = msg;
  el.classList.toggle("hidden", !msg);
}

/* --------------------------------------------------------------- history -- */
async function loadHistory() {
  try {
    const r = await fetch("/history?limit=20");
    const body = await r.json();
    const wrap = $("history");
    if (!body.entries || body.entries.length === 0) {
      wrap.innerHTML = '<p class="hint">No analyses yet.</p>';
      return;
    }
    wrap.innerHTML = body.entries.map((e) => {
      const tone = e.verdict.adulterant === "none" ? "ok"
        : e.verdict.adulterant === "inconclusive" ? "warn" : "bad";
      const label = e.verdict.adulterant.toUpperCase();
      return `<div class="entry">
        <span class="chip" data-tone="${tone}">${label}</span>
        <span class="when">${e.timestamp} · ${e.sample_name || "—"} · ${Math.round(e.verdict.confidence * 100)}%</span>
        <a href="/report/${e.id}" target="_blank">PDF ↗</a>
      </div>`;
    }).join("");
  } catch (err) {
    $("history").innerHTML = `<p class="error">history unavailable: ${err.message}</p>`;
  }
}

/* ----------------------------------------------------------- demo samples -- */
async function runDemo(label) {
  if (state.busy) return;
  state.busy = true;
  showVerdictPanel(true);
  setError("");
  $("v-chip").textContent = "rendering…";
  $("v-chip").dataset.tone = "warn";
  try {
    const imgR = await fetch(`/static/samples/${label}.jpg`);
    if (!imgR.ok) throw new Error(`demo sample '${label}' unavailable`);
    const blob = await imgR.blob();
    state.busy = false; // analyseBlob manages its own busy flag
    await analyseBlob(blob, `${label}.jpg`);
  } catch (err) {
    state.busy = false;
    setError(err.message || String(err));
  }
}

/* ------------------------------------------------------------------ wire -- */
$("snap").addEventListener("click", async () => {
  const blob = await grabFrame();
  if (blob) await analyseBlob(blob, "capture.jpg");
});

$("flip").addEventListener("click", () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  startCamera();
});

$("file-input").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) analyseBlob(file, file.name);
});

const drop = $("drop");
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) analyseBlob(file, file.name);
});

document.querySelectorAll("button.demo").forEach((btn) =>
  btn.addEventListener("click", () => runDemo(btn.dataset.sample)));

$("v-again").addEventListener("click", () => {
  showVerdictPanel(false);
  setError("");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

$("refresh").addEventListener("click", loadHistory);

startCamera();
ping();
loadHistory();
setInterval(ping, 15000);

// Deep link: /?demo=starch auto-runs a demo sample (dry-run rehearsals,
// headless screenshot capture for the deck).
const demoParam = new URLSearchParams(window.location.search).get("demo");
if (demoParam && document.querySelector(`button.demo[data-sample="${demoParam}"]`)) {
  setTimeout(() => runDemo(demoParam), 400);
}
