/* SaafScan — motion + presentation layer (ObsidianUI-inspired, vanilla).
   Split into two scopes:

   1. PRESENTATION (always on — even under prefers-reduced-motion):
        · confidence gauge arc follows #v-meter width
        · viewfinder camera chip state from #camera-state text
        · viewfinder "analysing" overlay + aria-busy from #v-chip text
        · workflow stepper states (verdict shown / history present)
   2. MOTION (only when the user has not requested reduced motion):
        · html.motion-ready gate for CSS entrance choreography
        · staggered panel entrance, confidence count-up, history skeleton

   Never changes what app.js computes — only how it is presented. */
"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const GAUGE_C = 326.73; // 2πr, r = 52

  /* ============================================================
     1 · PRESENTATION — always on
     ============================================================ */

  /* ---- gauge arc (data, not motion: applied even when static) --- */
  const arc = document.querySelector(".gauge-arc");
  const meter = $("v-meter");
  const conf = $("v-conf");
  const CONF_RE = /(\d+)%\s*confidence/i; // format written by app.js
  let countRaf = 0;

  function applyArc(pct) {
    if (arc) arc.style.strokeDashoffset = String(GAUGE_C * (1 - pct / 100));
  }

  function countUp(to) {
    cancelAnimationFrame(countRaf);
    if (!conf || !CONF_RE.test(conf.textContent)) return; // "—"/"no verdict": leave alone
    const started = performance.now();
    const dur = 650;
    (function tick(now) {
      const p = Math.min(1, (now - started) / dur);
      const eased = 1 - Math.pow(1 - p, 4);
      conf.textContent = Math.round(to * eased) + "% confidence";
      if (p < 1) countRaf = requestAnimationFrame(tick);
    })(started);
  }

  if (meter) {
    new MutationObserver(() => {
      const pct = parseFloat(meter.style.width) || 0;
      applyArc(pct);
      if (!reduce.matches) countUp(pct);
    }).observe(meter, { attributes: true, attributeFilter: ["style"] });
  }

  /* ---- camera status chip (text driven by app.js) --------------- */
  const camState = $("camera-state");
  const camChip = $("camera-chip");
  function syncCamChip() {
    if (!camState || !camChip) return;
    const t = camState.textContent || "";
    camChip.dataset.state = /live/i.test(t)
      ? "ok"
      : /(unavailable|unsupported|denied|error)/i.test(t)
        ? "bad"
        : "checking";
  }
  if (camState) {
    new MutationObserver(syncCamChip).observe(camState, {
      childList: true, characterData: true, subtree: true,
    });
    syncCamChip();
  }

  /* ---- "analysing" viewfinder overlay + aria-busy --------------- */
  const chip = $("v-chip");
  const viewfinder = $("viewfinder");
  const verdictCard = $("verdict-card");
  function syncBusy() {
    if (!chip) return;
    const busy = /(analysing|rendering)/i.test(chip.textContent || "");
    if (viewfinder) viewfinder.classList.toggle("analysing", busy);
    if (verdictCard) verdictCard.setAttribute("aria-busy", busy ? "true" : "false");
  }
  if (chip) {
    new MutationObserver(syncBusy).observe(chip, {
      childList: true, characterData: true, subtree: true,
    });
    syncBusy();
  }

  /* ---- workflow stepper ------------------------------------------
     Rail semantics: the active step is where the user is *now*
     (capture, or the verdict on screen); steps after a completed cycle
     are styled done. A latched flag prevents the refresh skeleton
     (which momentarily contains no .entry) from flickering the rail. */
  const steps = document.querySelectorAll(".step");
  const verdictPanel = $("verdict-panel");
  const history = $("history");
  let historySeen = false;
  function syncSteps() {
    if (!steps.length) return;
    const verdictShown = verdictPanel && !verdictPanel.classList.contains("hidden");
    if (history && history.querySelector(".entry")) historySeen = true;
    const current = verdictShown ? 1 : 0;
    steps.forEach((s, i) => {
      const active = i === current;
      const done = verdictShown ? i === 0 : i > 0 && historySeen;
      s.classList.toggle("active", active);
      s.classList.toggle("done", !active && done);
      if (active) s.setAttribute("aria-current", "step");
      else s.removeAttribute("aria-current");
    });
  }
  if (verdictPanel) {
    new MutationObserver(syncSteps).observe(verdictPanel, {
      attributes: true, attributeFilter: ["class"],
    });
  }
  if (history) {
    new MutationObserver(syncSteps).observe(history, { childList: true });
  }
  syncSteps();

  /* ============================================================
     2 · MOTION — skipped under prefers-reduced-motion
     ============================================================ */

  if (reduce.matches) return;
  document.documentElement.classList.add("motion-ready");

  /* ---- staggered entrance -------------------------------------- */
  const revealables = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry, i) => {
        if (!entry.isIntersecting) return;
        setTimeout(() => entry.target.classList.add("in"), i * 60);
        io.unobserve(entry.target);
      });
    }, { threshold: 0.04 });
    revealables.forEach((el) => io.observe(el));
  } else {
    revealables.forEach((el) => el.classList.add("in"));
  }

  /* ---- history skeleton while refetching ------------------------- */
  const refresh = $("refresh");
  if (refresh && history) {
    refresh.addEventListener("click", () => {
      const row = '<div class="entry-skeleton"><span></span><span></span><span></span></div>';
      history.innerHTML = row + row + row;
      // app.js's own fetch replaces innerHTML when the response lands.
    });
  }
})();
