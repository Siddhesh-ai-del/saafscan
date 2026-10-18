/* SaafScan — ambient backdrop (CanvasUI-inspired, hand-rolled vanilla).
   Paper dot-grid + drifting jade motes + a slow horizontal scan sweep.
   Decorative only, layout-neutral. Paused when the tab is hidden and
   reduced to a single static frame under prefers-reduced-motion. */
"use strict";
(function () {
  const canvas = document.getElementById("ambient-canvas");
  if (!canvas || !canvas.getContext) return;

  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const DPR = Math.min(window.devicePixelRatio || 1, 1.5);

  const GRID = 26;          // px between dots
  const SWEEP_MS = 14000;   // one slow pass of the scan band

  let w = 0;
  let h = 0;
  let motes = [];
  let raf = 0;
  let gridLayer = null; // static grid cached offscreen — never repainted per frame

  function seedMotes() {
    const count = Math.min(14, Math.max(8, Math.round((w * h) / 140000)));
    motes = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1 + Math.random() * 1.6,
      vx: (Math.random() - 0.5) * 0.12,
      vy: -(0.05 + Math.random() * 0.12),
      a: 0.1 + Math.random() * 0.16,
    }));
  }

  // The dot grid never moves — rasterize it once per resize instead of
  // issuing ~3,000 fillRect() calls every frame.
  function buildGridLayer() {
    gridLayer = document.createElement("canvas");
    gridLayer.width = Math.round(w * DPR);
    gridLayer.height = Math.round(h * DPR);
    const g = gridLayer.getContext("2d");
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    g.fillStyle = "rgba(22,24,28,0.05)";
    for (let x = GRID / 2; x < w; x += GRID) {
      for (let y = GRID / 2; y < h; y += GRID) g.fillRect(x, y, 1, 1);
    }
  }

  function resize() {
    w = canvas.clientWidth || window.innerWidth;
    h = canvas.clientHeight || window.innerHeight;
    canvas.width = Math.round(w * DPR);
    canvas.height = Math.round(h * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    buildGridLayer();
    seedMotes();
  }

  // Draws the cached grid layer (one blit — not thousands of rects).
  function drawGrid() {
    if (gridLayer) ctx.drawImage(gridLayer, 0, 0, w, h);
  }

  function drawSweep(t) {
    const p = (t % SWEEP_MS) / SWEEP_MS;
    const bandY = p * (h + 320) - 160;
    const grad = ctx.createLinearGradient(0, bandY - 130, 0, bandY + 130);
    grad.addColorStop(0, "rgba(15,138,95,0)");
    grad.addColorStop(0.5, "rgba(15,138,95,0.035)");
    grad.addColorStop(1, "rgba(15,138,95,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, bandY - 130, w, 260);
  }

  function drawMotes() {
    for (const m of motes) {
      ctx.beginPath();
      ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15,138,95," + m.a + ")";
      ctx.fill();
      m.x += m.vx;
      m.y += m.vy;
      if (m.y < -8) { m.y = h + 8; m.x = Math.random() * w; }
      if (m.x < -8) m.x = w + 8;
      else if (m.x > w + 8) m.x = -8;
    }
  }

  function frame(t) {
    ctx.clearRect(0, 0, w, h);
    drawGrid();
    drawSweep(t);
    drawMotes();
    raf = requestAnimationFrame(frame);
  }

  function start() {
    if (raf || document.hidden || reduce.matches) return;
    raf = requestAnimationFrame(frame);
  }

  function stop() {
    if (!raf) return;
    cancelAnimationFrame(raf);
    raf = 0;
  }

  window.addEventListener("resize", () => {
    resize();
    if (!raf) { // keep the static frame fresh when not animating
      ctx.clearRect(0, 0, w, h);
      drawGrid();
      drawSweep(0);
      drawMotes();
    }
  });
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  if (reduce.addEventListener) {
    reduce.addEventListener("change", () => {
      if (reduce.matches) { stop(); ctx.clearRect(0, 0, w, h); drawGrid(); drawSweep(0); drawMotes(); }
      else start();
    });
  }

  resize();
  if (reduce.matches) {
    ctx.clearRect(0, 0, w, h);
    drawGrid();
    drawSweep(0);
    drawMotes();
  } else {
    start();
  }
})();
