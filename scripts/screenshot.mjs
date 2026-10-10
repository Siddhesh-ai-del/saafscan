#!/usr/bin/env node
/**
 * SaafScan — Phase 6 screenshot harness for the Round-1 deck.
 *
 * Captures: (1) landing/upload UI, (2) verdict card for a positive sample,
 * (3) evidence history, (4) demo-mode verdict. Writes PNGs to screenshots/.
 *
 * Usage:  node scripts/screenshot.mjs [baseUrl]
 * Expects the server already running (bash start_demo.sh).
 */
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(
  join(process.env.HOME, "Desktop/Test/liq-glass/node_modules/"),
);
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://127.0.0.1:8000";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "screenshots");
mkdirSync(OUT, { recursive: true });

const shot = async (page, name) => {
  const path = join(OUT, name);
  await page.screenshot({ path, fullPage: false });
  console.log(`✓ ${path}`);
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });

try {
  // 1 — landing / capture UI
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);
  await shot(page, "01-landing-capture.png");

  // 2 — verdict: starch (positive) via demo mode
  await page.click('button.demo[data-sample="starch"]');
  await page.waitForSelector("#verdict-card:not(.hidden)", { timeout: 15000 });
  await page.waitForSelector('#v-chip:text("STARCH")', { timeout: 15000 });
  await page.waitForTimeout(900); // let the confidence meter animate
  await shot(page, "02-verdict-starch.png");

  // 3 — verdict: pure (negative control) for contrast
  await page.click("#v-again");
  await page.click('button.demo[data-sample="none"]');
  await page.waitForSelector('#v-chip:text("PURE")', { timeout: 15000 });
  await page.waitForTimeout(900);
  await shot(page, "03-verdict-pure.png");

  // 4 — evidence history with multiple entries
  await page.click("#refresh");
  await page.waitForTimeout(500);
  const history = page.locator("#history-panel");
  await history.scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await shot(page, "04-evidence-history.png");

  // 5 — history scrolled into view + full page overview
  await page.screenshot({ path: join(OUT, "05-full-overview.png"), fullPage: true });
  console.log(`✓ ${join(OUT, "05-full-overview.png")}`);
} finally {
  await browser.close();
}
