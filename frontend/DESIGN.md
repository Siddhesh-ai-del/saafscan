# SaafScan Frontend Design Spec — "Laboratory Editorial"

> Light editorial redesign of the SaafScan SPA. Hand-written vanilla HTML/CSS/JS —
> **no build step, no frameworks, no CDN** (FastAPI serves `frontend/` raw; the
> tagline claims *offline*). Design language inspired by ObsidianUI (component
> craft, tokens, motion, status patterns) and CanvasUI (ambient canvas effects),
> ported by hand because both libraries require React/Tailwind toolchains.

**Approved direction:** light editorial theme · no-build vanilla stack ·
100% of existing behavior preserved.

---

## 1. Design tokens (`css/tokens.css`)

### Color — paper & ink

| Token | Value | Role |
|---|---|---|
| `--paper` | `#F7F5F0` | page background (warm paper) |
| `--paper-2` | `#FDFCFA` | card surface (raised paper) |
| `--paper-3` | `#FFFFFF` | highest surface / insets |
| `--ink-9 … --ink-0` | `#16181C → #6E7178` ramp | text ramp: display ink → muted |
| `--ink` | `#16181C` | primary text |
| `--ink-soft` | `#3C3F45` | body text |
| `--muted` | `#6E7178` | secondary text |
| `--faint` | `#9A9DA3` | tertiary / kickers |
| `--line` | `rgba(22,24,28,.10)` | hairline borders |
| `--line-strong` | `rgba(22,24,28,.18)` | emphasized hairlines |
| `--accent` | `#0F8A5F` | deep jade (brand, links, primary) |
| `--accent-ink` | `#0A5C40` | accent on light (AA-safe text) |
| `--accent-wash` | `rgba(15,138,95,.08)` | tinted fills |

### Signal scale (maps to `data-tone` / `data-state` semantics)

| Token | Value | Meaning |
|---|---|---|
| `--signal-ok` / `--signal-ok-ink` | `#0F8A5F` / `#0A5C40` | pure / ready |
| `--signal-warn` / `--signal-warn-ink` | `#B7791F` / `#8A5A12` | inconclusive / checking |
| `--signal-bad` / `--signal-bad-ink` | `#C5372B` / `#9E2A20` | adulterated / offline |

All tone text at ≥14px uses the `*-ink` variants for AA contrast on paper.

### Type

| Token | Value |
|---|---|
| `--font-display` | `Fraunces` (self-hosted woff2, `font-display: swap`) → Georgia, serif |
| `--font-body` | `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` |
| `--font-mono` | `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` |
| scale (fluid) | `12 / 13 / 15 / 17 / 21 / 28 / 44` px via `clamp()` |
| numerals | `font-variant-numeric: tabular-nums` on `.mono`, gauges, history, latency |

### Space / radius / shadow / motion

- **Space:** 4px base — `--s-1…--s-8` = 4/8/12/16/24/32/48/64.
- **Radii:** `--r-sm 6px · --r-md 10px · --r-lg 16px · --r-pill 999px`.
- **Shadows:** 3 levels — hairline cards get `border: 1px solid var(--line)` +
  `inset 0 1px 0 rgba(255,255,255,.9)` top highlight + soft ambient shadow.
- **Motion:** `--ease: cubic-bezier(.22,1,.36,1)` (out-quart), durations
  `120 / 240 / 480ms`, `--z-*` scale. Global `prefers-reduced-motion` guard.

---

## 2. Component inventory

| # | Component | Spec |
|---|---|---|
| C1 | **TopBar** | Slim sticky bar, translucent paper + `backdrop-filter: blur(14px)`, SVG concentric lens mark, Fraunces wordmark, hairline bottom. |
| C2 | **StatusPill** | `#status` dot + soft ping; `data-state="ok|down|checking"`; mono label; reduced-motion kill-switch. |
| C3 | **Stepper** | `1 Capture → 2 Verdict → 3 Evidence log` rail (mono numerals), active/completed/upcoming states; verdict step activates when `#verdict-panel` un-hides. Replaces cyan uppercase `h2`s (headings become kickers). |
| C4 | **Panel** | Raised paper card: surface gradient + hairline + white top highlight + soft shadow; header slot = mono kicker + Fraunces title + action slot. |
| C5 | **Button system** | `.btn` primary / secondary(=`.btn` base) / ghost / `.demo`; `sm`/`md`; `:focus-visible` ring; `:disabled` dimmed; primary = jade fill with 1px light top edge; press = 1px translate. |
| C6 | **Viewfinder** | `#video` framed as a device: 4 SVG corner reticles, 4:3 aspect lock, slow scan-line sweep when live, mirrored preview, status chip on `#camera-state`; explicit unavailable state → "use upload". |
| C7 | **DropZone** | `#drop` as solid hairline card (no dashed cliché), inline SVG upload glyph, drag-over (`.over`) = accent glow + scale 1.01, stays a `<label>` wrapping `#file-input` (keyboard/AT safe). |
| C8 | **DemoChip** | `button.demo[data-sample]` pills with inline SVG icons (milk drop / starch grain / soap bubble), tone-tinted hover. No emoji. |
| C9 | **VerdictCard** | Hero: big Fraunces verdict word colored by `data-tone`, entrance reveal; **SVG arc gauge** driven by `#v-meter` width (count-up lives in the gauge, `#v-conf` untouched); severity badge; advice as left-accented callout; `details` with chevron + mono evidence block; actions row; `aria-live="polite"` on `#v-chip`. |
| C10 | **HistoryList** | Rows as ruled editorial list: tone dot + mono verdict label, **humanized UTC timestamp** ("10 Oct · 11:26 UTC"), sample, tabular confidence, `PDF ↗` ghost link; hover lift; empty + skeleton states. |
| C11 | **Banner** | `#v-error` = dismissible danger banner (icon, slide-in), still driven by `.hidden`. |
| C12 | **Footer** | Compact, centered, hairline top, mono legal line. |

---

## 3. Motion & ambient layer (CanvasUI-inspired, vanilla)

- `js/ambient.js` — one `<canvas>` behind the capture hero: faint dot grid +
  a few drifting jade particles + slow horizontal scan sweep. Paused on
  `document.hidden`, fully disabled under `prefers-reduced-motion`. ~3KB.
- `js/motion.js` — IntersectionObserver staggered entrance (translateY 12px +
  fade, 240ms, 40ms stagger), smooth `#verdict-panel` reveal, gauge arc +
  count-up (reads `#v-meter` style width via MutationObserver; `#v-conf` text is
  never rewritten), history row reveal — all gated on reduced-motion.

**Screenshot safety:** all entrance animations complete <600ms; headless
capture uses `--virtual-time-budget=12000`, so nothing may loop visibly in a
way that shifts layout mid-capture (ambient canvas is decorative-only and
layout-neutral).

---

## 4. Layout

- Asymmetric 2-column grid (`capture 1.15fr / verdict 1fr`); verdict space is
  *reserved away* while hidden (grid auto-flow — no empty hole).
- Full-width history below; footer.
- Breakpoints: 880px → single column (viewfinder full-bleed, demo chips
  horizontal scroll), 480px → touch targets ≥44px.

---

## 5. DOM contract with `app.js` (MUST NOT BREAK)

Required element IDs (every `$(…)` in `app.js`):

```
status · video · canvas · file-input · drop · camera-state · snap · flip ·
verdict-panel · v-chip · v-conf · v-meter · v-sev · v-advice · v-evidence ·
v-report · v-again · v-latency · v-error · history · refresh
```

Required classes / selectors:

- `.hidden` — toggled by JS on `#verdict-panel`, `#v-error`
- `.drop.over` — drag-over state on `#drop`
- `button.demo[data-sample="none|starch|detergent"]` — demo + `/?demo=` deep links
- history template (written by `app.js::loadHistory`): `.entry .chip[data-tone] .when`,
  empty state `.hint`, error `.error`
- `#v-meter` must remain a block element whose `style.width` is set by JS
  (the gauge mirrors it; the element itself keeps working)

Invariants asserted by `tests/test_frontend_contract.py`:

- every ID above exists in `index.html` exactly once
- `"SaafScan"` appears in the served HTML (also asserted by `tests/test_api.py`)
- no external `http(s)://` resource references (offline claim)
- every linked stylesheet/script exists on disk

Screenshot pack: `bash scripts/screenshot.sh` drives `/?demo=starch|detergent|none`
at 1360×1500 for the Tech4Food deck — layout keeps the 2-column composition.

---

## 6. Font

- **Fraunces** variable (opsz/wght 300–700, latin) + italic, self-hosted in
  `frontend/fonts/` with `OFL.txt` (SIL Open Font License 1.1).
- No CDN. `font-display: swap` + serif fallback stack (no FOIT).

## 7. Browser baseline

`:has()` (tone propagation, hero reflow) and `color-mix()` (severity badge,
gauge/meter tints) require **Chrome 111+ / Safari 16.4+ / Firefox 113+**.
Degradation is graceful: on older engines the capture hero stays in
column 1 (empty second column) and verdict tone colors fall back to jade
accent — severity is never conveyed by color alone (chip text, severity
badge label, and advice prose all carry it in text).
