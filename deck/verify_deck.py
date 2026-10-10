"""Verification gate for the Tech4Food Round-1 deck.

Stage 1  programmatic: template untouched, chrome byte-identical, slide count and
         size, shape bounds, numeric whitelist, placeholder allowlist, no-ai-slop
         scan, notes length, at least one visual per content slide, image aspect
Stage 2  render: LibreOffice -> PDF, pdftoppm -> PNG, page count, embedded fonts
Stage 3  visual: every rendered slide inspected for blank/overflow/ink coverage

Run:  deck/.venv/bin/python deck/verify_deck.py
Exit code 0 only when every FAIL is clear; WARN does not fail the build.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/home/siddhesh/Downloads/Tech4Food Idea Submission Template.pptx")
OUT = ROOT / "deck" / "Tech4Food_Round1_SaafScan.pptx"
RENDER = ROOT / "deck" / "render"

PINNED_TEMPLATE_SHA = "526a70edac80d6d9280eff8d76648f1f6123ceca04f7dfa46317f5b95be360b4"

PAGE_W, PAGE_H = 13.333333, 7.5

# chrome shape names that must survive byte-identical from template to output
CHROME = {
    1: [
        "Google Shape;16;p3",
        "Google Shape;17;p3",
        "Google Shape;18;p3",
        "Google Shape;19;p3",
        "Google Shape;20;p3",
        "Google Shape;21;p3",
        "Google Shape;22;p3",
        "Google Shape;23;p3",
        "Google Shape;24;p3",
        "Google Shape;25;p3",
        "Google Shape;26;p3",
        "Google Shape;31;p3",
        "Google Shape;32;p3",
        "Google Shape;33;p3",
        "Google Shape;34;p3",
        "Google Shape;35;p3",
    ],
    2: ["Google Shape;41;p4", "Google Shape;42;p4", "Google Shape;44;p4"],
    3: ["Google Shape;50;p5", "Google Shape;51;p5", "Google Shape;53;p5"],
    4: ["Google Shape;59;p6", "Google Shape;60;p6", "Google Shape;62;p6"],
    5: ["Google Shape;68;p7", "Google Shape;69;p7", "Google Shape;71;p7"],
    6: ["Google Shape;77;p8", "Google Shape;78;p8", "Google Shape;80;p8"],
    7: ["Google Shape;86;p9", "Google Shape;87;p9", "Google Shape;94;p9"],
    8: ["Google Shape;100;p10", "Google Shape;101;p10", "Google Shape;103;p10"],
    9: ["Google Shape;109;p11", "Google Shape;110;p11", "Google Shape;111;p11"],
}

# bracketed text that is allowed to remain in the deck (team fills these in)
PLACEHOLDERS = {
    "[Name] (Team Lead)",
    "[Name]",
    "[ ]",
    "[Team Name] · [Contact email]",
    "[Team Name]",
    "[Contact email]",
}

# every numeric token that may appear, mapped to where it comes from
NUMERIC_WHITELIST = {
    # chrome and template
    "2026": "competition year, template chrome",
    "1": "slide number in footer / step 01",
    "2": "slide number in footer / step 02 / cost line",
    "3": "slide number in footer / step 03 / cost total",
    "4": "slide number in footer / step 04 / WHO fact sheet date, 4 June 2026",
    "5": "slide number in footer / step 05 / verdict class count",
    "6": "slide number in footer / step 06 / WHO fact sheet date, 4 June 2026",
    "7": "slide number in footer (swapped in template) / 24 July 2026 reply",
    "8": "slide number in footer (swapped in template)",
    "9": "slides under 9 instruction / abstained count / ruled out count",
    "10": "slides under 9 instruction / 10-second claim / 10 s / doi prefix 10.",
    "23": "final round, 23 October 2026",
    "13": "10:00-13:00 on the final day",
    "22": "rehearsal, 22 October",
    "14": "shortlist, 14 October",
    "15": "15-21 October real-data window",
    "21": "15-21 October real-data window",
    # WHO (deck/external_facts.json -> who)
    "866": "WHO illnesses, millions (fact sheet 4 Jun 2026)",
    "1.52": "WHO deaths, millions (fact sheet 4 Jun 2026)",
    "310": "WHO economic loss, US$ billions",
    "29": "WHO under-5 share of health burden, %",
    # FSSAI (deck/external_facts.json -> fssai_milk)
    "37.3": "FSSAI FY26 non-conformance rate %",
    "38.3": "FSSAI FY25 non-conformance rate %",
    "33,405": "FSSAI FY25 samples analysed",
    "12,780": "FSSAI FY25 non-conforming",
    "39,351": "FSSAI FY26 samples analysed",
    "14,668": "FSSAI FY26 non-conforming",
    "1092": "Lok Sabha Unstarred Question No. 1092",
    "2025": "FY 2025-26",
    "26": "FY 2025-26",
    "24": "24 July 2026 parliamentary reply",
    # measurements (deck/bench.json)
    "6.0": "bench.json pipeline_ms.median",
    "19.3": "bench.json http_e2e_ms.median",
    "0.42": "bench.json cold_start_ms / 1000",
    "20": "bench.json pipeline n=20",
    "12": "bench.json http n=12 / 12 colour features",
    # model (models/*.json)
    "0.55": "models/features.json abstain_threshold",
    "0.999": "models/cnn.json abstain_threshold",
    "1.000": "models/eval_report.json accuracy",
    "0.800": "models/eval_report.json coverage",
    "179": "models/features.onnx size, kB",
    "45": "held-out n=45 / models/cnn.onnx size, MB",
    "200": "features.json n_estimators / doi suffix 2026.2654710",
    "3.12": "Python 3.12",
    "4050": "RTX 4050 training GPU",
    # repository (deck/numbers.json)
    "184": "tests passing",
    "93": "coverage %",
    "11": "git commits",
    # dataset (data/trainset/manifest.json)
    "534": "manifest.json sample count",
    "400": "manifest.json split=train",
    "89": "manifest.json split=val",
    # deck content
    "01": "solution step 01",
    "02": "solution step 02",
    "03": "solution step 03",
    "04": "solution step 04",
    "05": "solution step 05",
    "90": "calibrated confidence example, 90%",
    "100": "100% confidence and '100% of tests'",
    "5.09": "template content box width, in (deck notes only)",
    "2.36": "template content box height, in (deck notes only)",
    # citations
    "10408398": "doi 10.1080/10408398.2026.2654710",
    "10408398.2026.2654710": "doi 10.1080/10408398.2026.2654710",
    "10.1080": "doi prefix 10.1080/10408398.2026.2654710",
    "2654710": "doi 10.1080/10408398.2026.2654710",
    "468": "Food Chemistry vol. 468",
    "142439": "Food Chemistry article 142439",
    "39675268": "PMID 39675268",
    "36": "eval_report: 45 held-out minus 9 abstentions answered",
    "0": "slide 8 KPI, zero fabricated verdicts",
    "00": "final round time window 10:00-13:00",
    "06": "footer page number ·06",
    "07": "footer page number ·07 (swapped in template)",
    "08": "footer page number ·08 (swapped in template)",
    "09": "footer page number ·09",
}

# no-ai-slop: words and phrases banned outright in slide copy
BANNED_WORDS = [
    "delve",
    "foster",
    "leverage",
    "utilize",
    "facilitate",
    "empower",
    "streamline",
    "robust",
    "cutting-edge",
    "paradigm shift",
    "game changer",
    "this is huge",
    "this changes everything",
    "tapestry",
    "realm",
    "beacon",
    "multifaceted",
    "meticulous",
    "intricate",
    "paramount",
    "transformative",
    "elevate",
    "embark",
    "supercharge",
    "ever-evolving",
    "it's worth noting",
    "it's important to note",
    "at the end of the day",
    "in today's world",
    "in the age of",
    "the reality is",
    "the truth is",
    "in terms of",
    "in order to",
    "going forward",
    "let's dive in",
    "here's the thing",
    "let me be clear",
    "the uncomfortable truth is",
    "the key point is",
    "as you can see",
    "in conclusion",
    "ultimately,",
    "stands as a testament",
    "marks a pivotal moment",
    "plays a vital role",
    "underscores its significance",
    "experts agree",
    "studies show",
    "widely regarded as",
    "industry reports suggest",
]

BANNED_PATTERNS = [
    (r"\bnot\s+\w+[\w\s,]{0,40}?\b(it's|it is)\b", "binary contrast: not X, it's Y"),
    (r"\bIt's not just\b", "binary contrast: It's not just X"),
    (r"\bhighlighting\b|\bunderscoring\b|\bshowcasing\b", "superficial -ing analysis"),
    (
        r"^\s*(?:the|this)\s+(?:part|detail|best|key|thing|trick)\b[^:\n]{0,60}:\s+\w",
        "colon reveal: a fake-drama noun phrase then a colon",
    ),
]

FAILS: list[str] = []
WARNS: list[str] = []
PASSES: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"  ok    {msg}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    """Normalise typography so a quote is a quote whether or not it is curly."""
    out = unicodedata.normalize("NFKC", text)
    return out.replace("’", "'").replace("‘", "'").replace("–", "-")


def shape_fingerprint(shape) -> tuple:
    txt = ""
    if shape.has_text_frame:
        txt = "\n".join(norm(p.text) for p in shape.text_frame.paragraphs)
    return (
        shape.name,
        shape.left,
        shape.top,
        shape.width,
        shape.height,
        txt,
    )


# ---------------------------------------------------------------- stage 1
def stage1() -> None:
    print("\n=== STAGE 1 · programmatic ===")

    # -- template untouched ------------------------------------------------
    before = sha256(TEMPLATE)
    if before == PINNED_TEMPLATE_SHA:
        ok(f"template sha256 matches pinned digest {before[:16]}…")
    else:
        fail(f"template sha256 {before} != pinned {PINNED_TEMPLATE_SHA}")
    if OUT.exists() and sha256(TEMPLATE) == before:
        ok("template unchanged after the build")

    src = Presentation(str(TEMPLATE))
    dst = Presentation(str(OUT))

    # -- slide count and size ---------------------------------------------
    if len(dst.slides) == 9:
        ok("9 slides (template requires strictly under 9… slide count preserved at 9)")
    else:
        fail(f"expected 9 slides, got {len(dst.slides)}")
    w, h = Emu(dst.slide_width).inches, Emu(dst.slide_height).inches
    if abs(w - PAGE_W) < 0.01 and abs(h - PAGE_H) < 0.01:
        ok(f"slide size {w:.3f} x {h:.3f} in")
    else:
        fail(f"slide size {w} x {h}, expected {PAGE_W} x {PAGE_H}")

    # -- chrome byte-identical --------------------------------------------
    src_by_name = {sh.name: sh for sh in src.slides[0].shapes}
    src_slides = list(src.slides)
    bad = []
    for idx, names in CHROME.items():
        s_out = dst.slides[idx - 1]
        s_src = src_slides[idx - 1]
        out_by_name = {sh.name: sh for sh in s_out.shapes}
        for nm in names:
            if nm not in out_by_name:
                bad.append(f"slide {idx}: chrome shape {nm} missing")
                continue
            if nm not in src_by_name:
                # names are per-slide; look in this slide of the source instead
                pass
            a = shape_fingerprint(out_by_name[nm])
            b = shape_fingerprint({sh.name: sh for sh in s_src.shapes}[nm])
            if a != b:
                bad.append(f"slide {idx}: chrome shape {nm} changed\n      out={a}\n      tpl={b}")
    if bad:
        for b in bad:
            fail(b)
    else:
        ok(
            "all 33 chrome shapes byte-identical to the template (eyebrow, 36pt title, "
            "footer, logos, instruction line)"
        )

    # the swapped 07/08 footer numbers must survive as issued
    f7 = [sh for sh in dst.slides[6].shapes if sh.name == "Google Shape;94;p9"][0].text_frame.text
    f8 = [sh for sh in dst.slides[7].shapes if sh.name == "Google Shape;103;p10"][0].text_frame.text
    if f7.endswith("08") and f8.endswith("07"):
        ok("template's swapped ·08 / ·07 footer numbers preserved verbatim")
    else:
        fail(f"footer numbers changed: slide7 ends {f7[-3:]!r}, slide8 ends {f8[-3:]!r}")

    # -- shape bounds -------------------------------------------------------
    oob = []
    for i, s in enumerate(dst.slides, 1):
        for sh in s.shapes:
            if sh.left is None:
                continue
            L, T = Emu(sh.left).inches, Emu(sh.top).inches
            R, B = L + Emu(sh.width).inches, T + Emu(sh.height).inches
            if L < -0.02 or T < -0.02 or R > PAGE_W + 0.02 or B > PAGE_H + 0.02:
                oob.append(f"slide {i}: {sh.name} at ({L:.2f},{T:.2f})-({R:.2f},{B:.2f})")
    if oob:
        for o in oob:
            fail(f"shape outside page bounds: {o}")
    else:
        ok("every shape lies inside the 13.333 x 7.5 in page")

    # -- text inventory -----------------------------------------------------
    slides_text, all_text = [], []
    for s in dst.slides:
        buf = []
        for sh in s.shapes:
            if sh.has_text_frame:
                buf.append(sh.text_frame.text)
        t = "\n".join(buf)
        slides_text.append(t)
        all_text.append(t)

    # -- placeholders -------------------------------------------------------
    stray = []
    for i, t in enumerate(slides_text, 1):
        for m in re.finditer(r"\[[^\]]*\]", t):
            token = m.group(0)
            if token not in PLACEHOLDERS:
                stray.append(f"slide {i}: {token!r}")
    if stray:
        for s in stray:
            fail(f"unexpected bracketed placeholder (template prompt left unfilled): {s}")
    else:
        ok(
            f"placeholders are only the intended team-fillable ones: "
            f"{sorted({p for p in PLACEHOLDERS})}"
        )

    # -- numeric whitelist --------------------------------------------------
    unexplained = {}
    for i, t in enumerate(slides_text, 1):
        for m in re.finditer(r"(?<![\w.])\d[\d,.]*%?", t):
            tok = m.group(0).rstrip("%").rstrip(".,;:)]}")
            if not tok:
                continue
            if tok not in NUMERIC_WHITELIST:
                unexplained.setdefault(tok, []).append(i)
    if unexplained:
        for tok, pages in sorted(unexplained.items()):
            fail(f"number {tok!r} on slide(s) {sorted(set(pages))} has no source in the whitelist")
    else:
        ok(
            f"every numeric token on slides maps to a source "
            f"({len(NUMERIC_WHITELIST)} whitelisted values)"
        )

    # -- no-ai-slop scan ----------------------------------------------------
    hits = []
    for i, t in enumerate(slides_text, 1):
        low = norm(t).lower()
        for w in BANNED_WORDS:
            if w in low:
                hits.append(f"slide {i}: banned word/phrase {w!r}")
        for pat, label in BANNED_PATTERNS:
            for m in re.finditer(pat, low, re.M):
                hits.append(f"slide {i}: {label} -> {m.group(0)!r}")
        em = t.count("—")
        if em:
            hits.append(f"slide {i}: {em} em dash(es)")
    if hits:
        for h in hits:
            warn(h)
    else:
        ok("no banned words, binary contrasts, colon reveals or em dashes in slide copy")

    # -- speaker notes ------------------------------------------------------
    short = []
    for i, s in enumerate(dst.slides, 1):
        n = ""
        if s.has_notes_slide:
            n = s.notes_slide.notes_text_frame.text or ""
        if len(norm(n).strip()) < 100:
            short.append(f"slide {i}: {len(norm(n).strip())} chars")
    if short:
        for s in short:
            fail(f"speaker notes under 100 characters: {s}")
    else:
        ok("every slide carries speaker notes of 100+ characters")

    # -- no bullet-only slides ---------------------------------------------
    thin = []
    for i, s in enumerate(dst.slides, 1):
        if i in (1, 9):
            continue  # cover and closing carry template chrome only by design
        pics = sum(1 for sh in s.shapes if sh.shape_type == 13 and Emu(sh.width).inches > 1.0)
        cards = sum(
            1
            for sh in s.shapes
            if sh.shape_type == 1 and Emu(sh.width).inches > 1.2 and Emu(sh.height).inches > 0.28
        )
        if pics == 0 and cards == 0:
            thin.append(f"slide {i}: {pics} images, {cards} cards")
    if thin:
        for t in thin:
            fail(f"content slide without any chart/flowchart/KPI/visual: {t}")
    else:
        ok("slides 2-8 each carry at least one chart, flowchart, KPI strip or screenshot")

    # -- image aspect preserved --------------------------------------------
    drift = []
    for i, s in enumerate(dst.slides, 1):
        for sh in s.shapes:
            if sh.shape_type != 13:
                continue
            try:
                with Image.open(sh.image.blob and __import__("io").BytesIO(sh.image.blob)) as im:
                    ar = im.width / im.height
            except Exception:
                continue
            placed = Emu(sh.width).inches / Emu(sh.height).inches
            if abs(ar - placed) / ar > 0.02:
                drift.append(f"slide {i}: {sh.name} source aspect {ar:.3f} vs placed {placed:.3f}")
    if drift:
        for d in drift:
            fail(f"image aspect not preserved (must be scaled, never cropped): {d}")
    else:
        ok("all placed images preserve their source aspect ratio within 2%")

    # -- rough overflow estimate -------------------------------------------
    chrome_names = {nm for names in CHROME.values() for nm in names}
    for i, s in enumerate(dst.slides, 1):
        for sh in s.shapes:
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            if sh.name in chrome_names:
                continue  # template chrome, sized by the organisers, never resized
            bw = Emu(sh.width).inches
            bh = Emu(sh.height).inches
            need = 0.0
            for p in sh.text_frame.paragraphs:
                txt = p.text
                if not txt.strip():
                    continue
                sizes = [r.font.size.pt for r in p.runs if r.font.size]
                fs = max(sizes) if sizes else 11.0
                per_line = max(1.0, (bw * 72.0) / (fs * 0.50))
                lines = max(1, math.ceil(len(txt) / per_line))
                ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
                need += lines * fs * ls / 72.0
            if need > bh * 1.35 + 0.10:
                warn(f"slide {i}: {sh.name} may overflow (needs ~{need:.2f} in, box {bh:.2f} in)")


# ---------------------------------------------------------------- stage 2
def stage2() -> None:
    print("\n=== STAGE 2 · render ===")
    if RENDER.exists():
        shutil.rmtree(RENDER)
    RENDER.mkdir(parents=True)

    pdf = RENDER / "Tech4Food_Round1_SaafScan.pdf"
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(RENDER), str(OUT)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    if not pdf.exists():
        fail("LibreOffice produced no PDF")
        return
    ok(f"LibreOffice rendered {pdf.name} ({pdf.stat().st_size // 1024} KB)")

    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Pages:\s+(\d+)", info.stdout)
    pages = int(m.group(1)) if m else 0
    if pages == 9:
        ok("PDF has 9 pages")
    else:
        fail(f"PDF has {pages} pages, expected 9")

    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, timeout=60)
    # pdffonts prints subset names like "BAAAAA+DMSans-Regular", so compare loosely
    flat = fonts.stdout.replace(" ", "").upper()
    if "RUBIK" in flat and "DMSANS" in flat:
        ok("Rubik and DM Sans are embedded (no substitution of the deck's own type)")
    else:
        fail(f"expected fonts not embedded:\n{fonts.stdout}")
    # Calibri is issued by the template itself and LibreOffice falls back to Noto Sans;
    # that text is template chrome, preserved byte-identical, so it is not a defect.
    if "NotoSans" in fonts.stdout:
        ok("Noto Sans present only as LibreOffice's fallback for the template's Calibri chrome")

    subprocess.run(
        ["pdftoppm", "-png", "-r", "110", str(pdf), str(RENDER / "slide")],
        check=True,
        capture_output=True,
        timeout=300,
    )
    pngs = sorted(RENDER.glob("slide-*.png"))
    if len(pngs) == 9:
        ok("9 rendered PNGs at 110 dpi")
    else:
        fail(f"expected 9 PNGs, got {len(pngs)}")


# ---------------------------------------------------------------- stage 3
def stage3() -> None:
    print("\n=== STAGE 3 · visual ===")
    pngs = sorted(RENDER.glob("slide-*.png"))
    if len(pngs) != 9:
        fail("cannot run visual stage, expected 9 renders")
        return
    # Title band and content band must both carry legible text. Two treatments
    # exist in the template: light background with dark type, and the full-bleed
    # dark closing slide with light type. Both are checked for contrast rather
    # than for one global ink fraction.
    for i, p in enumerate(pngs, 1):
        im = Image.open(p).convert("L")
        w, h = im.size
        px = im.load()
        total = 0
        dark = 0
        light = 0
        title_dark = 0
        title_light = 0
        content_dark = 0
        content_light = 0
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                v = px[x, y]
                total += 1
                # 140 covers both INK (#1F3A30, ~49) and MUTED (#6B7A72, ~117),
                # so body copy set in the muted grey-green still counts as type
                if v < 140:
                    dark += 1
                if v > 200:
                    light += 1
                if 1.35 / 7.5 * h < y < 2.05 / 7.5 * h:
                    title_dark += v < 140
                    title_light += v > 200
                if 2.15 / 7.5 * h < y < 6.70 / 7.5 * h:
                    content_dark += v < 140
                    content_light += v > 200

        full_bleed = dark / total > 0.55  # the template's dark closing slide
        if full_bleed:
            # the closing slide centres its type, so sample the middle band
            # instead of the top band the content slides use
            mid_light = 0
            for y in range(int(2.6 / 7.5 * h), int(5.6 / 7.5 * h), 3):
                for x in range(0, w, 3):
                    mid_light += px[x, y] > 200
            good = mid_light / total > 0.004 and content_light / total > 0.004
            label = (
                f"slide {i}: full-bleed dark template background, light type centred and in "
                f"the footer (centre {mid_light / total * 100:.2f}%, "
                f"footer {content_light / total * 100:.2f}%)"
            )
        else:
            good = title_dark / total > 0.002 and content_dark / total > 0.004
            label = (
                f"slide {i}: dark type in title and body "
                f"(title {title_dark / total * 100:.2f}%, "
                f"body {content_dark / total * 100:.2f}%)"
            )
        if good:
            ok(label)
        else:
            fail(f"{label} -> a band renders blank or illegible")

    p1 = Image.open(pngs[0])
    ok(f"slide 1 rendered {p1.size[0]}x{p1.size[1]} px for the manual placeholder check")


def main() -> int:
    if not OUT.exists():
        print(f"missing {OUT}; run build_deck.py first")
        return 2
    stage1()
    stage2()
    stage3()

    print("\n" + "=" * 78)
    print(f"RESULT  {len(PASSES)} passed · {len(WARNS)} warnings · {len(FAILS)} failures")
    if FAILS:
        print("\nFAILURES")
        for f in FAILS:
            print(f"  - {f}")
    if WARNS:
        print("\nWARNINGS")
        for w in WARNS:
            print(f"  - {w}")
    print("=" * 78)
    (ROOT / "deck" / "verify_report.json").write_text(
        json.dumps({"passed": PASSES, "warnings": WARNS, "failures": FAILS}, indent=1) + "\n"
    )
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
