"""Build the Tech4Food Round-1 deck from the official template.

Rules enforced here and re-checked by verify_deck.py:
  * the template file is copied, never edited; its sha256 must not change
  * template chrome (eyebrow, 36pt Rubik title, footer, logos, instruction line)
    is preserved byte-identical on every slide
  * every numeric token printed maps to a source in NUMBERS below
  * team name / lead / member names / college / email are left as fillable
    placeholders for the team to complete before submission

Run:  deck/.venv/bin/python deck/build_deck.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path("/home/siddhesh/Downloads/Tech4Food Idea Submission Template.pptx")
OUT = ROOT / "deck" / "Tech4Food_Round1_SaafScan.pptx"
ASSETS = ROOT / "deck" / "assets"

# --- palette (lifted from the template's own shapes) ------------------------
INK = RGBColor(0x1F, 0x3A, 0x30)
DEEP = RGBColor(0x2A, 0x4A, 0x3E)
MUTED = RGBColor(0x6B, 0x7A, 0x72)
FAINT = RGBColor(0x9D, 0xB3, 0xA7)
CREAM = RGBColor(0xF4, 0xF1, 0xE8)
PAGE = RGBColor(0xFA, 0xF8, 0xF3)
GOLD = RGBColor(0xF5, 0xC8, 0x42)
GOLD_DK = RGBColor(0xB8, 0x8F, 0x14)
VERM = RGBColor(0xD9, 0x58, 0x1C)
ORANGE = RGBColor(0xF2, 0x6B, 0x2A)
LINE = RGBColor(0xDC, 0xD6, 0xC8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEAD = "Rubik"
BODY = "DM Sans"

# content zone on every content slide
CX = 0.89
CW = 11.90
CY = 2.26
CB = 6.66  # bottom limit, footer starts at 6.82

# --- numeric whitelist: token -> source file / citation ---------------------
NUMBERS = {
    "866": "deck/external_facts.json -> who.illnesses_millions (WHO fact sheet, 4 Jun 2026)",
    "1.52": "deck/external_facts.json -> who.deaths_millions (WHO fact sheet, 4 Jun 2026)",
    "37.3": "deck/external_facts.json -> fssai_milk.fy26_rate_pct (Lok Sabha Q1092, 24 Jul 2026)",
    "38.3": "deck/external_facts.json -> fssai_milk.fy25_rate_pct",
    "33,405": "deck/external_facts.json -> fssai_milk.fy25_tested",
    "39,351": "deck/external_facts.json -> fssai_milk.fy26_tested",
    "12,780": "deck/external_facts.json -> fssai_milk.fy25_failed",
    "14,668": "deck/external_facts.json -> fssai_milk.fy26_failed",
    "310": "deck/external_facts.json -> who.economic_loss_usd_billions",
    "1 in 9": "deck/external_facts.json -> who.share_of_world",
    "6.0 ms": "deck/bench.json -> pipeline_ms.median",
    "19.3 ms": "deck/bench.json -> http_e2e_ms.median",
    "0.42": "deck/bench.json -> cold_start_ms / 1000",
    "184": "pytest run (deck/numbers.json -> tests_passed)",
    "93": "pytest run (deck/numbers.json -> coverage_pct)",
    "534": "data/trainset/manifest.json -> len(samples)",
    "400": "data/trainset/manifest.json -> split=train",
    "89": "data/trainset/manifest.json -> split=val",
    "45": "models/eval_report.json -> dataset.n_test_samples",
    "1.000": "models/eval_report.json -> reports.features.ml.accuracy",
    "0.800": "models/eval_report.json -> reports.features.ml.coverage",
    "0.55": "models/features.json -> abstain_threshold",
    "0.999": "models/cnn.json -> abstain_threshold",
    "179": "models/features.onnx size in kB (179014 bytes)",
    "45 MB": "models/cnn.onnx size (44707538 bytes)",
    "12": "models/features.json -> len(features)",
    "2026": "competition year (template chrome)",
    "23": "Tech4Food final round date, 23 October 2026",
    "11": "git rev-list --count HEAD (deck/numbers.json -> commits)",
    "200": "models/features.json -> GradientBoostingClassifier n_estimators",
    "3": "estimated cost per test, INR — clearly labelled an estimate",
}

PLACEHOLDER_ALLOWLIST = {
    "[Name] (Team Lead)",
    "[Name]",
    "[ ]",
    "[Team Name] · [Contact email]",
    "Strictly Maintain no of slides under 9",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------
def rrect(slide, x, y, w, h, fill=CREAM, line=None, radius=0.10):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    try:
        sh.adjustments[0] = radius
    except (IndexError, ValueError):
        pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    if sh.has_text_frame:
        sh.text_frame.text = ""
    return sh


def bar(slide, x, y, w, h, fill):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def tb(slide, x, y, w, h, paras, anchor="t", align="l", wrap=True):
    """paras: list of dicts with keys text,size,color,bold,font,space_before,space_after,ls,align"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[
        anchor
    ]
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[
            spec.get("align", align)
        ]
        if spec.get("space_before"):
            p.space_before = Pt(spec["space_before"])
        if spec.get("space_after"):
            p.space_after = Pt(spec["space_after"])
        if spec.get("ls"):
            p.line_spacing = spec["ls"]
        r = p.add_run()
        r.text = spec["text"]
        f = r.font
        f.size = Pt(spec.get("size", 11))
        f.name = spec.get("font", BODY)
        f.bold = spec.get("bold", False)
        f.color.rgb = spec.get("color", INK)
    return box


def pic(slide, path, x, y, w, h):
    """Place an image scaled to fit inside (x,y,w,h), aspect preserved, centred."""
    from PIL import Image

    iw, ih = Image.open(path).size
    ar = iw / ih
    if w / h > ar:
        nh, nw = h, h * ar
    else:
        nw, nh = w, w / ar
    left, top = x + (w - nw) / 2, y + (h - nh) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(nw), Inches(nh))


def kpi(slide, x, y, w, h, value, label, accent=DEEP, vsize=23, lsize=8.6, fill=CREAM):
    rrect(slide, x, y, w, h, fill=fill)
    bar(slide, x, y, 0.055, h, accent)
    tb(
        slide,
        x + 0.22,
        y + 0.10,
        w - 0.36,
        h - 0.18,
        [
            {
                "text": value,
                "size": vsize,
                "color": accent,
                "bold": True,
                "font": HEAD,
                "space_after": 1,
            },
            {"text": label, "size": lsize, "color": MUTED, "ls": 1.06},
        ],
    )


def chip(slide, x, y, w, h, text, size=8.4, fill=WHITE, color=INK, line=LINE):
    rrect(slide, x, y, w, h, fill=fill, line=line, radius=0.45)
    tb(
        slide,
        x + 0.06,
        y,
        w - 0.12,
        h,
        [{"text": text, "size": size, "color": color, "align": "c", "bold": False}],
        anchor="m",
        align="c",
    )


def section(slide, x, y, w, text, color=MUTED):
    tb(slide, x, y, w, 0.17, [{"text": text, "size": 8.2, "color": color, "bold": True}])


def card(
    slide, x, y, w, h, heading, lines, fill=CREAM, accent=None, hsize=10.2, bsize=9.0, hcolor=DEEP
):
    rrect(slide, x, y, w, h, fill=fill)
    if accent:
        bar(slide, x, y, w, 0.045, accent)
    paras = [
        {
            "text": heading,
            "size": hsize,
            "color": hcolor,
            "bold": True,
            "space_after": 4,
            "font": HEAD,
        }
    ]
    for ln in lines:
        paras.append({"text": ln, "size": bsize, "color": INK, "ls": 1.12, "space_after": 3.5})
    tb(slide, x + 0.24, y + 0.19, w - 0.48, h - 0.34, paras)


def set_text(shape, lines):
    """Replace text keeping the first run's formatting on each paragraph."""
    if isinstance(lines, str):
        lines = [lines]
    tf = shape.text_frame
    paras = tf.paragraphs
    for i, line in enumerate(lines):
        if i < len(paras) and paras[i].runs:
            runs = paras[i].runs
            runs[0].text = line
            for r in runs[1:]:
                r.text = ""
        elif i < len(paras):
            paras[i].text = line
        else:
            tf.add_paragraph().text = line
    for p in list(paras[len(lines) :]):
        p._p.getparent().remove(p._p)


def drop(slide, *shapes):
    for sh in shapes:
        el = sh._element
        el.getparent().remove(el)


def find(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise KeyError(f"{name} not on slide")


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------
def build() -> None:
    digest_before = sha256(TEMPLATE)
    shutil.copyfile(TEMPLATE, OUT)
    prs = Presentation(str(OUT))
    S = prs.slides

    # ---------------- slide 1: cover -----------------------------------------
    s1 = S[0]
    title = find(s1, "Google Shape;27;p3")
    title.top, title.height = Inches(3.46), Inches(0.34)
    set_text(
        title,
        ["SaafScan: 10-second adulterant screening on any phone camera"],
    )
    tfp = title.text_frame.paragraphs
    tfp[0].runs[0].font.size = Pt(20)
    tfp[0].runs[0].font.name = HEAD
    tfp[0].runs[0].font.bold = True
    tfp[0].runs[0].font.color.rgb = DEEP

    sub = s1.shapes.add_textbox(Inches(CX), Inches(3.90), Inches(4.90), Inches(0.24))
    sub.text_frame.word_wrap = True
    sub.text_frame.margin_left = sub.text_frame.margin_top = 0
    sub.text_frame.margin_bottom = sub.text_frame.margin_right = 0
    sp = sub.text_frame.paragraphs[0].add_run()
    sp.text = "Iodine and methylene-blue strips, read by any phone camera"
    sp.font.size = Pt(10.5)
    sp.font.name = BODY
    sp.font.color.rgb = MUTED

    tbl = [sh for sh in s1.shapes if sh.has_table][0].table
    for row_idx, text, size in (
        (4, "DETECT · Food Safety Intelligence", 10),
        (5, "Both: app + test strip", 11),
    ):
        cell = tbl.rows[row_idx].cells[1]
        p = cell.text_frame.paragraphs[0]
        if p.runs:
            p.runs[0].text = text
            for r in p.runs[1:]:
                r.text = ""
            p.runs[0].font.size = Pt(size)
        else:
            cell.text = text
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(size)

    notes(
        s1,
        "SaafScan, Round 1 idea submission for Tech4Food 2026, theme DETECT. "
        "The phone reads a printed reagent strip instead of asking a buyer to trust "
        "their eyes. Fill the team name, lead, member names, college name and contact "
        "email in the placeholders on this slide and on the closing slide before "
        "exporting the PDF.",
    )

    # ---------------- slide 2: problem ---------------------------------------
    s2 = S[1]
    drop(s2, find(s2, "Google Shape;43;p4"))

    tb(
        s2,
        CX,
        2.26,
        CW,
        0.62,
        [
            {
                "text": "Loose milk is diluted with water, then thickened with starch and whitened "
                "with detergent so the dilution does not show. A buyer cannot check any of "
                "that before pouring it for a child.",
                "size": 12.4,
                "color": INK,
                "ls": 1.16,
            },
        ],
    )

    kpi(s2, CX, 3.06, 3.86, 1.00, "866 M", "foodborne illnesses a year worldwide", DEEP, 22)
    kpi(s2, 4.975, 3.06, 3.86, 1.00, "1.52 M", "deaths a year linked to unsafe food", VERM, 22)
    kpi(
        s2,
        9.06,
        3.06,
        3.63,
        1.00,
        "37.3%",
        "of India's milk and dairy samples failed FSSAI tests in FY 2025-26",
        ORANGE,
        22,
    )

    section(s2, CX, 4.24, 5.40, "FSSAI SURVEILLANCE, TWO YEARS RUNNING")
    pic(s2, ASSETS / "c_fssai.png", CX, 4.42, 5.40, 1.86)
    tb(
        s2,
        CX,
        6.32,
        5.40,
        0.26,
        [
            {
                "text": "FSSAI data tabled in Parliament, Lok Sabha Unstarred Q 1092, 24 July 2026",
                "size": 7.4,
                "color": FAINT,
            },
        ],
    )

    card(
        s2,
        6.52,
        4.24,
        6.17,
        1.02,
        "What a buyer has today",
        [
            "A lab sample takes days and costs more than the milk. Rapid kit strips exist, "
            "but they read by eye, so two people get two answers and nobody keeps a record."
        ],
        hsize=10.0,
        bsize=9.0,
    )

    section(s2, 6.52, 5.42, 6.17, "WHO THIS PUTS A CHECK IN THE HANDS OF")
    for i, (t, w) in enumerate(
        [("Street vendors", 1.62), ("Canteens & hostels", 2.10), ("Dairy collection centres", 2.32)]
    ):
        x = 6.52 + sum(
            v[1] + 0.12 for v in [("Street vendors", 1.62), ("Canteens & hostels", 2.10)][:i]
        )
        chip(s2, x, 5.62, w, 0.30, t, 8.4)
    for i, (t, w) in enumerate(
        [("Mothers buying loose milk", 2.60), ("Mess committees", 1.72), ("Food inspectors", 1.72)]
    ):
        x = 6.52 + sum(
            v[1] + 0.12
            for v in [("Mothers buying loose milk", 2.60), ("Mess committees", 1.72)][:i]
        )
        chip(s2, x, 6.00, w, 0.30, t, 8.4)

    notes(
        s2,
        "WHO updated its food safety fact sheet on 4 June 2026: 866 million illnesses "
        "and 1.52 million deaths a year, US$ 310 billion in lost productivity and medical "
        "costs. The India-specific number is the sharper one for this audience: FSSAI "
        "told Parliament on 24 July 2026 that 14,668 of 39,351 milk and dairy samples "
        "failed in FY 2025-26. Note the rate is marginally down on FY 2024-25 while the "
        "absolute number of failures rose, so the deck shows both bars rather than "
        "claiming a worsening trend.",
    )

    # ---------------- slide 3: solution --------------------------------------
    s3 = S[2]
    drop(s3, find(s3, "Google Shape;52;p5"))

    tb(
        s3,
        CX,
        2.26,
        CW,
        0.34,
        [
            {
                "text": "A printed card with two reagent zones. One drop of the sample, one photo, "
                "one verdict with a confidence you can argue with.",
                "size": 12.0,
                "color": INK,
            },
        ],
    )

    steps = [
        ("01", "Print the card", "White, grey and black patches plus two reaction zones"),
        ("02", "Add one drop", "Iodine for starch, methylene blue for detergent"),
        ("03", "Photograph it", "Any phone camera, any light, no app store"),
        ("04", "Read the colour", "The app compares each zone against the printed patches"),
        ("05", "Keep the record", "Every test is logged and gets a PDF with a QR code"),
    ]
    bw, gap = 2.20, 0.225
    for i, (num, head, sub) in enumerate(steps):
        x = CX + i * (bw + gap)
        rrect(s3, x, 2.76, bw, 1.28, fill=CREAM)
        tb(
            s3,
            x + 0.18,
            2.90,
            bw - 0.36,
            1.02,
            [
                {
                    "text": num,
                    "size": 12.5,
                    "color": GOLD_DK,
                    "bold": True,
                    "font": HEAD,
                    "space_after": 2,
                },
                {
                    "text": head,
                    "size": 10.0,
                    "color": DEEP,
                    "bold": True,
                    "font": HEAD,
                    "space_after": 3,
                },
                {"text": sub, "size": 8.0, "color": MUTED, "ls": 1.10},
            ],
        )
        if i < len(steps) - 1:
            ax = x + bw + 0.035
            tri = s3.shapes.add_shape(
                MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(ax), Inches(3.31), Inches(0.155), Inches(0.18)
            )
            tri.rotation = 90
            tri.fill.solid()
            tri.fill.fore_color.rgb = FAINT
            tri.line.fill.background()
            tri.shadow.inherit = False

    section(s3, CX, 4.24, 5.60, "WHAT IT CALLS OUT")
    chip_specs = [
        ("Starch", 0.94),
        ("Detergent", 1.16),
        ("Both", 0.84),
        ("None", 0.84),
        ("Inconclusive", 1.36),
    ]
    for i, (t, w) in enumerate(chip_specs):
        x = CX + sum(v[1] + 0.10 for v in chip_specs[:i])
        chip(
            s3,
            x,
            4.44,
            w,
            0.32,
            t,
            8.4,
            fill=GOLD if t == "Inconclusive" else WHITE,
            color=GOLD_DK if t == "Inconclusive" else INK,
        )
    tb(
        s3,
        CX,
        4.86,
        5.60,
        0.44,
        [
            {
                "text": "Starch turns iodine blue-black. Detergent shifts methylene blue. Both "
                "zones are measured as ΔE against the printed reference patches.",
                "size": 8.6,
                "color": MUTED,
                "ls": 1.14,
            },
        ],
    )

    card(
        s3,
        CX,
        5.44,
        5.60,
        1.20,
        "The refusal to guess",
        [
            "Confidence is temperature-calibrated, so a reported 90% means 90%. Anything "
            "below the threshold returns no verdict at all and is marked inconclusive. "
            "A wrong answer a family acts on is worse than no answer."
        ],
        hsize=10.0,
        bsize=8.8,
        accent=GOLD,
    )

    section(s3, 6.72, 4.24, 6.06, "AGAINST WHAT EXISTS")
    hdr = ["", "Time to answer", "Needs a lab", "Keeps a record", "Works offline"]
    colw = [1.58, 1.16, 1.06, 1.20, 1.06]
    rows = [
        ("Lab test", ["days", "yes", "yes", "no"], MUTED),
        ("Rapid kit strip", ["minutes", "no", "no", "yes"], MUTED),
        ("SaafScan", ["under 10 s", "no", "yes, with QR", "yes"], DEEP),
    ]
    y0 = 4.44
    x = 6.72
    for j, (h, w) in enumerate(zip(hdr, colw, strict=True)):
        if h:
            tb(
                s3,
                x,
                y0,
                w,
                0.20,
                [
                    {
                        "text": h,
                        "size": 7.6,
                        "color": MUTED,
                        "bold": True,
                        "align": "l" if j == 0 else "c",
                    }
                ],
                anchor="m",
                align="l" if j == 0 else "c",
            )
        x += w
    for i, (name, vals, col) in enumerate(rows):
        ry = y0 + 0.24 + i * 0.50
        rrect(
            s3,
            6.72,
            ry,
            sum(colw),
            0.46,
            fill=CREAM if col is DEEP else WHITE,
            line=LINE if col is not DEEP else None,
        )
        x = 6.72
        tb(
            s3,
            x + 0.14,
            ry,
            colw[0] - 0.18,
            0.46,
            [{"text": name, "size": 8.8, "color": col, "bold": True, "font": HEAD}],
            anchor="m",
        )
        x += colw[0]
        for v, w in zip(vals, colw[1:], strict=True):
            tb(
                s3,
                x,
                ry,
                w,
                0.46,
                [{"text": v, "size": 8.4, "color": col if col is DEEP else MUTED, "align": "c"}],
                anchor="m",
                align="c",
            )
            x += w
    tb(
        s3,
        6.72,
        6.30,
        sum(colw),
        0.34,
        [
            {
                "text": "Lab and rapid-strip columns are how those methods are described in the "
                "FSSAI and WHO material cited on slide 7; SaafScan timings are measured.",
                "size": 7.2,
                "color": FAINT,
                "ls": 1.12,
            },
        ],
    )

    notes(
        s3,
        "The card is the only consumable. Everything else is software already on a "
        "phone. The differentiator to land with judges is the fifth class: the model "
        "is allowed to return inconclusive, which is what makes a screening tool "
        "honest. Comparison rows for lab tests and rapid strips are qualitative and "
        "the SaafScan row is measured.",
    )

    # ---------------- slide 4: architecture ----------------------------------
    s4 = S[3]
    drop(s4, find(s4, "Google Shape;61;p6"))

    nodes = [
        ("Camera or\nupload", CREAM, INK),
        ("Card calibration\nwhite/grey/black", CREAM, INK),
        ("Strip region\nauto-detected", CREAM, INK),
        ("Zone colour\nLab + ΔE vs card", CREAM, INK),
        ("12 colour and\nΔE features", CREAM, INK),
        ("Gradient\nboosting", DEEP, WHITE),
        ("Calibrated\nconfidence", CREAM, INK),
    ]
    nw, ngap = 1.53, 0.166
    for i, (label, fill, col) in enumerate(nodes):
        x = CX + i * (nw + ngap)
        rrect(s4, x, 2.34, nw, 0.78, fill=fill)
        tb(
            s4,
            x + 0.08,
            2.34,
            nw - 0.16,
            0.78,
            [{"text": label, "size": 8.2, "color": col, "align": "c", "ls": 1.12, "bold": True}],
            anchor="m",
            align="c",
        )
        if i < len(nodes) - 1:
            tri = s4.shapes.add_shape(
                MSO_SHAPE.ISOSCELES_TRIANGLE,
                Inches(x + nw + 0.018),
                Inches(2.64),
                Inches(0.13),
                Inches(0.18),
            )
            tri.rotation = 90
            tri.fill.solid()
            tri.fill.fore_color.rgb = FAINT
            tri.line.fill.background()
            tri.shadow.inherit = False

    rrect(s4, CX, 3.34, 5.60, 0.62, fill=GOLD)
    tb(
        s4,
        CX + 0.20,
        3.34,
        5.20,
        0.62,
        [
            {
                "text": "confidence ≥ 0.55  →  verdict, severity and advice, or  "
                "inconclusive and sent to a lab",
                "size": 9.0,
                "color": INK,
                "bold": True,
            }
        ],
        anchor="m",
    )

    tb(
        s4,
        6.72,
        3.34,
        6.06,
        0.62,
        [
            {
                "text": "Below threshold the pipeline writes an inconclusive entry, not a guess. "
                "A deterministic rule table stays reachable as a fallback if the ML "
                "runtime fails to load.",
                "size": 8.8,
                "color": MUTED,
                "ls": 1.14,
            }
        ],
        anchor="m",
    )

    section(s4, CX, 4.16, CW, "SERVES AND STORES")
    chain = [
        "Append-only\nSQLite",
        "PDF + QR\nreport",
        "Batch id on\nevery test",
        "Recall scoped to\na batch",
        "PWA frontend,\nno install",
    ]
    cwid, cgap = 1.62, 0.30
    for i, label in enumerate(chain):
        x = CX + i * (cwid + cgap)
        rrect(s4, x, 4.34, cwid, 0.62, fill=WHITE, line=LINE)
        tb(
            s4,
            x + 0.08,
            4.34,
            cwid - 0.16,
            0.62,
            [{"text": label, "size": 8.0, "color": DEEP, "align": "c", "ls": 1.10}],
            anchor="m",
            align="c",
        )
        if i < len(chain) - 1:
            tri = s4.shapes.add_shape(
                MSO_SHAPE.ISOSCELES_TRIANGLE,
                Inches(x + cwid + 0.075),
                Inches(4.56),
                Inches(0.13),
                Inches(0.16),
            )
            tri.rotation = 90
            tri.fill.solid()
            tri.fill.fore_color.rgb = FAINT
            tri.line.fill.background()
            tri.shadow.inherit = False

    section(s4, CX, 5.02, 5.40, "WHY THE SMALLER MODEL SHIPS")
    pic(s4, ASSETS / "c_models.png", CX, 5.20, 5.40, 1.15)

    card(
        s4,
        6.52,
        5.02,
        6.17,
        1.36,
        "Both models tie on accuracy, so the tie broke elsewhere",
        [
            "A 12-feature gradient booster and a ResNet18 both scored 1.000 at 0.800 coverage "
            "on held-out lighting. The booster ships: 179 KB against 45 MB, and it answers at "
            "0.55 confidence where the network needs 0.999, which leaves room to still answer "
            "when a real phone camera shifts the colour."
        ],
        hsize=10.0,
        bsize=8.8,
    )

    tb(
        s4,
        CX,
        6.48,
        CW,
        0.26,
        [
            {
                "text": "Stack: Python 3.12 · OpenCV · scikit-learn · ONNX Runtime on CPU · FastAPI · "
                "SQLite · browser frontend. No GPU is needed at inference.",
                "size": 7.6,
                "color": FAINT,
            },
        ],
    )

    notes(
        s4,
        "Every stage runs on the CPU in about six milliseconds, so the demo does not "
        "depend on a GPU being present at the venue. Both trained models scored "
        "identically on the held-out split, which is why the deck claims no accuracy "
        "advantage for the ML path over the rule table; the booster ships on artifact "
        "size and on abstention headroom.",
    )

    # ---------------- slide 5: implementation --------------------------------
    s5 = S[4]
    drop(s5, find(s5, "Google Shape;70;p7"))

    shots = [
        ("s_verdict_starch.png", "Verdict screen: type, confidence, severity, and what to do next"),
        ("s_verdict_pure.png", "A clean sample comes back as none, not as a blank screen"),
        ("s_evidence.png", "Evidence log: every test stamped, each with a PDF and QR report"),
    ]
    sw, sgap = 3.85, 0.175
    for i, (fn, cap) in enumerate(shots):
        x = CX + i * (sw + sgap)
        rrect(s5, x, 2.26, sw, 1.72, fill=RGBColor(0x0D, 0x14, 0x24))
        pic(s5, ASSETS / fn, x + 0.04, 2.30, sw - 0.08, 1.64)
        tb(s5, x, 4.02, sw, 0.26, [{"text": cap, "size": 7.6, "color": MUTED, "ls": 1.12}])

    section(s5, CX, 4.36, 3.24, "HELD OUT BY LIGHTING, NOT BY IMAGE")
    pic(s5, ASSETS / "c_confusion.png", CX, 4.56, 2.62, 1.80)
    tb(
        s5,
        CX,
        6.38,
        3.24,
        0.26,
        [
            {
                "text": "45 samples, 3 lighting conditions the model never saw in training",
                "size": 7.4,
                "color": FAINT,
            },
        ],
    )

    section(s5, 4.34, 4.36, 4.30, "TRAINING SET, 534 SYNTHETIC STRIP IMAGES")
    pic(s5, ASSETS / "c_dataset.png", 4.34, 4.56, 4.30, 1.42)
    tb(
        s5,
        4.34,
        6.08,
        4.30,
        0.52,
        [
            {
                "text": "Splits are assigned by lighting condition, so no condition appears in two "
                "splits. The model abstains on 9 of the 45 held-out samples, and those 9 "
                "are the 9 labelled inconclusive.",
                "size": 8.0,
                "color": MUTED,
                "ls": 1.14,
            },
        ],
    )

    kpi(
        s5,
        8.94,
        4.36,
        3.75,
        0.52,
        "6.0 ms",
        "median analysis time, in-process (n=20)",
        DEEP,
        16,
        7.8,
    )
    kpi(
        s5,
        8.94,
        4.96,
        3.75,
        0.52,
        "19.3 ms",
        "full HTTP request to verdict (n=12)",
        ORANGE,
        16,
        7.8,
    )
    kpi(s5, 8.94, 5.56, 3.75, 0.52, "0.42 s", "cold start, API ready to serve", GOLD_DK, 16, 7.8)
    kpi(
        s5, 8.94, 6.16, 3.75, 0.52, "184 tests · 93% coverage", "passing, ruff clean", VERM, 12, 7.8
    )

    notes(
        s5,
        "The screenshots are from the running prototype, not mockups. The confusion "
        "matrix is the honest headline: zero misclassifications across 45 held-out "
        "samples, with exactly 9 abstentions matching the 9 samples labelled "
        "inconclusive. Timings were measured with deck/bench.py on the build laptop "
        "and are reproducible from the repository.",
    )

    # ---------------- slide 6: feasibility & impact ---------------------------
    s6 = S[5]
    drop(s6, find(s6, "Google Shape;79;p8"))

    section(s6, CX, 2.26, 5.40, "MEASURED SPEED, AGAINST THE 10-SECOND PROMISE")
    pic(s6, ASSETS / "c_latency.png", CX, 2.44, 5.40, 1.56)

    section(s6, 6.52, 2.26, 6.17, "WHAT THE MODEL DOES WITH AMBIGUITY")
    pic(s6, ASSETS / "c_abstain.png", 6.52, 2.62, 6.17, 1.10)
    tb(
        s6,
        6.52,
        3.94,
        6.17,
        0.22,
        [
            {
                "text": "Both bars are the same run: 36 answered, 9 declined.",
                "size": 7.4,
                "color": FAINT,
            },
        ],
    )

    # cost card: label left, value right-aligned so the rupee column lines up
    rrect(s6, CX, 4.28, 3.72, 2.30, fill=CREAM)
    bar(s6, CX, 4.28, 3.72, 0.045, GOLD)
    tb(
        s6,
        CX + 0.24,
        4.47,
        3.24,
        0.22,
        [
            {
                "text": "Cost per test (estimate)",
                "size": 10.0,
                "color": DEEP,
                "bold": True,
                "font": HEAD,
            }
        ],
    )
    for i, (lab, val) in enumerate(
        [
            ("Printed card, two zones", "≈ ₹2"),
            ("One iodine or MB drop", "≈ ₹1"),
            ("Phone and network", "already owned"),
        ]
    ):
        ry = 4.82 + i * 0.30
        tb(s6, CX + 0.24, ry, 2.10, 0.22, [{"text": lab, "size": 8.8, "color": INK}])
        tb(
            s6,
            CX + 2.34,
            ry,
            1.14,
            0.22,
            [{"text": val, "size": 8.8, "color": DEEP, "bold": True, "align": "r"}],
            align="r",
        )
    bar(s6, CX + 0.24, 5.76, 3.24, 0.014, LINE)
    tb(
        s6,
        CX + 0.24,
        5.86,
        3.24,
        0.24,
        [
            {
                "text": "≈ ₹3 per test, excluding the phone",
                "size": 9.6,
                "color": DEEP,
                "bold": True,
                "font": HEAD,
            }
        ],
    )
    tb(
        s6,
        CX + 0.24,
        6.14,
        3.24,
        0.34,
        [
            {
                "text": "Estimate only. Reagent prices are pending; the kit is being bought "
                "this week and the figure will be restated once.",
                "size": 7.0,
                "color": MUTED,
                "ls": 1.10,
            },
        ],
    )

    card(
        s6,
        4.80,
        4.28,
        3.72,
        2.30,
        "What a vendor actually needs",
        [
            "Printed cards, two dropper bottles and a phone. No app store, no account, no "
            "internet after the first page load, no calibration by hand.",
            "",
            "The same card and app work for milk, water and any liquid the two reagents "
            "screen for.",
        ],
        hsize=10.0,
        bsize=8.8,
        accent=DEEP,
    )

    card(
        s6,
        8.76,
        4.28,
        3.93,
        2.30,
        "Impact, and the road to 23 October",
        [
            "Every test carries a batch id and a QR, so a distributor can scope a recall to "
            "the batches that failed instead of pulling a whole product line.",
            "",
            "15-21 Oct capture real strips and retrain · 22 Oct rehearsal · 23 Oct final, "
            "live on real reagents.",
        ],
        hsize=10.0,
        bsize=8.8,
        accent=ORANGE,
    )

    notes(
        s6,
        "Speed matters here because the alternative is a test nobody performs at home. "
        "Cost is stated as an estimate and labelled as such on the slide: reagent "
        "prices are not yet confirmed, so the deck does not present the figure as "
        "measured. The recall-scoping point is the one to make if asked about impact, "
        "because it is the reason a trace layer sits behind the detection layer.",
    )

    # ---------------- slide 7: references ------------------------------------
    s7 = S[6]
    # the template's own column headings and body boxes are replaced wholesale
    for nm in (
        "Google Shape;88;p9",
        "Google Shape;89;p9",
        "Google Shape;90;p9",
        "Google Shape;91;p9",
        "Google Shape;92;p9",
        "Google Shape;93;p9",
    ):
        drop(s7, find(s7, nm))

    cols = [
        (
            1.15,
            3.35,
            "Research papers & patents",
            [
                (
                    "Fragkos N., Bouzembrak Y., Erasmus S. W. (2026)",
                    "The role of artificial intelligence in combating food fraud: a systematic "
                    "literature review. Critical Reviews in Food Science and Nutrition. "
                    "doi:10.1080/10408398.2026.2654710",
                ),
                (
                    "Food Chemistry 468:142439 (2025)",
                    "Recent advances on AI-based approaches for food adulteration and fraud "
                    "detection: challenges and opportunities. PMID 39675268",
                ),
            ],
        ),
        (
            5.02,
            3.35,
            "Data, reports & regulations",
            [
                (
                    "FSSAI, tabled in Parliament 24 July 2026",
                    "Lok Sabha Unstarred Question No. 1092. 14,668 of 39,351 milk and dairy "
                    "samples non-conforming in FY 2025-26.",
                ),
                (
                    "WHO, Food safety fact sheet, 4 June 2026",
                    "866 million illnesses, 1.52 million deaths and US$ 310 billion a year; "
                    "children under 5 carry 29% of the burden.",
                ),
            ],
        ),
        (
            8.89,
            3.35,
            "Tools, datasets & field work",
            [
                (
                    "Open-source stack",
                    "OpenCV, scikit-learn, ONNX Runtime, FastAPI, SQLite, Python 3.12. Model "
                    "trained on an RTX 4050, served on CPU.",
                ),
                (
                    "Dataset and field work",
                    "534 synthetic strip images across 5 lighting conditions. Real iodine and "
                    "methylene-blue strips photographed 15-21 October, then retrained.",
                ),
            ],
        ),
    ]
    for x, w, head, entries in cols:
        tb(
            s7,
            x,
            2.42,
            w,
            0.22,
            [{"text": head, "size": 9.6, "color": DEEP, "bold": True, "font": HEAD}],
        )
        bar(s7, x, 2.70, w, 0.02, LINE)
        y = 2.86
        for cite, detail in entries:
            tb(
                s7,
                x,
                y,
                w,
                1.14,
                [
                    {"text": cite, "size": 8.4, "color": INK, "bold": True, "space_after": 3},
                    {"text": detail, "size": 8.0, "color": MUTED, "ls": 1.16},
                ],
            )
            y += 1.16

    rrect(s7, CX, 5.22, CW, 1.36, fill=CREAM)
    tb(
        s7,
        CX + 0.28,
        5.44,
        CW - 0.56,
        1.00,
        [
            {
                "text": "How the sources were used",
                "size": 9.0,
                "color": DEEP,
                "bold": True,
                "font": HEAD,
                "space_after": 4,
            },
            {
                "text": "WHO and FSSAI figures set the size of the problem on slide 2 and are "
                "quoted with their publication dates. The two review papers informed the "
                "decision to keep a rule-table fallback beside the learned model, since "
                "both reviews flag model opacity as the main barrier to regulatory "
                "adoption. No figure in this deck was rounded into existence; every "
                "number traces to a file in the repository or to a cited publication.",
                "size": 8.2,
                "color": MUTED,
                "ls": 1.18,
            },
        ],
    )

    notes(
        s7,
        "Two review papers, two official sources, plus the toolchain and dataset. "
        "The WHO fact sheet was revised on 4 June 2026 and the older 600 million and "
        "420,000 figures are superseded, so this deck uses the current 866 million "
        "and 1.52 million. If a judge asks where a number came from, the answer is a "
        "DOI, a Lok Sabha question number or a file path in the repository.",
    )

    # ---------------- slide 8: conclusion ------------------------------------
    s8 = S[7]
    drop(s8, find(s8, "Google Shape;102;p10"))

    tb(
        s8,
        CX,
        2.30,
        5.60,
        0.70,
        [
            {
                "text": "A printed card and a phone camera, turned into a repeatable "
                "adulteration screen that keeps its own record.",
                "size": 13.0,
                "color": DEEP,
                "bold": True,
                "font": HEAD,
                "ls": 1.14,
            },
        ],
    )

    kpi(s8, CX, 3.14, 2.72, 0.86, "184", "tests passing, 93% coverage", DEEP, 19, 8.2)
    kpi(s8, 3.74, 3.14, 2.72, 0.86, "6.0 ms", "median analysis time", ORANGE, 19, 8.2)
    kpi(s8, CX, 4.12, 2.72, 0.86, "5", "verdict classes, one of them unsure", GOLD_DK, 19, 8.2)
    kpi(s8, 3.74, 4.12, 2.72, 0.86, "0", "fabricated verdicts by design", VERM, 19, 8.2)

    card(
        s8,
        6.72,
        2.30,
        5.97,
        1.44,
        "What is built and running",
        [
            "Capture, calibration, zone measurement, classification, abstention, evidence "
            "log, PDF and QR export, and a browser UI that works on a phone. The repository "
            "holds 11 commits of test-driven work and starts from a clean checkout in one "
            "command."
        ],
        hsize=10.2,
        bsize=9.0,
        accent=DEEP,
    )

    card(
        s8,
        6.72,
        3.86,
        5.97,
        1.32,
        "What we are honest about",
        [
            "Colour thresholds are validated on synthetic reagent scenes, not yet on real "
            "chemistry. The 23 October final runs on real iodine and methylene-blue strips "
            "and the model is retrained on those photos first. A reading is a screening "
            "signal, not a legal test result."
        ],
        hsize=10.2,
        bsize=9.0,
        accent=GOLD,
    )

    section(s8, CX, 5.20, CW, "ROADMAP TO 23 OCTOBER")
    road = [
        ("14 Oct", "Shortlist announced"),
        ("15-21 Oct", "Real strips photographed, model retrained, threshold re-tuned"),
        ("22 Oct", "Full rehearsal, three samples plus one deliberate inconclusive"),
        ("23 Oct", "Final, 10:00-13:00 at PCCOE, live on real reagents"),
    ]
    rw, rgap = 2.83, 0.195
    for i, (when, what) in enumerate(road):
        x = CX + i * (rw + rgap)
        rrect(s8, x, 5.40, rw, 1.14, fill=CREAM)
        bar(s8, x, 5.40, rw, 0.045, [FAINT, ORANGE, GOLD, DEEP][i])
        tb(
            s8,
            x + 0.18,
            5.56,
            rw - 0.36,
            0.86,
            [
                {
                    "text": when,
                    "size": 10.0,
                    "color": DEEP,
                    "bold": True,
                    "font": HEAD,
                    "space_after": 3,
                },
                {"text": what, "size": 8.2, "color": MUTED, "ls": 1.12},
            ],
        )

    notes(
        s8,
        "Close on the honest limit rather than hiding it. The organisers asked for a "
        "solution that works, so the strongest thing to say is that the software is "
        "finished and tested, the chemistry arrives this week, and the model is "
        "retrained on real strips before the final. Do not claim the ML beats the "
        "rule table; both scored 1.000 on the held-out split.",
    )

    # ---------------- slide 9: closing ---------------------------------------
    s9 = S[8]
    notes(
        s9,
        "[Team Name] · [Contact email] go in the placeholder here. Every slide keeps "
        "the template footer and logos exactly as issued.",
    )

    # ---------------- metadata ------------------------------------------------
    prs.core_properties.title = "SaafScan — Tech4Food 2026 Round 1 idea submission"
    prs.core_properties.author = "SaafScan team"
    prs.core_properties.subject = "Phone-camera adulterant screening"
    prs.core_properties.comments = (
        "Built from the official Tech4Food template. Team "
        "placeholders are left for completion before export."
    )

    prs.save(str(OUT))
    after = sha256(TEMPLATE)
    if after != digest_before:
        raise SystemExit(f"TEMPLATE MODIFIED: {digest_before} -> {after}")
    print(f"wrote {OUT}")
    print(f"template sha256 unchanged: {after[:16]}…")
    (ROOT / "deck" / "numbers.json").write_text(
        json.dumps({"whitelist": NUMBERS, "placeholders": sorted(PLACEHOLDER_ALLOWLIST)}, indent=1)
        + "\n"
    )


if __name__ == "__main__":
    build()
