"""PDF + QR report generation (fpdf2 + qrcode). ASCII-safe output."""

from __future__ import annotations

from io import BytesIO

import qrcode
from fpdf import FPDF

from saafscan.verdict import Verdict

_TONE = {
    "none": (25, 135, 84),
    "low": (180, 130, 20),
    "moderate": (190, 90, 20),
    "high": (180, 40, 40),
}


def _ascii(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _qr_png(payload: str) -> BytesIO:
    img = qrcode.make(payload)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class _Report(FPDF):
    pass


def build_report(
    verdict: Verdict,
    sample_name: str,
    timestamp: str,
    image_bytes: bytes | None = None,
) -> bytes:
    """Render a one-page PDF verdict report with a QR code. Pure function."""
    pdf = _Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 40, 70)
    pdf.cell(0, 12, _ascii("SaafScan - Test Report"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 100, 115)
    pdf.cell(0, 6, _ascii(f"Sample: {sample_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _ascii(f"Recorded: {timestamp}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Verdict banner
    r, g, b = _TONE.get(verdict.severity, (60, 60, 60))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    label = {
        "none": "NO ADULTERANT DETECTED",
        "starch": "STARCH ADULTERATED",
        "detergent": "DETERGENT ADULTERATED",
        "both": "MULTIPLE ADULTERANTS",
        "inconclusive": "INCONCLUSIVE - RE-TEST",
    }.get(verdict.adulterant, verdict.adulterant.upper())
    pdf.cell(0, 14, _ascii(label), fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(20, 40, 70)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)
    pdf.cell(
        0, 7, _ascii(f"Confidence: {verdict.confidence * 100:.0f}%"), new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(0, 7, _ascii(f"Severity: {verdict.severity.upper()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Advice", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _ascii(verdict.advice))
    pdf.ln(3)

    if verdict.evidence:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(
            0, 7, "Colour evidence (deltaE vs printed reference)", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.set_font("Courier", "", 9)
        for key, value in sorted(verdict.evidence.items()):
            pdf.cell(0, 6, _ascii(f"  {key}: {value:.2f}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    if image_bytes:
        try:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Sample image", new_x="LMARGIN", new_y="NEXT")
            pdf.image(BytesIO(image_bytes), w=70)
            pdf.ln(3)
        except Exception:  # noqa: BLE001 — image is optional decoration only
            pass

    # QR with machine-readable verdict summary
    payload = (
        f"SaafScan|{label}|conf={verdict.confidence:.2f}|"
        f"sev={verdict.severity}|{timestamp}|{sample_name}"
    )
    pdf.image(_qr_png(payload), x=160, y=262, w=30)
    pdf.set_xy(150, 294)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(90, 100, 115)
    pdf.cell(40, 4, "Scan to verify", align="C")

    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.cell(0, 6, "SaafScan student prototype - not a certified testing device.", align="C")
    return bytes(pdf.output())
