"""RED tests: evidence log (append-only SQLite) and PDF report generation."""

from __future__ import annotations

import pytest

from saafscan.evidence import EvidenceStore
from saafscan.report import build_report
from saafscan.verdict import Verdict


@pytest.fixture()
def store(tmp_path) -> EvidenceStore:
    return EvidenceStore(db_path=tmp_path / "evidence.db")


def _verdict(adulterant: str = "starch") -> Verdict:
    return Verdict(
        adulterant=adulterant,
        confidence=0.91,
        severity="moderate",
        advice="Do not use this sample. Re-test with a fresh strip.",
        evidence={"starch": 61.5},
    )


# --- evidence store ---------------------------------------------------------


def test_record_then_list_roundtrip(store: EvidenceStore) -> None:
    vid = store.record(_verdict(), sample_name="milk-canteen", image_bytes=b"\x89PNG-fake")
    entries = store.list(limit=10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == vid
    assert entry.verdict.adulterant == "starch"
    assert entry.sample_name == "milk-canteen"
    assert entry.timestamp


def test_list_is_newest_first(store: EvidenceStore) -> None:
    first = store.record(_verdict("none"), sample_name="a")
    second = store.record(_verdict("starch"), sample_name="b")
    ids = [e.id for e in store.list()]
    assert ids == [second, first]


def test_list_respects_limit(store: EvidenceStore) -> None:
    for i in range(5):
        store.record(_verdict(), sample_name=f"s{i}")
    assert len(store.list(limit=2)) == 2


def test_get_returns_none_for_unknown_id(store: EvidenceStore) -> None:
    assert store.get("nope") is None


def test_get_roundtrips_full_entry(store: EvidenceStore) -> None:
    vid = store.record(_verdict("both"), sample_name="x", image_bytes=b"imgdata")
    entry = store.get(vid)
    assert entry is not None
    assert entry.verdict.adulterant == "both"
    assert entry.image_bytes == b"imgdata"


def test_store_is_append_only_no_update_or_delete_api(store: EvidenceStore) -> None:
    """Evidence must be tamper-evident: no mutation API by design."""
    assert not hasattr(store, "update")
    assert not hasattr(store, "delete")
    assert not hasattr(store, "remove")


def test_store_survives_reopen(store: EvidenceStore, tmp_path) -> None:
    vid = store.record(_verdict(), sample_name="persist")
    reopened = EvidenceStore(db_path=tmp_path / "evidence.db")
    assert reopened.get(vid) is not None


# --- report -----------------------------------------------------------------


def test_build_report_returns_valid_pdf_bytes() -> None:
    pdf = build_report(
        _verdict("starch"),
        sample_name="hostel-mess-milk",
        timestamp="2026-10-10T18:00:00+05:30",
        image_bytes=None,
    )
    assert isinstance(pdf, bytes)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 800


def test_report_embeds_verdict_text() -> None:
    pdf = build_report(
        _verdict("detergent"), sample_name="s1", timestamp="2026-10-10T18:00:00+05:30"
    )
    # fpdf2 compresses streams; ensure at minimum it is a well-formed multi-object PDF.
    assert pdf.count(b"obj") >= 3
    assert b"%%EOF" in pdf


def test_report_handles_missing_image_gracefully() -> None:
    pdf = build_report(
        _verdict("none"),
        sample_name="no-image",
        timestamp="2026-10-10T18:00:00+05:30",
        image_bytes=None,
    )
    assert pdf[:5] == b"%PDF-"


def test_report_embeds_sample_image_when_provided() -> None:
    import cv2
    import numpy as np

    img = np.full((120, 120, 3), (60, 140, 200), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    pdf = build_report(
        _verdict(),
        sample_name="with-image",
        timestamp="2026-10-10T18:00:00+05:30",
        image_bytes=buf.tobytes(),
    )
    assert pdf[:5] == b"%PDF-"


def test_report_survives_long_sample_names() -> None:
    pdf = build_report(
        _verdict(),
        sample_name="a" * 300,
        timestamp="2026-10-10T18:00:00+05:30",
    )
    assert pdf[:5] == b"%PDF-"
