"""RED tests: FastAPI surface — /analyze, /history, /report, /health, static."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from saafscan.api import app
from saafscan.synthetic import render_random_photo, render_sample


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh client with an isolated evidence store per test."""
    from saafscan import evidence

    monkeypatch.setattr(evidence, "DB_PATH", tmp_path / "evidence.db")
    with TestClient(app) as c:
        yield c


def _jpeg(img: np.ndarray) -> bytes:
    import cv2

    # cv2 encodes channel-first-BGR; our scenes are RGB, so swap on the way in.
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1])
    assert ok
    return buf.tobytes()


def _png(img: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", img[:, :, ::-1])
    assert ok
    return buf.tobytes()


# --- health -----------------------------------------------------------------


def test_health_reports_ready(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "ready"}
    assert body["models_loaded"] is True


# --- analyze ----------------------------------------------------------------


@pytest.mark.parametrize("label", ["none", "starch", "detergent", "both"])
def test_analyze_returns_verdict_for_each_sample(client: TestClient, label: str) -> None:
    r = client.post(
        "/analyze",
        files={"file": ("sample.jpg", _jpeg(render_sample(label, seed=7)), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"]["adulterant"] == label
    assert 0.0 <= body["verdict"]["confidence"] <= 1.0
    assert body["verdict"]["severity"] in {"none", "low", "moderate", "high"}
    assert isinstance(body["id"], str) and body["id"]


def test_analyze_rejects_random_photo_with_422(client: TestClient) -> None:
    r = client.post(
        "/analyze", files={"file": ("selfie.png", _png(render_random_photo(seed=11)), "image/png")}
    )
    assert r.status_code == 422
    detail = r.json()["detail"].lower()
    assert "card" in detail or "not appear" in detail


def test_analyze_rejects_non_image_payload(client: TestClient) -> None:
    r = client.post(
        "/analyze", files={"file": ("notes.txt", b"dear judge, please pass me", "text/plain")}
    )
    assert r.status_code == 422


def test_analyze_rejects_empty_file(client: TestClient) -> None:
    r = client.post("/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 422


def test_analyze_rejects_oversized_upload(client: TestClient) -> None:
    big = np.full((4000, 4000, 3), 128, dtype=np.uint8)
    payload = _png(big)
    # If already under the cap, pad with a synthetic over-limit payload.
    if len(payload) < 12 * 1024 * 1024:
        payload = payload + b"\x00" * (12 * 1024 * 1024 - len(payload) + 1)
    r = client.post("/analyze", files={"file": ("huge.png", payload, "image/png")})
    assert r.status_code in {413, 422}


def test_analyze_requires_the_file_field(client: TestClient) -> None:
    r = client.post("/analyze", data={})
    assert r.status_code == 422


def test_analyze_rejects_bad_http_method(client: TestClient) -> None:
    assert client.get("/analyze").status_code == 405


# --- history ----------------------------------------------------------------


def test_history_starts_empty_then_records_analyses(client: TestClient) -> None:
    r0 = client.get("/history")
    assert r0.status_code == 200
    assert r0.json()["entries"] == []

    client.post(
        "/analyze", files={"file": ("s.jpg", _jpeg(render_sample("starch", seed=7)), "image/jpeg")}
    )
    r1 = client.get("/history")
    entries = r1.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["verdict"]["adulterant"] == "starch"
    assert entries[0]["timestamp"]


def test_history_is_capped(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/analyze",
            files={"file": (f"s{i}.jpg", _jpeg(render_sample("none", seed=i)), "image/jpeg")},
        )
    r = client.get("/history", params={"limit": 2})
    assert len(r.json()["entries"]) == 2


# --- report -----------------------------------------------------------------


def test_report_returns_pdf_for_existing_entry(client: TestClient) -> None:
    created = client.post(
        "/analyze",
        files={"file": ("s.jpg", _jpeg(render_sample("detergent", seed=7)), "image/jpeg")},
    ).json()
    r = client.get(f"/report/{created['id']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content[:5] == b"%PDF-"
    assert len(r.content) > 800


def test_report_404s_for_unknown_entry(client: TestClient) -> None:
    assert client.get("/report/does-not-exist").status_code == 404


# --- static frontend --------------------------------------------------------


def test_index_serves_the_spa(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"SaafScan" in r.content


# --- security baseline ------------------------------------------------------


def test_debug_stacktrace_not_leaked_on_bad_input(client: TestClient) -> None:
    r = client.post("/analyze", files={"file": ("x.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 422
    assert "Traceback" not in r.text
    assert 'File "/' not in r.text
