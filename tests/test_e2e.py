"""RED tests: end-to-end gauntlet — upload → verdict → history → PDF, plus rejects.

This is the exact success-criteria path from the implementation plan:
3 synthetic samples classified correctly, malformed uploads rejected,
report retrievable, all through the public HTTP surface.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from saafscan.api import app
from saafscan.synthetic import render_random_photo, render_sample


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from saafscan import evidence

    monkeypatch.setattr(evidence, "DB_PATH", tmp_path / "evidence.db")
    with TestClient(app) as c:
        yield c


def _upload(client: TestClient, img: np.ndarray, name: str = "s.jpg"):
    import cv2

    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1])  # scenes are RGB; cv2 wants BGR
    assert ok
    return client.post("/analyze", files={"file": (name, buf.tobytes(), "image/jpeg")})


GOLDEN = [("none", "none"), ("starch", "starch"), ("detergent", "detergent")]


def test_demo_gauntlet_three_samples_full_path(client: TestClient) -> None:
    """The demo gauntlet: 3 samples, each through verdict → history → PDF."""
    seen_ids: list[str] = []
    for label, expected in GOLDEN:
        r = _upload(client, render_sample(label, seed=7))
        assert r.status_code == 200, f"{label}: {r.text}"
        body = r.json()
        assert body["verdict"]["adulterant"] == expected
        seen_ids.append(body["id"])

    # History now holds all three, newest first.
    history = client.get("/history").json()["entries"]
    assert len(history) == 3
    assert [e["verdict"]["adulterant"] for e in history] == ["detergent", "starch", "none"]

    # Every entry has a real PDF report.
    for vid in seen_ids:
        rep = client.get(f"/report/{vid}")
        assert rep.status_code == 200
        assert rep.content[:5] == b"%PDF-"


@pytest.mark.parametrize(
    ("payload", "ctype", "why"),
    [
        (b"", "image/jpeg", "empty file"),
        (b"not an image", "image/jpeg", "garbage bytes"),
        (b"hello world", "text/plain", "text file"),
        (b"<html><body>hi</body></html>", "text/html", "html masquerading as image"),
    ],
)
def test_malformed_uploads_are_rejected(
    client: TestClient, payload: bytes, ctype: str, why: str
) -> None:
    r = client.post("/analyze", files={"file": ("bad.jpg", payload, ctype)})
    assert r.status_code == 422, f"{why} was not rejected ({r.status_code})"
    assert "Traceback" not in r.text


def test_selfie_like_photo_is_rejected_not_diagnosed(client: TestClient) -> None:
    """Non-fundus/non-test image must be rejected with guidance, never a verdict."""
    r = _upload(client, render_random_photo(seed=11), name="selfie.png")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str) and len(detail) > 10


def test_rejections_do_not_pollute_history(client: TestClient) -> None:
    _upload(client, render_sample("starch", seed=7))
    client.post("/analyze", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
    client.post("/analyze", files={"file": ("bad2.jpg", b"", "image/jpeg")})
    history = client.get("/history").json()["entries"]
    assert len(history) == 1, "rejected uploads must not create evidence entries"


def test_health_is_ok_before_and_after_load(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    _upload(client, render_sample("none", seed=7))
    assert client.get("/health").status_code == 200


def test_spa_references_required_assets(client: TestClient) -> None:
    html = client.get("/").text
    for needle in ("SaafScan", "/analyze", "/history"):
        assert needle in html, f"SPA missing {needle!r}"
