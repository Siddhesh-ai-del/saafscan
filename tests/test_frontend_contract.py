"""Lock the frontend DOM contract between index.html and app.js (TDD-first).

The redesign may restyle anything, but app.js drives the UI by these exact
IDs/classes/selectors. If this test goes red, the redesign broke behavior.
Also enforces the offline invariant: no external resource references.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
INDEX = FRONTEND / "index.html"
APP_JS = FRONTEND / "app.js"

# Every $("…") selector in app.js, plus #camera-hint-adjacent markup contracts.
REQUIRED_IDS = [
    "status",
    "video",
    "canvas",
    "file-input",
    "drop",
    "camera-state",
    "snap",
    "flip",
    "verdict-panel",
    "v-chip",
    "v-conf",
    "v-meter",
    "v-sev",
    "v-advice",
    "v-evidence",
    "v-report",
    "v-again",
    "v-latency",
    "v-error",
    "history",
    "refresh",
]

REQUIRED_DEMO_SAMPLES = ["none", "starch", "detergent"]


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_every_id_required_by_app_js_exists_once(html: str) -> None:
    for el_id in REQUIRED_IDS:
        occurrences = re.findall(rf'id="{re.escape(el_id)}"', html)
        assert len(occurrences) == 1, f"#{el_id} must appear exactly once, found {len(occurrences)}"


def test_demo_buttons_keep_their_sample_data_attributes(html: str) -> None:
    assert 'button class="btn demo"' in html or 'class="btn demo"' in html
    for sample in REQUIRED_DEMO_SAMPLES:
        assert f'data-sample="{sample}"' in html, f"missing demo button for {sample}"


def test_drag_state_markup_is_preserved(html: str) -> None:
    # app.js toggles .hidden and .drop.over — both classes must stay styled.
    css_sources = list(FRONTEND.glob("css/*.css")) + [FRONTEND / "styles.css"]
    css = "\n".join(p.read_text(encoding="utf-8") for p in css_sources if p.exists())
    assert ".hidden" in css, ".hidden class must be styled (JS toggles it)"
    assert re.search(r"\.drop(\s|\.|,|:).*over|\.drop\.over", css), (
        ".drop.over drag state must be styled"
    )


def test_history_template_classes_are_styled(html: str) -> None:
    css_sources = list(FRONTEND.glob("css/*.css")) + [FRONTEND / "styles.css"]
    css = "\n".join(p.read_text(encoding="utf-8") for p in css_sources if p.exists())
    for cls in ["entry", "chip", "when", "conf", "hint", "error"]:
        assert re.search(rf"\.{cls}(\b|[:.\s,])", css), f".{cls} (written by app.js) must be styled"


def test_brand_string_present_for_the_api_spa_test(html: str) -> None:
    assert "SaafScan" in html


def test_no_external_resource_references(html: str) -> None:
    """Offline invariant: nothing is fetched from a CDN or third party."""
    refs = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert refs == [], f"external references found: {refs}"
    for css_path in FRONTEND.glob("css/*.css"):
        urls = re.findall(r"url\((https?://[^)]+)\)", css_path.read_text(encoding="utf-8"))
        assert urls == [], f"{css_path.name} references external URLs: {urls}"
    for js_path in FRONTEND.glob("**/*.js"):
        # Protocol-relative ("//host/...") or absolute fetches/loadings.
        urls = re.findall(r'["\'](?:https?://|//)[^"\']+', js_path.read_text(encoding="utf-8"))
        assert urls == [], f"{js_path.name} references external URLs: {urls}"


def test_linked_stylesheets_exist_on_disk(html: str) -> None:
    links = re.findall(r'<link rel="stylesheet" href="/static/([^"]+)"', html)
    assert links, "index.html must link at least one stylesheet"
    for rel in links:
        assert (FRONTEND / rel).exists(), f"linked stylesheet missing: {rel}"


def test_linked_scripts_exist_on_disk(html: str) -> None:
    scripts = re.findall(r'<script src="/static/([^"]+)"></script>', html)
    for required in ["app.js", "js/ambient.js", "js/motion.js"]:
        assert required in scripts, f"{required} must be loaded by index.html"
    for rel in scripts:
        assert (FRONTEND / rel).exists(), f"linked script missing: {rel}"
