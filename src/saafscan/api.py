"""FastAPI app: /analyze, /history, /report/{id}, /health, static SPA.

Uploads are size- and type-validated at the boundary; unusable images get a
structured 422 with a human-readable message and never become evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from saafscan import evidence as evidence_mod
from saafscan.evidence import EvidenceStore
from saafscan.pipeline import PipelineError, analyze
from saafscan.report import build_report
from saafscan.synthetic import ADULTERANTS, sample_jpeg

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="SaafScan", version="0.1.0")


def _store() -> EvidenceStore:
    """Fresh store per request so DB_PATH stays overridable (tests, env)."""
    return EvidenceStore(db_path=evidence_mod.DB_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": True, "version": app.version}


@app.post("/analyze")
async def analyze_upload(file: Annotated[UploadFile, File()]) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please upload a smaller photo.",
        )

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=422,
            detail="This file could not be read as an image. Please upload a JPEG or PNG photo.",
        )
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = analyze(image)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sample_name = Path(file.filename or "upload").stem
    entry_id = _store().record(result.verdict, sample_name=sample_name, image_bytes=payload)
    return {
        "id": entry_id,
        "verdict": result.verdict.to_dict(),
        "timing_ms": round(result.timing_ms, 1),
        "zones": {k: [round(c, 1) for c in v] for k, v in result.zones.items()},
    }


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    return {"entries": [entry.to_dict() for entry in _store().list(limit=limit)]}


@app.get("/report/{entry_id}")
def report(entry_id: str) -> Response:
    entry = _store().get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="No such evidence entry.")
    pdf = build_report(
        entry.verdict,
        sample_name=entry.sample_name or "unknown",
        timestamp=entry.timestamp,
        image_bytes=entry.image_bytes,
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="saafscan-{entry_id}.pdf"'},
    )


@app.get("/static/samples/{label}.jpg")
def demo_sample(label: str) -> Response:
    if label not in ADULTERANTS:
        raise HTTPException(status_code=404, detail="Unknown demo sample.")
    return Response(content=sample_jpeg(label), media_type="image/jpeg")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
