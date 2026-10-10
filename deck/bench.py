"""Repeatable benchmark for every performance number printed in the deck.

Writes deck/bench.json. Run:  .venv/bin/python deck/bench.py   (from repo root)

Sources:
  deck/bench.json   ->  pipeline_ms, http_ms, cold_start_ms
  models/eval_report.json -> accuracy, coverage, abstained, delta, thresholds
  data/trainset/manifest.json -> dataset sizes and splits
"""

from __future__ import annotations

import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saafscan.pipeline import analyze  # noqa: E402
from saafscan.synthetic import render_sample  # noqa: E402


def bench_pipeline(n_per_class: int = 5) -> dict:
    img = render_sample("starch", seed=0)
    for _ in range(3):
        analyze(img)
    lat, inner = [], []
    for label in ("none", "starch", "detergent", "both"):
        for seed in range(n_per_class):
            im = render_sample(label, seed=seed)
            t0 = time.perf_counter()
            analysis = analyze(im)
            lat.append((time.perf_counter() - t0) * 1000)
            inner.append(analysis.timing_ms)
    lat.sort()
    return {
        "n": len(lat),
        "min": round(lat[0], 1),
        "median": round(statistics.median(lat), 1),
        "p95": round(lat[int(0.95 * len(lat)) - 1], 1),
        "max": round(lat[-1], 1),
        "pipeline_reported_median": round(statistics.median(sorted(inner)), 1),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def bench_http_and_cold_start() -> dict:
    """Cold start = process spawn -> first successful /health, measured on a fresh server."""
    port = _free_port()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/siddhesh",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
    }
    t_spawn = time.perf_counter()
    proc = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin/python"),
            "-m",
            "uvicorn",
            "saafscan.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cold = None
    try:
        for _ in range(400):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if r.status == 200:
                        cold = (time.perf_counter() - t_spawn) * 1000
                        break
            except (urllib.error.URLError, OSError, ConnectionError):
                time.sleep(0.05)
        if cold is None:
            raise RuntimeError("server never became healthy")

        # end-to-end: JPEG upload -> verdict, over real HTTP
        lat = []
        for label in ("none", "starch", "detergent", "both"):
            for seed in range(3):
                payload = render_sample(label, seed=seed)
                import io

                import PIL.Image as Image

                buf = io.BytesIO()
                Image.fromarray(payload).save(buf, format="JPEG", quality=92)
                body = buf.getvalue()
                boundary = "----benchbound"
                data = (
                    (
                        f"--{boundary}\r\nContent-Disposition: form-data; "
                        f'name="file"; filename="{label}.jpg"\r\n'
                        f"Content-Type: image/jpeg\r\n\r\n"
                    ).encode()
                    + body
                    + f"\r\n--{boundary}--\r\n".encode()
                )
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/analyze",
                    data=data,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
                lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        http = {
            "n": len(lat),
            "min": round(lat[0], 1),
            "median": round(statistics.median(lat), 1),
            "max": round(lat[-1], 1),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"cold_start_ms": round(cold, 1), "http_e2e_ms": http}


def main() -> None:
    out = {
        "pipeline_ms": bench_pipeline(),
        **bench_http_and_cold_start(),
        "host": "laptop, CPU inference",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = ROOT / "deck" / "bench.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
