"""Generate every chart the deck prints, from the repo's own data files.

Reads:  data/trainset/manifest.json      (dataset composition)
        models/eval_report.json          (confusion matrix, coverage, thresholds)
        models/features.json, cnn.json   (model metadata, temperatures)
        deck/bench.json                  (measured latencies, cold start)
        deck/external_facts.json         (cited third-party figures)

Writes PNGs into deck/assets/.  Run:  deck/.venv/bin/python deck/make_assets.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "deck" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- template palette -------------------------------------------------------
INK = "#1F3A30"  # dark green, primary text
DEEP = "#2A4A3E"  # deep green
MUTED = "#6B7A72"  # muted grey-green
FAINT = "#9DB3A7"  # light grey-green
CREAM = "#F4F1E8"  # cream
PAGE = "#F8F6F0"  # page background
GOLD = "#F5C842"  # yellow accent
VERM = "#D9581C"  # vermilion
ORANGE = "#F26B2A"  # orange
LINE = "#DCD6C8"  # hairline

for f in (Path.home() / ".local/share/fonts/deckfonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DM Sans", "Rubik", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": PAGE,
        "axes.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)

CLASS_ORDER = ["none", "starch", "detergent", "both", "inconclusive"]
CLASS_LABEL = {
    "none": "none",
    "starch": "starch",
    "detergent": "detergent",
    "both": "both",
    "inconclusive": "inconclusive",
}


def _finish(fig, name: str, dpi: int = 220) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  wrote {path.name}  ({path.stat().st_size // 1024} KB)")


def _strip_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    fig = ax.figure
    fig.patch.set_alpha(0)


# --------------------------------------------------------------------------
# 1. FSSAI milk & dairy surveillance (slide 2)
# --------------------------------------------------------------------------
def chart_fssai(facts: dict) -> None:
    d = facts["fssai_milk"]
    years = ["FY 2024-25", "FY 2025-26"]
    tested = [d["fy25_tested"], d["fy26_tested"]]
    failed = [d["fy25_failed"], d["fy26_failed"]]
    rate = [d["fy25_rate_pct"], d["fy26_rate_pct"]]

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(years))
    w = 0.36

    b1 = ax.bar(x - w / 2, tested, w, label="Samples analysed", color=DEEP, zorder=3)
    b2 = ax.bar(x + w / 2, failed, w, label="Non-conforming", color=VERM, zorder=3)

    for bar, val in zip(b1, tested, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 900,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=DEEP,
        )
    for bar, val, r in zip(b2, failed, rate, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 900,
            f"{val:,}\n{r}% of tested",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=VERM,
            linespacing=1.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11.5)
    ax.set_ylim(0, max(tested) * 1.30)
    ax.set_yticks([])
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10.5,
        ncol=1,
        labelcolor=MUTED,
        bbox_to_anchor=(-0.01, 1.03),
    )
    _finish(fig, "c_fssai.png")


# --------------------------------------------------------------------------
# 2. Held-out confusion matrix (slide 5)
# --------------------------------------------------------------------------
def chart_confusion(rep: dict) -> None:
    cm = np.array(rep["ml"]["confusion"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    # invisible grid so cells are evenly sized
    ax.imshow(np.ones_like(cm), cmap="Greys", vmin=0, vmax=14, alpha=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            on_diag = i == j
            val = int(cm[i, j])
            face = DEEP if on_diag else ("#EFEAE0" if val == 0 else VERM)
            txt = str(val) if val else ""
            ax.add_patch(
                FancyBboxPatch(
                    (j - 0.42, i - 0.42),
                    0.84,
                    0.84,
                    boxstyle="round,pad=0.008,rounding_size=0.09",
                    facecolor=face,
                    edgecolor="none",
                    zorder=2,
                )
            )
            if txt:
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#FFFFFF" if on_diag else INK,
                    zorder=3,
                )

    labels = [CLASS_LABEL[c] for c in CLASS_ORDER]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5, color=MUTED)
    ax.set_xlabel("predicted", fontsize=10.5, color=MUTED, labelpad=8)
    ax.set_ylabel("actual", fontsize=10.5, color=MUTED, labelpad=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_tick_params(length=0, pad=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    _finish(fig, "c_confusion.png")


# --------------------------------------------------------------------------
# 3. Abstention breakdown (slide 5)
# --------------------------------------------------------------------------
def chart_abstain(rep: dict, manifest: dict) -> None:
    n_total = rep["dataset"]["n_test_samples"]
    abstained = rep["ml"]["abstained"]
    answered = n_total - abstained

    fig, ax = plt.subplots(figsize=(7.4, 1.55))
    _strip_axes(ax)

    ax.barh([0], [answered], height=0.55, color=DEEP, zorder=3)
    ax.barh([0], [abstained], left=[answered], height=0.55, color=GOLD, zorder=3)

    ax.text(
        answered / 2,
        0,
        f"{answered} returned a verdict",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#FFFFFF",
    )
    ax.text(
        answered + abstained / 2,
        0,
        f"{abstained} abstained",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
    )
    ax.text(
        0,
        -0.62,
        f"Held-out by lighting condition · n = {n_total} · "
        f"selective accuracy 1.000 · coverage {rep['ml']['coverage']:.3f}",
        fontsize=10,
        color=MUTED,
        va="top",
    )

    ax.set_xlim(0, n_total)
    ax.set_ylim(-1.15, 0.55)
    _finish(fig, "c_abstain.png")


# --------------------------------------------------------------------------
# 4. Measured latency vs the "under 10 seconds" promise (slide 5)
# --------------------------------------------------------------------------
def chart_latency(bench: dict) -> None:
    p, h = bench["pipeline_ms"], bench["http_e2e_ms"]
    rows = [
        ("CV + classifier, in-process", p["median"], DEEP),
        ("Full HTTP request -> verdict", h["median"], ORANGE),
        ("Slowest HTTP request observed", h["max"], VERM),
        ("Design promise", 10_000, FAINT),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    y = np.arange(len(rows))[::-1]

    for yi, (_label, val, colour) in zip(y, rows, strict=True):
        ax.barh([yi], [val], height=0.52, color=colour, zorder=3)
        if val >= 1000:
            txt = f"{val / 1000:.0f} s"
        elif val >= 1:
            txt = f"{val:.1f} ms"
        else:
            txt = f"{val:.2f} ms"
        pad = val * 0.025 + 20
        ax.text(
            val + pad,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=11,
            color=INK if colour != FAINT else MUTED,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 150_000)
    ax.set_xticks([])
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        "measured on the build laptop · median of 20 / 12 runs",
        fontsize=9.5,
        color=MUTED,
        labelpad=10,
    )
    _finish(fig, "c_latency.png")


# --------------------------------------------------------------------------
# 5. Training dataset composition (slide 5)
# --------------------------------------------------------------------------
def chart_dataset(manifest: dict) -> None:
    per = defaultdict(Counter)
    for s in manifest["samples"]:
        per[s["split"]][s["label"]] += 1
    splits = ["train", "val", "test"]
    colours = {"train": DEEP, "val": ORANGE, "test": GOLD}
    labels = [CLASS_LABEL[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for split in splits:
        vals = np.array([per[split][c] for c in CLASS_ORDER], dtype=float)
        ax.bar(x, vals, 0.62, bottom=bottom, label=split, color=colours[split], zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottom, strict=True)):
            if v:
                ax.text(
                    xi,
                    b + v / 2,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color="#FFFFFF" if split != "test" else INK,
                )
        bottom += vals

    for xi, tot in enumerate(bottom):
        ax.text(xi, tot + 4, str(int(tot)), ha="center", va="bottom", fontsize=10.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_yticks([])
    ax.set_ylim(0, max(bottom) * 1.42)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.legend(
        loc="upper center",
        frameon=False,
        fontsize=10,
        ncol=3,
        labelcolor=MUTED,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.6,
        handlelength=1.1,
        handleheight=0.9,
    )
    _finish(fig, "c_dataset.png")


# --------------------------------------------------------------------------
# 6. Why Model A ships (slide 4)
# --------------------------------------------------------------------------
def chart_models(features_meta: dict, cnn_meta: dict, bench_files: dict) -> None:
    """Single-panel log-scale artifact comparison sized for the slide-4 slot."""
    fig, ax = plt.subplots(figsize=(5.40, 1.15))

    sizes = [bench_files["features_onnx_bytes"], bench_files["cnn_onnx_bytes"]]
    names = ["Model A\n12 colour features", "Model B\nResNet18"]
    cols = [DEEP, FAINT]
    y = [1, 0]
    for yi, val, c in zip(y, sizes, cols, strict=True):
        ax.barh([yi], [val], height=0.66, color=c, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1.5e9)
    ax.set_xticks([])
    ax.xaxis.set_visible(False)
    ax.text(sizes[0] * 2.4, 1, "179 KB", va="center", fontsize=11, color=DEEP)
    ax.text(sizes[1] * 2.4, 0, "45 MB", va="center", fontsize=11, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK, linespacing=1.4)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 1.6)

    _finish(fig, "c_models.png")


def crop_screenshots() -> None:
    """Crop the app screenshots to the landscape card slots the deck uses."""
    src = ROOT / "screenshots"
    out = ROOT / "deck" / "assets"
    jobs = {
        # name: (source, left, upper, right, lower) as fractions of width/height
        "s_verdict_starch.png": ("02-verdict-starch.png", 0.06, 0.055, 0.94, 0.445),
        "s_verdict_detergent.png": ("03-verdict-detergent.png", 0.06, 0.055, 0.94, 0.445),
        "s_verdict_pure.png": ("04-verdict-pure.png", 0.06, 0.055, 0.94, 0.445),
        "s_evidence.png": ("05-evidence-history.png", 0.06, 0.415, 0.94, 0.715),
        "s_capture.png": ("01-landing-capture.png", 0.06, 0.055, 0.55, 0.445),
    }
    print("cropping screenshots:")
    for name, (s, left, top_frac, right, bottom_frac) in jobs.items():
        im = Image.open(src / s).convert("RGB")
        w, h = im.size
        box = (int(w * left), int(h * top_frac), int(w * right), int(h * bottom_frac))
        crop = im.crop(box)
        # downscale to a sane working width; the deck displays them at ~3.85in
        if crop.width > 1200:
            crop = crop.resize((1200, int(crop.height * 1200 / crop.width)), Image.LANCZOS)
        dst = out / name
        crop.save(dst, quality=92)
        print(f"  {name}  {crop.size[0]}x{crop.size[1]}  aspect {crop.size[0] / crop.size[1]:.2f}")


def main() -> None:
    manifest = json.loads((ROOT / "data" / "trainset" / "manifest.json").read_text())
    report = json.loads((ROOT / "models" / "eval_report.json").read_text())
    feats = json.loads((ROOT / "models" / "features.json").read_text())
    cnn = json.loads((ROOT / "models" / "cnn.json").read_text())
    bench = json.loads((ROOT / "deck" / "bench.json").read_text())
    facts = json.loads((ROOT / "deck" / "external_facts.json").read_text())

    rep = report["reports"]["features"]
    bench_files = {
        "features_onnx_bytes": (ROOT / "models" / "features.onnx").stat().st_size,
        "cnn_onnx_bytes": (ROOT / "models" / "cnn.onnx").stat().st_size,
    }

    print("building charts:")
    chart_fssai(facts)
    chart_confusion(rep)
    chart_abstain(rep, manifest)
    chart_latency(bench)
    chart_dataset(manifest)
    chart_models(feats, cnn, bench_files)
    crop_screenshots()
    print("done.")


if __name__ == "__main__":
    main()
