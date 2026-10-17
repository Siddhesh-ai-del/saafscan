"""RED tests: ingesting a real strip photograph into the training set.

Round 1 promised the model would be retrained on real strip photos before the
final, and nothing in the repository can do that yet — `build_dataset` only
ever calls the synthetic renderer, while `ml.train.load_dataset` reads
manifest.json + .npy crops without caring how a crop was produced. Ingestion is
the missing link.

The split policy is the part that can quietly ruin the result: synthetic data
splits by lighting condition so no condition appears in two splits. Real photos
taken of the same lamp in one sitting must be grouped the same way, or the
held-out accuracy measures nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from saafscan.ingest import ingest_directory, ingest_photo, main
from saafscan.roi import STRIP_SIZE, detect_strip
from saafscan.synthetic import LIGHTING_CONDITIONS, render_training_scene


@pytest.fixture()
def scene_dir(tmp_path: Path) -> Path:
    """Three synthetic 'photographs' standing in for shots off a phone.

    Synthetic renders are used as fixtures because they are what the pipeline
    provably detects; the ingest path itself is exercised end to end.
    """
    photos = tmp_path / "photos"
    photos.mkdir()
    for index, label in enumerate(("none", "starch", "detergent")):
        scene = render_training_scene(label, LIGHTING_CONDITIONS[0], seed=index)
        import cv2

        cv2.imwrite(
            str(photos / f"S001_{label}_{index}.jpg"),
            cv2.cvtColor(scene, cv2.COLOR_RGB2BGR),
        )
    return photos


def test_ingest_turns_a_photograph_into_a_canonical_crop(tmp_path: Path) -> None:
    """The crop must be the exact tensor the classifier consumes."""
    import cv2

    scene = render_training_scene("starch", LIGHTING_CONDITIONS[0], seed=1)
    photo = tmp_path / "shot.jpg"
    cv2.imwrite(str(photo), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    sample = ingest_photo(photo, label="starch", dataset_dir=tmp_path / "ds", session="S001")

    crop = np.load(tmp_path / "ds" / sample["path"])
    # STRIP_SIZE is (width, height); the array is (height, width, channels).
    assert crop.shape == STRIP_SIZE[::-1] + (3,)
    assert sample["label"] == "starch"
    assert sample["session"] == "S001"


def test_ingest_records_a_manifest_the_trainer_can_read(tmp_path: Path, scene_dir: Path) -> None:
    result = ingest_directory(scene_dir, dataset_dir=tmp_path / "ds")

    import json

    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["samples"], "ingest produced no samples"
    for sample in manifest["samples"]:
        assert (tmp_path / "ds" / sample["path"]).is_file()
    assert result["ingested"] == len(manifest["samples"])


def test_a_photograph_the_detector_cannot_use_is_reported_not_silently_dropped(
    tmp_path: Path,
) -> None:
    import cv2

    from saafscan.synthetic import render_random_photo

    photo = tmp_path / "selfie.jpg"
    cv2.imwrite(str(photo), cv2.cvtColor(render_random_photo(seed=3), cv2.COLOR_RGB2BGR))

    result = ingest_photo(
        photo, label="none", dataset_dir=tmp_path / "ds", session="S001", strict=False
    )

    assert result["ingested"] is False
    assert "card" in result["reason"].lower() or "strip" in result["reason"].lower()
    assert not (tmp_path / "ds" / "manifest.json").exists()


def test_strict_mode_raises_instead_of_reporting(tmp_path: Path) -> None:
    import cv2

    from saafscan.synthetic import render_random_photo

    photo = tmp_path / "selfie.jpg"
    cv2.imwrite(str(photo), cv2.cvtColor(render_random_photo(seed=3), cv2.COLOR_RGB2BGR))

    with pytest.raises(ValueError) as exc:
        ingest_photo(photo, label="none", dataset_dir=tmp_path / "ds", session="S001")
    assert "card" in str(exc.value).lower() or "strip" in str(exc.value).lower()


def test_all_photos_from_one_session_land_in_the_same_split(
    tmp_path: Path, scene_dir: Path
) -> None:
    """No leakage: a session is one lighting setup, so it is one split."""
    ingest_directory(scene_dir, dataset_dir=tmp_path / "ds")

    import json

    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    by_session: dict[str, set[str]] = {}
    for sample in manifest["samples"]:
        by_session.setdefault(sample["session"], set()).add(sample["split"])
    for session, splits in by_session.items():
        assert len(splits) == 1, f"session {session} spans splits {splits}"


def test_split_assignment_is_deterministic_across_runs(tmp_path: Path) -> None:
    from saafscan.ingest import split_for_session

    assert split_for_session("S001") == split_for_session("S001")
    assert split_for_session("S001") in {"train", "val", "test"}


def test_reingesting_the_same_directory_does_not_duplicate_samples(
    tmp_path: Path, scene_dir: Path
) -> None:
    ingest_directory(scene_dir, dataset_dir=tmp_path / "ds")
    second = ingest_directory(scene_dir, dataset_dir=tmp_path / "ds")

    import json

    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    paths = [s["path"] for s in manifest["samples"]]
    assert len(paths) == len(set(paths)), "duplicate crops after a second run"
    assert second["ingested"] >= 0


def test_ingest_rejects_an_unknown_label(tmp_path: Path, scene_dir: Path) -> None:
    with pytest.raises(ValueError):
        ingest_directory(scene_dir, dataset_dir=tmp_path / "ds", label="magic")


def test_ingest_finds_labels_in_the_filename_by_default(tmp_path: Path, scene_dir: Path) -> None:
    result = ingest_directory(scene_dir, dataset_dir=tmp_path / "ds")
    assert result["ingested"] == 3
    assert result["rejected"] == 0


def test_strip_detection_is_available_for_the_ingested_shots(
    tmp_path: Path, scene_dir: Path
) -> None:
    """Sanity: the fixture shots really do contain a detectable strip."""
    import cv2

    first = sorted(scene_dir.glob("*.jpg"))[0]
    image = cv2.cvtColor(cv2.imread(str(first)), cv2.COLOR_BGR2RGB)
    assert detect_strip(image) is not None


# --------------------------------------------------------------------------
# command line: this is what runs on shoot day, standing next to the strips
# --------------------------------------------------------------------------
def test_cli_ingests_a_folder_and_prints_a_summary(tmp_path: Path, scene_dir: Path, capsys) -> None:
    dataset = tmp_path / "realset"
    assert main([str(scene_dir), "--dataset", str(dataset)]) == 0

    out = capsys.readouterr().out
    assert "3 taken in" in out
    assert (dataset / "manifest.json").is_file()


def test_cli_refuses_an_unknown_label_without_touching_the_dataset(
    tmp_path: Path, scene_dir: Path, capsys
) -> None:
    dataset = tmp_path / "realset"
    assert main([str(scene_dir), "--dataset", str(dataset), "--label", "magic"]) == 2
    assert "unknown label" in capsys.readouterr().err
    assert not (dataset / "manifest.json").exists()


def test_cli_reports_a_refused_shot_but_keeps_the_good_ones(
    tmp_path: Path, scene_dir: Path, capsys
) -> None:
    import cv2

    from saafscan.synthetic import render_random_photo

    cv2.imwrite(
        str(scene_dir / "S001_none_9.jpg"),
        cv2.cvtColor(render_random_photo(seed=5), cv2.COLOR_RGB2BGR),
    )
    dataset = tmp_path / "realset"

    assert main([str(scene_dir), "--dataset", str(dataset)]) == 0
    captured = capsys.readouterr()
    assert "3 taken in" in captured.out
    assert "1 refused" in captured.out
    assert "S001_none_9.jpg" in captured.err


def test_cli_fails_when_not_a_single_shot_survives(tmp_path: Path, capsys) -> None:
    import cv2

    from saafscan.synthetic import render_random_photo

    folder = tmp_path / "bad"
    folder.mkdir()
    cv2.imwrite(
        str(folder / "S001_selfie_0.jpg"),
        cv2.cvtColor(render_random_photo(seed=6), cv2.COLOR_RGB2BGR),
    )

    assert main([str(folder), "--dataset", str(tmp_path / "realset")]) == 1
    assert "refused" in capsys.readouterr().err


def test_cli_fails_cleanly_on_an_empty_folder(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty), "--dataset", str(tmp_path / "realset")]) == 2
    assert "no photographs" in capsys.readouterr().err
