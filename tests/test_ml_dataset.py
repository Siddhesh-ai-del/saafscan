"""RED tests: training-dataset extension of the synthetic scene generator.

Contracts under test:
- LightingCondition sampling is deterministic and identifies a condition stably.
- The train/val/test split is BY LIGHTING CONDITION (disjoint, never random).
- render_training_scene covers the five verdict classes incl. 'inconclusive'.
- build_dataset emits a manifest with per-condition provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.synthetic import (
    LIGHTING_CONDITIONS,
    TRAINING_CLASSES,
    LightingCondition,
    build_dataset,
    render_training_scene,
    sample_lighting,
    split_for_condition,
)

# Training classes: the four reactions plus a learned abstain class.
EXPECTED_CLASSES = {"none", "starch", "detergent", "both", "inconclusive"}


def test_training_classes_include_reactions_and_abstain() -> None:
    assert set(TRAINING_CLASSES) == EXPECTED_CLASSES


def test_sample_lighting_is_deterministic() -> None:
    a = sample_lighting(5)
    b = sample_lighting(5)
    assert a == b
    assert isinstance(a, LightingCondition)


def test_sample_lighting_varies_with_seed() -> None:
    assert sample_lighting(1) != sample_lighting(2)


def test_lighting_conditions_are_many_and_stable() -> None:
    """The catalogue must support a condition-disjoint 3-way split."""
    assert len(LIGHTING_CONDITIONS) >= 30
    keys = [c.key for c in LIGHTING_CONDITIONS]
    assert len(set(keys)) == len(keys), "condition keys must be unique"
    # Deterministic across processes (stable string form).
    assert LIGHTING_CONDITIONS[0] == sample_lighting(LIGHTING_CONDITIONS[0].seed)


def test_split_for_condition_is_disjoint_and_deterministic() -> None:
    splits = {split_for_condition(c) for c in LIGHTING_CONDITIONS}
    assert splits == {"train", "val", "test"}
    # Every condition maps to exactly one split, stable across calls.
    for c in LIGHTING_CONDITIONS[:10]:
        assert split_for_condition(c) == split_for_condition(c)
    # No condition key appears in more than one split.
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for c in LIGHTING_CONDITIONS:
        by_split[split_for_condition(c)].append(c.key)
    all_keys = [k for v in by_split.values() for k in v]
    assert len(all_keys) == len(set(all_keys))
    assert min(len(v) for v in by_split.values()) >= 3, "each split needs conditions"


def test_split_never_puts_a_condition_in_val_and_test() -> None:
    """The leakage guard: val and test conditions must be disjoint."""
    val = {c.key for c in LIGHTING_CONDITIONS if split_for_condition(c) == "val"}
    test = {c.key for c in LIGHTING_CONDITIONS if split_for_condition(c) == "test"}
    train = {c.key for c in LIGHTING_CONDITIONS if split_for_condition(c) == "train"}
    assert not (val & test)
    assert not (train & test)
    assert not (train & val)


@pytest.mark.parametrize("label", sorted(EXPECTED_CLASSES))
def test_render_training_scene_per_class(label: str) -> None:
    img = render_training_scene(label, sample_lighting(0), seed=3)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_render_training_scene_is_deterministic() -> None:
    cond = sample_lighting(0)
    assert np.array_equal(
        render_training_scene("starch", cond, seed=3),
        render_training_scene("starch", cond, seed=3),
    )


def test_render_training_scene_rejects_unknown_class() -> None:
    with pytest.raises(ValueError):
        render_training_scene("plutonium", sample_lighting(0), seed=1)


def test_inconclusive_scenes_really_are_ambiguous() -> None:
    """The abstain class must land in the rule-table's inconclusive band."""
    from saafscan.classifier import classify_zones
    from saafscan.roi import extract_zones

    checked = 0
    for seed in range(8):
        img = render_training_scene("inconclusive", sample_lighting(seed), seed=seed)
        verdict = classify_zones(extract_zones(img))
        checked += verdict.adulterant == "inconclusive"
    assert checked >= 6, f"only {checked}/8 inconclusive scenes read as inconclusive to the rules"


@pytest.mark.parametrize("label", ["none", "starch", "detergent", "both"])
def test_clear_training_scenes_match_the_rules(label: str) -> None:
    """Non-abstain training labels must agree with the rule-table baseline."""
    from saafscan.classifier import classify_zones
    from saafscan.roi import extract_zones

    for seed in range(3):
        img = render_training_scene(label, sample_lighting(seed), seed=seed)
        assert classify_zones(extract_zones(img)).adulterant == label


def test_build_dataset_emits_manifest_and_arrays(tmp_path) -> None:
    manifest = build_dataset(
        tmp_path,
        conditions=[sample_lighting(0), sample_lighting(1)],
        seeds=(0, 1),
    )
    assert manifest["version"] == 1
    assert set(manifest["classes"]) == EXPECTED_CLASSES
    assert len(manifest["samples"]) == 2 * 2 * len(EXPECTED_CLASSES)
    sample = manifest["samples"][0]
    assert {"label", "condition_key", "split", "seed", "path"} <= set(sample)
    # Arrays exist on disk.
    crop = np.load(tmp_path / sample["path"])
    assert crop.shape[0] > 0
    # Splits in the manifest match split_for_condition.
    for s in manifest["samples"]:
        cond = next(
            c for c in (sample_lighting(0), sample_lighting(1)) if c.key == s["condition_key"]
        )
        assert s["split"] == split_for_condition(cond)
