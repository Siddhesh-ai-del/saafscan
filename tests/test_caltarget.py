"""Printable colour targets: validate the imaging half before any chemistry.

A printed target is not an adulterant result and must never be presented as
one. What it buys is separation of the two ways this demo can fail. A real
strip that misreads could be the camera, the white balance, the strip
detector, the zone extraction — or the chemistry. A printed target has no
chemistry to go wrong, so if a printed `detergent` target photographs and
reads back as detergent, the whole imaging chain is proven, and any later
failure is unambiguously the chemistry.

Drawn with :mod:`saafscan.printstrip`'s geometry and coloured by
:func:`saafscan.synthetic.zone_colours`, so a printed target is the *same
physical object* as the real strip it stands in for and the *same* strip the
classifier was trained on. Those two were once allowed to drift apart, and the
model learned the drawing instead of the chemistry.
"""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.caltarget import (
    TARGETS,
    render_sheet,
    render_strip,
    target_page,
    zone_colours,
)
from saafscan.classifier import CLEAR, FIRED, classify_zones
from saafscan.color import delta_e
from saafscan.printstrip import CONTROL_COLOUR, sheet_header
from saafscan.printstrip import render_strip as render_template
from saafscan.roi import STRIP_SIZE, ZONE_NOMINAL, zones_from_strip
from saafscan.synthetic import training_strip_canvas


def test_there_is_one_target_per_class_the_model_can_emit() -> None:
    """Five targets, matching TRAINING_CLASSES exactly."""
    from saafscan.synthetic import TRAINING_CLASSES

    assert TARGETS == TRAINING_CLASSES


def test_every_target_has_all_three_zones() -> None:
    for label in TARGETS:
        assert set(zone_colours(label)) == {"control", "starch", "detergent"}, label


def test_the_control_is_the_same_patch_a_real_strip_carries() -> None:
    """A target that differs from a real strip calibrates nothing.

    The control is what ties a printed target to a reagent-loaded one, so it
    has to read back exactly as saafscan.printstrip's own template does. It is
    compared against that read-back rather than the raw CONTROL_COLOUR because
    at strip resolution the dotted outline bleeds about 2 ΔE into every window
    — identically for both, which is the point.
    """
    real = zones_from_strip(render_template(*STRIP_SIZE))
    for label in TARGETS:
        target = zones_from_strip(render_strip(*STRIP_SIZE, label))
        np.testing.assert_allclose(target["control"], real["control"], atol=1.0, err_msg=label)
        assert (
            delta_e(
                np.asarray(target["control"], np.float32), np.asarray(CONTROL_COLOUR, np.float32)
            )
            < 5.0
        ), label


def test_each_zone_reads_back_as_the_colour_it_was_painted() -> None:
    """The windows must measure the target, not the paper around it.

    This is the failure photo5 taught: colour outside a measurement window is
    invisible to the verdict, and untouched paper reads as a fully decoloured
    — i.e. reacted — zone. Compared in ΔE, not per channel, because that is the
    only unit the classifier ever sees, and a couple of ΔE of outline bleed is
    nothing against CLEAR = 40.
    """
    for label in TARGETS:
        zones = zones_from_strip(render_strip(*STRIP_SIZE, label))
        for name, rgb in zone_colours(label).items():
            d = delta_e(np.asarray(zones[name], np.float32), np.asarray(rgb, np.float32))
            assert d < 5.0, f"{label}/{name}: reads back {d:.1f} ΔE from what it was painted"


def test_reactions_are_on_the_right_side_of_the_fired_threshold() -> None:
    """A 'starch' target that did not fire would calibrate us to expect nothing."""
    for label in TARGETS:
        if label == "inconclusive":
            continue  # sitting inside the grey band is its whole job — see below
        for name in ("starch", "detergent"):
            d = delta_e(
                np.asarray(zone_colours(label)[name], np.float32),
                np.asarray(ZONE_NOMINAL[name], np.float32),
            )
            expected_fired = label in (name, "both")
            if expected_fired:
                assert d >= FIRED, f"{label}/{name}: only {d:.1f}, below FIRED"
            else:
                assert d < CLEAR, f"{label}/{name}: {d:.1f} drifts into the grey band"


def test_the_inconclusive_target_is_ambiguous_not_a_near_miss() -> None:
    """It must sit inside CLEAR..FIRED, not just short of FIRED.

    Ambiguity is the classifier's refusal to guess, so a target that happens
    to land a whisker under the threshold would be an accidental pass rather
    than a test of the behaviour we care about. One zone doubtful and one
    clean is also the shape a real half-dipped strip has, and the rule table
    calls it inconclusive on either zone alone.
    """
    starch = delta_e(
        np.asarray(zone_colours("inconclusive")["starch"], np.float32),
        np.asarray(ZONE_NOMINAL["starch"], np.float32),
    )
    assert CLEAR <= starch < FIRED, f"starch: {starch:.1f} outside the ambiguity band"

    detergent = delta_e(
        np.asarray(zone_colours("inconclusive")["detergent"], np.float32),
        np.asarray(ZONE_NOMINAL["detergent"], np.float32),
    )
    assert detergent < CLEAR, "the untouched zone must still read as untouched"


def test_a_printed_target_reads_back_as_its_own_class() -> None:
    """The point of the whole exercise, end to end.

    Render a target, push it through the same read-back the live demo uses,
    and the verdict must be the label it was drawn as. If this fails there is
    no point photographing anything: the imaging chain cannot tell these
    colours apart, and no amount of chemistry will fix that.
    """
    for label in TARGETS:
        verdict = classify_zones(zones_from_strip(render_strip(*STRIP_SIZE, label)))
        assert verdict.adulterant == label, f"{label}: got {verdict.adulterant}"


def test_the_sheet_is_a4_and_holds_a_class_per_strip() -> None:
    """A page you can print on the same printer as the real template."""
    sheet = render_sheet(dpi=150)
    assert sheet.shape[0] == pytest.approx(1754, abs=2)  # 297 mm tall
    assert sheet.shape[1] == pytest.approx(1240, abs=2)  # 210 mm wide


def test_the_target_sheet_does_not_instruct_reagent_application() -> None:
    """An operator following the template header would pipette onto their own
    reference, which is exactly the contamination the RED box exists to prevent.
    """
    target = sheet_header(target_page())
    template = sheet_header()

    assert "apply reagent" not in target[1]
    assert "apply reagent" in template[1]
    assert "COLOUR REFERENCE ONLY" in target[1]
    assert "never ingest these" in target[1]


def test_the_training_strip_is_the_strip_we_print() -> None:
    """Train on exactly the object the camera will be shown.

    They already diverged once: synthetic drew solid bands with no dotted
    rules on a 238-grey page while the printer puts proud boxes and dotted
    rules on white paper. The model learned the drawing rather than the
    chemistry — 1.000 on held-out synthetic lighting, but 2/5 on the real
    printed strips it would actually be shown. Pixel equality, not
    similarity, is the guarantee that cannot drift apart again.
    """
    for label in TARGETS:
        np.testing.assert_array_equal(
            training_strip_canvas(label),
            render_strip(*STRIP_SIZE, label),
            err_msg=label,
        )


def test_strip_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        render_strip(0, 100, "none")
