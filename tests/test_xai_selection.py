"""Tests for src/neurolens/xai/selection.py (ranking logic, no DB)."""

from __future__ import annotations

from collections import Counter

from neurolens.xai.selection import classify_reason, stratified_select


def test_glioma_to_notumor_is_tier_1() -> None:
    """The dangerous false negative is always the top priority."""
    assert classify_reason("glioma", "notumor", 0.5, is_correct=False) == "glioma_to_notumor"


def test_confident_error_is_tier_2() -> None:
    """A wrong, high-confidence prediction (not glioma->notumor) is tier 2."""
    assert classify_reason("meningioma", "pituitary", 0.9, is_correct=False) == "confident_error"


def test_confident_glioma_correct_is_tier_3() -> None:
    """A correct, very-confident glioma is tier 3 (what a right answer looks like)."""
    assert classify_reason("glioma", "glioma", 0.97, is_correct=True) == "confident_glioma_correct"


def test_uninteresting_prediction_is_none() -> None:
    """A correct non-glioma with ordinary confidence is not a target."""
    assert classify_reason("pituitary", "pituitary", 0.8, is_correct=True) is None
    # low-confidence error is below the threshold
    assert classify_reason("notumor", "glioma", 0.6, is_correct=False) is None


def _rows(prefix: str, reason: str, n: int) -> list[tuple[str, str, str, float, bool]]:
    """Helper to mint n distinct rows of a given tier."""
    recipes = {
        "glioma_to_notumor": ("glioma", "notumor", 0.8, False),
        "confident_error": ("meningioma", "pituitary", 0.9, False),
        "confident_glioma_correct": ("glioma", "glioma", 0.99, True),
    }
    tl, pl, conf, ok = recipes[reason]
    return [(f"{prefix}{i}.jpg", tl, pl, conf, ok) for i in range(n)]


def test_stratified_mix_respects_quotas() -> None:
    """With plenty in each tier, the mix matches the default strata (0.5/0.3/0.2)."""
    rows = (
        _rows("err", "glioma_to_notumor", 50)
        + _rows("cor", "confident_glioma_correct", 50)
        + _rows("oth", "confident_error", 50)
    )
    result = stratified_select(rows, num_images=10)
    counts = Counter(s.reason for s in result)
    assert counts["glioma_to_notumor"] == 5
    assert counts["confident_glioma_correct"] == 3
    assert counts["confident_error"] == 2


def test_backfill_when_a_tier_is_short() -> None:
    """If a tier can't meet its quota, the rest is back-filled to reach num_images."""
    rows = _rows("err", "glioma_to_notumor", 9)  # only tier-1 available
    result = stratified_select(rows, num_images=5)
    assert len(result) == 5
    assert all(s.reason == "glioma_to_notumor" for s in result)


def test_dedupe_keeps_highest_priority_occurrence() -> None:
    """The same image across folds collapses to its highest-priority row."""
    rows = [
        ("img.jpg", "glioma", "glioma", 0.99, True),  # tier 3 occurrence
        ("img.jpg", "glioma", "notumor", 0.55, False),  # tier 1 occurrence (should win)
    ]
    result = stratified_select(rows, num_images=10)
    assert len(result) == 1
    assert result[0].reason == "glioma_to_notumor"


def test_respects_num_images_cap() -> None:
    """Selection never returns more than requested."""
    rows = _rows("err", "glioma_to_notumor", 20)
    assert len(stratified_select(rows, num_images=5)) == 5


def test_result_ordered_by_tier_then_confidence() -> None:
    """Output is ordered tier-first, then by descending confidence."""
    rows = [
        ("a.jpg", "glioma", "notumor", 0.6, False),  # tier 1
        ("b.jpg", "glioma", "glioma", 0.99, True),  # tier 3
        ("c.jpg", "meningioma", "glioma", 0.9, False),  # tier 2
    ]
    result = stratified_select(rows, num_images=10)
    assert [s.image_filename for s in result] == ["a.jpg", "c.jpg", "b.jpg"]
