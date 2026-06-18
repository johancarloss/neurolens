"""Select which test images to explain (Phase 3).

XAI is expensive (LIME alone is ~30-90s/image), so we don't explain all 1,600
test images — we pick the most *informative* ones. Phase 2 told us where to
look: glioma is the architecture-independent weak spot, and glioma->notumor is
the most dangerous error (a missed cancer). Each candidate is tagged by tier:

    glioma_to_notumor          the dangerous false negative (the headline error)
    confident_error            the model was sure and wrong (other classes)
    confident_glioma_correct   what a *correct*, confident glioma looks like

We sample a **stratified mix** of these so the analysis can contrast "where the
model looks when it errs" vs "when it gets glioma right" — not only errors.

The DB call and the ranking are split so the ranking is unit-testable without
a database.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from neurolens.db.repository import get_connection

# A raw prediction row: (image_filename, true_label, predicted_label, confidence, is_correct)
PredictionRow = tuple[str, str, str, float, bool]

# Lower number = higher priority (used for ordering and backfill).
_TIER_ORDER = {
    "glioma_to_notumor": 0,
    "confident_error": 1,
    "confident_glioma_correct": 2,
}

# Default mix: half the dangerous error, a third correct-glioma contrast, the
# rest other confident errors. Fractions need not sum to exactly 1 (backfill
# tops up to num_images from whatever candidates remain).
DEFAULT_STRATA: dict[str, float] = {
    "glioma_to_notumor": 0.5,
    "confident_glioma_correct": 0.3,
    "confident_error": 0.2,
}


@dataclass(frozen=True)
class SelectedImage:
    """One image chosen for XAI analysis, with why it was chosen."""

    image_filename: str
    true_label: str
    predicted_label: str
    confidence: float
    is_correct: bool
    reason: str


def classify_reason(
    true_label: str,
    predicted_label: str,
    confidence: float,
    is_correct: bool,
    *,
    confident_error_min: float = 0.7,
    confident_correct_min: float = 0.95,
) -> str | None:
    """Return the priority tier for a prediction, or None if it isn't a target."""
    if true_label == "glioma" and predicted_label == "notumor":
        return "glioma_to_notumor"
    if not is_correct and confidence > confident_error_min:
        return "confident_error"
    if is_correct and confidence > confident_correct_min and true_label == "glioma":
        return "confident_glioma_correct"
    return None


def _dedupe(rows: Iterable[PredictionRow]) -> dict[str, SelectedImage]:
    """Collapse rows to one SelectedImage per filename (highest-priority occurrence).

    An image appears in every fold's test eval, so the same filename shows up in
    many rows; we keep its highest-priority tier.
    """
    best: dict[str, SelectedImage] = {}
    for image_filename, true_label, predicted_label, confidence, is_correct in rows:
        reason = classify_reason(true_label, predicted_label, float(confidence), bool(is_correct))
        if reason is None:
            continue
        candidate = SelectedImage(
            image_filename, true_label, predicted_label, float(confidence), bool(is_correct), reason
        )
        current = best.get(image_filename)
        if current is None or _TIER_ORDER[reason] < _TIER_ORDER[current.reason]:
            best[image_filename] = candidate
    return best


def stratified_select(
    rows: Iterable[PredictionRow],
    num_images: int,
    strata: dict[str, float] | None = None,
) -> list[SelectedImage]:
    """Pick a stratified mix of target images.

    Each tier gets ``round(num_images * fraction)`` slots (its highest-confidence
    members). If the quotas under-fill (a tier is short), the remainder is
    back-filled from the leftover candidates in priority order, so the result is
    as close to ``num_images`` as the available data allows.
    """
    strata = strata or DEFAULT_STRATA
    best = _dedupe(rows)

    by_reason: dict[str, list[SelectedImage]] = defaultdict(list)
    for selected in best.values():
        by_reason[selected.reason].append(selected)
    for group in by_reason.values():
        group.sort(key=lambda s: -s.confidence)

    chosen: list[SelectedImage] = []
    chosen_files: set[str] = set()
    for reason, fraction in strata.items():
        quota = round(num_images * fraction)
        for selected in by_reason.get(reason, [])[:quota]:
            chosen.append(selected)
            chosen_files.add(selected.image_filename)

    if len(chosen) < num_images:
        leftover = sorted(
            (s for s in best.values() if s.image_filename not in chosen_files),
            key=lambda s: (_TIER_ORDER[s.reason], -s.confidence),
        )
        for selected in leftover:
            if len(chosen) >= num_images:
                break
            chosen.append(selected)

    chosen.sort(key=lambda s: (_TIER_ORDER[s.reason], -s.confidence))
    return chosen[:num_images]


_CANDIDATE_SQL = """
    SELECT image_filename, true_label, predicted_label, confidence, is_correct
    FROM neurolens.predictions
    WHERE run_id = ANY(%s)
"""


def select_images(
    run_ids: Sequence[int],
    num_images: int,
    strata: dict[str, float] | None = None,
) -> list[SelectedImage]:
    """Query predictions for the given runs and return a stratified target mix."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(_CANDIDATE_SQL, (list(run_ids),))
        rows = cur.fetchall()
    return stratified_select(rows, num_images, strata)
