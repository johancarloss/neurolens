"""Generate the VGG16 vs ResNet50 side-by-side confusion matrices (Phase 2).

Pulls the durable per-image predictions from PostgreSQL (the 5 production
stage-2 runs per architecture), aggregates a 4x4 confusion matrix for each,
and renders them side by side (row-normalized = recall) to a PNG.

Run IDs are passed explicitly so the smoke run (80 predictions) is never mixed
into the production aggregate (5 x 1600 = 8000 predictions per architecture).

Usage:
    uv run python scripts/plot_confusion_matrices.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurolens.db.repository import get_connection

# Production stage-2 run IDs (1600 predictions each). The single resnet50 smoke
# run (id 24, 80 predictions) is deliberately excluded.
VGG16_RUN_IDS = [14, 16, 18, 20, 22]
RESNET50_RUN_IDS = [26, 28, 30, 32, 34]

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
OUT_PATH = (
    Path(__file__).parent.parent / "docs" / "public" / "assets" / "phase-2-confusion-matrices.png"
)


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def confusion_matrix_for(run_ids: list[int]) -> np.ndarray:
    """Aggregate a [true, pred] count matrix over the given runs' predictions."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT true_label, predicted_label FROM neurolens.predictions WHERE run_id = ANY(%s);",
            (run_ids,),
        )
        for true_label, pred_label in cur.fetchall():
            counts[(true_label, pred_label)] += 1
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for i, true_cls in enumerate(CLASSES):
        for j, pred_cls in enumerate(CLASSES):
            matrix[i, j] = counts[(true_cls, pred_cls)]
    return matrix


def _draw(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    """Draw one row-normalized (recall) heatmap with count + percentage labels."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.clip(row_sums, 1, None)
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            pct = normalized[i, j] * 100
            color = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def main() -> None:
    _load_env()
    vgg = confusion_matrix_for(VGG16_RUN_IDS)
    resnet = confusion_matrix_for(RESNET50_RUN_IDS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw(axes[0], vgg, f"VGG16  (acc {np.trace(vgg) / vgg.sum() * 100:.2f}%)")
    _draw(axes[1], resnet, f"ResNet50  (acc {np.trace(resnet) / resnet.sum() * 100:.2f}%)")
    fig.suptitle(
        "Aggregate confusion matrices over 5 folds (8000 predictions each) — row-normalized (recall)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"saved: {OUT_PATH}")
    print(f"VGG16 total predictions:    {vgg.sum()}")
    print(f"ResNet50 total predictions: {resnet.sum()}")


if __name__ == "__main__":
    main()
