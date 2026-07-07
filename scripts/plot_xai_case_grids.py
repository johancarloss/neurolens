"""Generate 2x3 comparison grids for the Phase 3 XAI thesis figures.

Two cases, each rendered as a 2 (arch: VGG16 / ResNet50) x 3 (technique:
Grad-CAM / LIME / SHAP) grid of pre-generated saliency overlays, with the
underlying Grad-CAM sparsity annotated per cell:

    Case 1 (ERROR)   Te-gl_277.jpg  — a real glioma read as "notumor"
    Case 2 (CORRECT) Te-gl_113.jpg  — the model got a confident glioma right

The overlays themselves come from the Phase 3 batch run (already colored
heatmaps over the MRI); this script only assembles them into the publication
figures with titles, row/column labels, and per-cell sparsity annotations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

from neurolens.db.repository import get_connection

XAI_DIR = Path(__file__).parent.parent / "kernel-output" / "phase-3-xai" / "xai"
OUT_DIR = Path(__file__).parent.parent / "docs" / "public" / "assets"
# The dataset lives outside the repo (Kaggle download). Original MRIs are read
# only to render the "no overlay" reference column in each figure.
DATASET_ROOT = Path.home() / "datasets" / "brain-tumor-mri" / "Testing" / "glioma"

ARCHS = ["vgg16", "resnet50"]
ARCH_LABEL = {"vgg16": "VGG16", "resnet50": "ResNet50"}
METHODS = ["gradcam", "lime", "shap"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "lime": "LIME", "shap": "SHAP"}


@dataclass(frozen=True)
class CaseSpec:
    """One case: an image filename explained by both architectures.

    Prediction ids differ per arch (one row per fold + arch in the DB); we look
    them up so we can pull the correct sparsity and locate the PNG on disk.
    """

    slug: str
    image_filename: str
    title: str
    subtitle: str


CASES = [
    CaseSpec(
        slug="error",
        image_filename="Te-gl_277.jpg",
        title="Case 1 (error) — real glioma read as “notumor”",
        subtitle="True: glioma  ·  Predicted: notumor  (both architectures)",
    ),
    CaseSpec(
        slug="correct",
        image_filename="Te-gl_113.jpg",
        title="Case 2 (contrast) — confident correct glioma",
        subtitle="True: glioma  ·  Predicted: glioma  (both architectures)",
    ),
]


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fetch_pred_and_sparsity(image_filename: str) -> dict[str, dict[str, float | int]]:
    """Return {arch: {pred_id, gc, lime, shap}} for one image, both architectures.

    Sparsity is annotated on each cell as a small caption; Grad-CAM's is the one
    that drove Case selection (Finding #6), so it gets the emphasis in text.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT xa.metadata->>'arch' AS arch, p.id,
                   xc.sparsity_gradcam, xc.sparsity_lime, xc.sparsity_shap
            FROM neurolens.xai_comparisons xc
            JOIN neurolens.xai_artifacts xa ON xa.prediction_id = xc.prediction_id
            JOIN neurolens.predictions p ON p.id = xc.prediction_id
            WHERE p.image_filename = %s
            GROUP BY xa.metadata->>'arch', p.id,
                     xc.sparsity_gradcam, xc.sparsity_lime, xc.sparsity_shap
            """,
            (image_filename,),
        )
        out: dict[str, dict[str, float | int]] = {}
        for arch, pred_id, gc, li, sh in cur.fetchall():
            out[arch] = {
                "pred_id": int(pred_id),
                "gradcam": float(gc),
                "lime": float(li),
                "shap": float(sh),
            }
        return out


def _render_case(case: CaseSpec) -> Path:
    """Render one case as a 2x4 figure (Original + 3 techniques) and return the path.

    The Original column repeats the same MRI on both rows so each row reads
    left-to-right as "what the scan shows" -> "what each XAI thinks the model saw."
    Repeating the reference beats a shared cell across rows because the visual
    alignment (original next to each arch's overlays) makes the comparison
    faster to parse.
    """
    lookup = _fetch_pred_and_sparsity(case.image_filename)
    original = imread(DATASET_ROOT / case.image_filename)

    columns = ["original", *METHODS]
    column_labels = {"original": "MRI (original)", **METHOD_LABEL}

    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(12.5, 6.8),
        constrained_layout=True,
    )
    fig.suptitle(
        f"{case.title}\n{case.subtitle}",
        fontsize=12,
        fontweight="bold",
        linespacing=1.4,
    )

    for row, arch in enumerate(ARCHS):
        pred_id = lookup[arch]["pred_id"]
        for col, key in enumerate(columns):
            ax = axes[row, col]
            if key == "original":
                ax.imshow(original, cmap="gray")
            else:
                ax.imshow(imread(XAI_DIR / f"{arch}_{pred_id}_{key}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col == 0:
                ax.set_ylabel(ARCH_LABEL[arch], fontsize=11, fontweight="bold", labelpad=10)
            if row == 0:
                ax.set_title(column_labels[key], fontsize=11, fontweight="bold", pad=6)

            # Sparsity caption only on the technique columns; blank under Original
            # so the eye reads "reference (no annotation) -> annotated overlays".
            if key != "original":
                sparsity = lookup[arch][key]
                caption = f"sparsity {sparsity * 100:.1f}%"
            else:
                caption = ""
            ax.text(
                0.5,
                -0.05,
                caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                color="#444",
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"phase-3-case-{case.slug}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    _load_env()
    for case in CASES:
        path = _render_case(case)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
