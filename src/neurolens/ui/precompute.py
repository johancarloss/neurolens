"""Pre-computed example results for the demo (Phase 4, Bloco 4).

The curated examples must open instantly, but a live run is ~4 min on CPU
(LIME/SHAP dominate). We run each example once, offline, and persist the overlays
(PNG) + the numeric results (JSON) so the UI can serve them from disk. Free
uploads still compute live.

Layout on disk (one directory per example, keyed by the image stem):

    precomputed/<stem>/
        original.png
        <arch>_gradcam.png · <arch>_lime.png · <arch>_shap.png
        result.json   # per-arch probs, predicted_label, times_ms
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from neurolens.ui.inference import ArchResult, ExplainResult

_TECHNIQUES = ("gradcam", "lime", "shap")


def result_dir(precomputed_root: str | Path, stem: str) -> Path:
    """Directory holding one example's pre-computed artifacts."""
    return Path(precomputed_root) / stem


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_result(result: ExplainResult, out_dir: str | Path) -> None:
    """Persist one ``ExplainResult`` (overlays as PNG, numbers as JSON)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_png(result.original, out / "original.png")

    meta: dict[str, dict] = {"per_arch": {}}
    for arch, r in result.per_arch.items():
        _save_png(r.gradcam, out / f"{arch}_gradcam.png")
        _save_png(r.lime, out / f"{arch}_lime.png")
        _save_png(r.shap, out / f"{arch}_shap.png")
        meta["per_arch"][arch] = {
            "probs": r.probs,
            "predicted_label": r.predicted_label,
            "times_ms": r.times_ms,
        }
    (out / "result.json").write_text(json.dumps(meta, indent=2))


def load_result(out_dir: str | Path, archs: list[str]) -> ExplainResult | None:
    """Load a pre-computed result, or ``None`` if it is missing/incomplete.

    Returning ``None`` lets the UI fall back to a live run — so a missing cache
    degrades gracefully instead of raising.
    """
    out = Path(out_dir)
    meta_path = out / "result.json"
    original_path = out / "original.png"
    if not meta_path.exists() or not original_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    per_arch: dict[str, ArchResult] = {}
    for arch in archs:
        spec = meta.get("per_arch", {}).get(arch)
        pngs = {tech: out / f"{arch}_{tech}.png" for tech in _TECHNIQUES}
        if spec is None or not all(p.exists() for p in pngs.values()):
            return None
        per_arch[arch] = ArchResult(
            arch=arch,
            probs=spec["probs"],
            predicted_label=spec["predicted_label"],
            gradcam=_load_png(pngs["gradcam"]),
            lime=_load_png(pngs["lime"]),
            shap=_load_png(pngs["shap"]),
            times_ms=spec.get("times_ms", {}),
        )
    return ExplainResult(original=_load_png(original_path), per_arch=per_arch)
