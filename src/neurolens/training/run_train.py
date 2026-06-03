"""End-to-end, architecture-agnostic training entry point.

Called from ``kernel/runner/run.py`` after the kernel bootstrap clones the
repo and installs deps. The kernel passes ``config_profile`` (e.g.
``"smoke_micro"``, ``"vgg16"``, ``"resnet50"``) which selects the YAML pair
under ``configs/{profile}_stage{1,2}.yaml``. The architecture itself comes
from ``config.arch`` — this module never hardcodes a model.

Pipeline (per fold):
    1. Load stage 1 + stage 2 configs for the requested profile
    2. Set seeds for reproducibility
    3. Discover dataset (kaggle_paths) and build train/test ImageFolders
    4. Apply optional smoke-test sample caps (``train/test_samples_per_class``)
    5. Stratified K-fold on train indices; pick ``target_fold`` from config
       (None -> all folds run sequentially)
    6. STAGE 1 (head only): build the arch at stage=1, train ``epochs`` epochs
    7. STAGE 2 (fine-tune late block): unfreeze via the factory, swap optimizer
       (LR=1e-4), train another ``epochs`` epochs
    8. Evaluate on TEST set; BULK-persist predictions to Postgres
    9. Save best checkpoint; finish W&B + Postgres runs

Both ``config_profile`` and ``target_fold`` are config-driven (no env-var
surprises). Switch modes by editing ``configs/active_run.yaml`` and pushing.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from neurolens.config import TrainConfig, load_config
from neurolens.data.dataset import CLASSES, build_dataset
from neurolens.data.kaggle_paths import discover_brain_tumor_dataset
from neurolens.data.transforms import eval_transforms, train_transforms
from neurolens.db.repository import insert_predictions_bulk
from neurolens.models.factory import build_model, unfreeze_for_stage2
from neurolens.tracking.composite import CompositeLogger
from neurolens.training.cv import stratified_kfold_indices
from neurolens.training.evaluator import evaluate_test_set
from neurolens.training.trainer import Trainer

# The universal runner kernel URL (arch-agnostic; one kernel for every job).
KAGGLE_KERNEL_URL = "https://www.kaggle.com/code/johancarloss/neurolens-runner"


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit_sha(repo_dir: Path) -> str | None:
    """Return ``git rev-parse --short HEAD`` for the cloned repo (best effort)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _build_loaders(
    train_subset: Subset,
    val_subset: Subset,
    test_dataset: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with sane workers + pin_memory settings."""
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _build_logger(
    config: TrainConfig,
    fold_idx: int,
    stage_phase: str,
    git_commit: str | None,
    jsonl_dir: Path,
) -> CompositeLogger:
    """Construct a CompositeLogger with arch-parametrized naming for this run."""
    arch = config.arch
    return CompositeLogger(
        project="neurolens",
        experiment=f"{arch}-5fold",
        config={
            **config.model_dump(),
            "fold": fold_idx,
            "stage_phase": stage_phase,
        },
        jsonl_dir=jsonl_dir,
        wandb_group=f"{arch}-{stage_phase}",
        wandb_name=f"{arch}-{stage_phase}-f{fold_idx}",
        wandb_tags=[arch, stage_phase, f"fold{fold_idx}"],
        git_commit=git_commit,
        kaggle_kernel_url=KAGGLE_KERNEL_URL,
    )


def _persist_predictions(
    run_id: int | None,
    test_dataset: Any,
    test_metrics: dict[str, Any],
) -> None:
    """Batch-insert all test predictions into Postgres in a single round-trip.

    Previously this used a tight loop of ``insert_prediction`` (one SSL
    handshake per row). For 1600 rows on remote Postgres that took ~30 min
    per fold. The batched version using ``execute_values`` typically takes
    a few seconds.
    """
    if run_id is None:
        return
    targets = test_metrics["targets"]
    preds = test_metrics["preds"]
    probs = test_metrics["probs"]
    samples = test_dataset.samples  # ImageFolder: list[tuple[path, class_idx]]

    rows: list[dict[str, Any]] = []
    for i, (true_idx, pred_idx, prob_vec) in enumerate(zip(targets, preds, probs, strict=True)):
        img_path, _ = samples[i]
        rows.append(
            {
                "image_path": str(img_path),
                "image_filename": Path(img_path).name,
                "true_label": CLASSES[int(true_idx)],
                "predicted_label": CLASSES[int(pred_idx)],
                "probs": {cls: float(prob_vec[j]) for j, cls in enumerate(CLASSES)},
                "confidence": float(prob_vec[int(pred_idx)]),
            }
        )

    try:
        n_inserted = insert_predictions_bulk(run_id, rows)
        print(f"[run_train] persisted {n_inserted} predictions to Postgres (bulk)")
    except Exception as exc:  # noqa: BLE001 — never break training on a logging fail
        print(f"[run_train] WARNING: insert_predictions_bulk failed: {exc!r}")


def run_one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    full_train_dataset: Any,
    test_dataset: Any,
    config_stage1: TrainConfig,
    config_stage2: TrainConfig,
    device: torch.device,
    output_dir: Path,
    git_commit: str | None,
) -> dict[str, float]:
    """Run stage 1 then stage 2 for one fold; evaluate on the test set.

    Returns a small summary dict (fold idx, test accuracy, macro F1).
    """
    arch = config_stage1.arch
    train_subset = Subset(full_train_dataset, train_idx.tolist())
    val_subset = Subset(full_train_dataset, val_idx.tolist())
    train_loader, val_loader, test_loader = _build_loaders(
        train_subset,
        val_subset,
        test_dataset,
        batch_size=config_stage1.batch_size,
        num_workers=config_stage1.num_workers,
    )

    # === STAGE 1: head only =================================================
    model = build_model(arch, num_classes=4, stage=1, dropout=config_stage1.dropout)
    logger_s1 = _build_logger(
        config=config_stage1,
        fold_idx=fold_idx,
        stage_phase="stage1",
        git_commit=git_commit,
        jsonl_dir=output_dir / "jsonl",
    )
    # Stage 1 trains only the head. Filtering by ``requires_grad`` is
    # arch-agnostic (and identical to ``model.classifier.parameters()`` for
    # VGG16, since only the head is unfrozen here).
    stage1_params = [p for p in model.parameters() if p.requires_grad]
    trainer_s1 = Trainer(
        model=model,
        optimizer=torch.optim.Adam(stage1_params, lr=config_stage1.lr),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        logger=logger_s1,
        checkpoint_dir=output_dir / "checkpoints" / f"{arch}_fold{fold_idx}_stage1",
        save_best_only=config_stage1.save_best_only,
        early_stopping_patience=config_stage1.early_stopping_patience,
    )
    print(f"[fold {fold_idx}] === STAGE 1 (head only, lr={config_stage1.lr}) ===")
    trainer_s1.fit(train_loader, val_loader, epochs=config_stage1.epochs)
    logger_s1.finish(status="completed")

    # === STAGE 2: unfreeze late block + new optimizer ======================
    unfreeze_for_stage2(model, arch)
    logger_s2 = _build_logger(
        config=config_stage2,
        fold_idx=fold_idx,
        stage_phase="stage2",
        git_commit=git_commit,
        jsonl_dir=output_dir / "jsonl",
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainer_s2 = Trainer(
        model=model,
        optimizer=torch.optim.Adam(trainable_params, lr=config_stage2.lr),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        logger=logger_s2,
        checkpoint_dir=output_dir / "checkpoints" / f"{arch}_fold{fold_idx}_stage2",
        save_best_only=config_stage2.save_best_only,
        early_stopping_patience=config_stage2.early_stopping_patience,
    )
    print(f"[fold {fold_idx}] === STAGE 2 (fine-tune late block, lr={config_stage2.lr}) ===")
    trainer_s2.fit(train_loader, val_loader, epochs=config_stage2.epochs)

    # === TEST EVAL ===========================================================
    print(f"[fold {fold_idx}] === TEST SET EVALUATION ===")
    test_metrics = evaluate_test_set(model, test_loader, device, classes=CLASSES)
    print(
        f"[fold {fold_idx}] test_acc={test_metrics['overall_accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    # Persist per-image predictions (1600 rows expected on full test set)
    _persist_predictions(logger_s2.run_id, test_dataset, test_metrics)

    # Save final fold checkpoint at a stable, arch-namespaced path
    final_ckpt = output_dir / "checkpoints" / f"{arch}_fold{fold_idx}_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_ckpt)

    logger_s2.finish(
        status="completed",
        final_metrics={
            "test_accuracy": test_metrics["overall_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            **{f"test_f1_{cls}": test_metrics["per_class"][cls]["f1-score"] for cls in CLASSES},
        },
    )

    return {
        "fold": fold_idx,
        "test_accuracy": test_metrics["overall_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }


def _cap_samples_per_class(dataset: Any, per_class: int | None, seed: int = 42) -> Any:
    """Return a Subset capping the number of samples per class.

    When ``per_class`` is None, returns the dataset unchanged (no Subset wrapping).
    Used by smoke configs to keep micro runs under ~2 minutes.
    """
    if per_class is None:
        return dataset
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    classes = sorted(set(targets.tolist()))
    selected: list[int] = []
    for cls in classes:
        idxs = np.where(targets == cls)[0]
        chosen = rng.choice(idxs, size=min(per_class, len(idxs)), replace=False)
        selected.extend(chosen.tolist())
    selected = sorted(selected)

    class _CappedView(Subset):
        """Subset that also exposes ``.targets`` and ``.samples`` for compatibility."""

        @property
        def targets(self) -> list[int]:
            return [self.dataset.targets[i] for i in self.indices]  # type: ignore[index]

        @property
        def samples(self) -> list[tuple[str, int]]:
            return [self.dataset.samples[i] for i in self.indices]  # type: ignore[index]

    return _CappedView(dataset, selected)


def main(config_profile: str | None = None) -> None:
    """Entry point invoked from the Kaggle kernel.

    Args:
        config_profile: name of the config pair to load (e.g. ``"vgg16"``,
            ``"resnet50"``, ``"smoke_micro"``). Resolves to
            ``configs/{profile}_stage1.yaml`` and ``configs/{profile}_stage2.yaml``.
            When None, falls back to the ``CONFIG_PROFILE`` env var, then to
            ``"vgg16"``. The architecture comes from ``config.arch``.
    """
    repo_dir = Path("/kaggle/working/neurolens-repo")
    output_dir = Path("/kaggle/working")
    git_commit = _git_commit_sha(repo_dir)

    profile = config_profile or os.environ.get("CONFIG_PROFILE") or "vgg16"
    print(f"[run_train] config_profile={profile}")

    config_stage1 = load_config(repo_dir / "configs" / f"{profile}_stage1.yaml")
    config_stage2 = load_config(repo_dir / "configs" / f"{profile}_stage2.yaml")
    _set_seeds(config_stage1.seed)
    print(f"[run_train] arch={config_stage1.arch}")

    paths = discover_brain_tumor_dataset()
    full_train = build_dataset(
        paths.data_root,
        transform=train_transforms(
            image_size=config_stage1.image_size,
            shear_range=config_stage1.augmentation.shear_range,
            zoom_range=config_stage1.augmentation.zoom_range,
            horizontal_flip=config_stage1.augmentation.horizontal_flip,
        ),
        split="train",
    )
    test_dataset = build_dataset(
        paths.data_root,
        transform=eval_transforms(image_size=config_stage1.image_size),
        split="test",
    )

    # Apply smoke-test caps if configured
    if config_stage1.train_samples_per_class is not None:
        print(
            f"[run_train] capping train at {config_stage1.train_samples_per_class}"
            f" samples/class (original: {len(full_train)})"
        )
        full_train = _cap_samples_per_class(
            full_train, config_stage1.train_samples_per_class, seed=config_stage1.seed
        )
        print(f"[run_train] capped train size: {len(full_train)}")
    if config_stage1.test_samples_per_class is not None:
        print(
            f"[run_train] capping test at {config_stage1.test_samples_per_class}"
            f" samples/class (original: {len(test_dataset)})"
        )
        test_dataset = _cap_samples_per_class(
            test_dataset, config_stage1.test_samples_per_class, seed=config_stage1.seed
        )
        print(f"[run_train] capped test size: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_train] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[run_train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[run_train] CUDA: {torch.version.cuda}")

    # target_fold comes from the config (deterministic, no env-var surprises)
    target_fold = config_stage1.target_fold
    if target_fold is not None:
        print(f"[run_train] running ONLY fold {target_fold} (config.target_fold)")
    else:
        print("[run_train] config.target_fold=None — running ALL folds sequentially")

    results: list[dict[str, float]] = []
    for fold, train_idx, val_idx in stratified_kfold_indices(
        full_train.targets, n_splits=config_stage1.cv_folds, seed=config_stage1.seed
    ):
        if target_fold is not None and fold != target_fold:
            continue
        results.append(
            run_one_fold(
                fold_idx=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                full_train_dataset=full_train,
                test_dataset=test_dataset,
                config_stage1=config_stage1,
                config_stage2=config_stage2,
                device=device,
                output_dir=output_dir,
                git_commit=git_commit,
            )
        )

    print("=" * 60)
    print("FOLD RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"  Fold {int(r['fold'])}: "
            f"test_acc={r['test_accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}"
        )
    if len(results) > 1:
        accs = [r["test_accuracy"] for r in results]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"  Mean test_acc: {mean_acc:.4f}  Std: {std_acc:.4f}")
    print("=" * 60)
