"""Integration test for CompositeLogger (W&B + Postgres + JSONL).

Skipped automatically in CI because DATABASE_URL and WANDB_API_KEY are not set.
Run locally with both env vars exported to validate dual-write end-to-end.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

_HAS_DB = bool(os.environ.get("DATABASE_URL"))
_HAS_WANDB = bool(os.environ.get("WANDB_API_KEY"))


@pytest.fixture
def jsonl_dir() -> Path:
    """Provide a temporary directory for JSONL output (cleaned by tempfile)."""
    with tempfile.TemporaryDirectory(prefix="neurolens-jsonl-") as tmp:
        yield Path(tmp)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_DB and _HAS_WANDB),
    reason="Requires DATABASE_URL and WANDB_API_KEY env vars",
)
def test_composite_logger_writes_to_all_three(jsonl_dir: Path) -> None:
    """Full happy path: init -> log -> finish across all three sinks."""
    from neurolens.tracking.composite import CompositeLogger

    logger = CompositeLogger(
        project="neurolens",
        experiment="composite-logger-test",
        config={"arch": "test", "stage": 0, "lr": 0.001},
        jsonl_dir=jsonl_dir,
        wandb_tags=["test", "ci-skipped"],
    )

    # All three sinks should be live
    assert logger.run_id is not None, "Postgres run_id should be set"
    assert logger.wandb_run is not None, "W&B run should be set"
    assert logger.jsonl_path is not None, "JSONL path should be set"

    logger.log(epoch=0, phase="train", metrics={"loss": 1.5, "acc": 0.5})
    logger.log(epoch=0, phase="val", metrics={"loss": 1.3, "acc": 0.6})
    logger.finish(
        status="completed",
        final_metrics={"test_acc": 0.95, "test_f1": 0.94},
    )

    # JSONL was written with the expected events
    lines = logger.jsonl_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == [
        "run_start",
        "epoch_end",
        "epoch_end",
        "run_complete",
    ], f"Unexpected JSONL events sequence: {events}"


def test_composite_logger_requires_arch_in_config(jsonl_dir: Path) -> None:
    """Constructor refuses config without 'arch' key (pre-validation)."""
    from neurolens.tracking.composite import CompositeLogger

    with pytest.raises(ValueError, match="arch"):
        CompositeLogger(
            project="neurolens",
            experiment="bad-config-test",
            config={"stage": 0},  # missing 'arch'
            jsonl_dir=jsonl_dir,
        )


def test_composite_logger_requires_stage_in_config(jsonl_dir: Path) -> None:
    """Constructor refuses config without 'stage' key (pre-validation)."""
    from neurolens.tracking.composite import CompositeLogger

    with pytest.raises(ValueError, match="stage"):
        CompositeLogger(
            project="neurolens",
            experiment="bad-config-test",
            config={"arch": "test"},  # missing 'stage'
            jsonl_dir=jsonl_dir,
        )
