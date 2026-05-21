"""PostgreSQL repository for NeuroLens experiment tracking.

This module is the source-of-truth writer for runs, metrics, predictions
and XAI artifacts. Reads the connection string from the DATABASE_URL
environment variable; never accepts it as a function argument to avoid
credentials leaking into logs / repr.

Design decisions (see docs/private/blueprint/02-data-model.md):
- Connection per call (no global pool) — fits Kaggle Kernel single-process model.
- SSL is required (sslmode=require). VPS exposes Postgres over the internet.
- Tenacity retry (3 attempts, exponential backoff) on transient failures.
- All writes are committed inside the context manager; rollback on exception.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PgConnection  # noqa: N812 — third-party class name
from psycopg2.extras import Json, execute_values
from tenacity import retry, stop_after_attempt, wait_exponential

CONNECT_TIMEOUT_SECONDS = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_MAX_SECONDS = 10


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, max=RETRY_BACKOFF_MAX_SECONDS),
    reraise=True,
)
def _connect() -> PgConnection:
    """Open a new PostgreSQL connection with SSL and retry on transient failure.

    Reads DATABASE_URL from environment. The retry handles cases where the
    VPS is briefly unreachable; permanent issues (auth failures, wrong db)
    still surface after exhausting attempts.

    Raises:
        KeyError: if DATABASE_URL is not set.
        psycopg2.OperationalError: if connection cannot be established after retries.
    """
    db_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(
        db_url,
        sslmode="require",
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
    )


@contextmanager
def get_connection() -> Iterator[PgConnection]:
    """Yield a PostgreSQL connection; commit on success, rollback on exception.

    Use as a context manager around a unit of work:

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT ...")

    The connection is closed automatically after the block exits.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_experiment(name: str, description: str | None = None) -> int:
    """Insert an experiment row; return its id (idempotent on name).

    If an experiment with the same name already exists, returns its id
    without modifying the existing row.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO neurolens.experiments (name, description) "
            "VALUES (%s, %s) ON CONFLICT (name) DO NOTHING RETURNING id",
            (name, description),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT id FROM neurolens.experiments WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
        assert row is not None, "experiment row not found after upsert"
        return int(row[0])


def insert_run(
    experiment_name: str,
    arch: str,
    stage: int,
    hyperparams: dict[str, Any],
    *,
    fold: int | None = None,
    wandb_run_id: str | None = None,
    wandb_url: str | None = None,
    git_commit: str | None = None,
    kaggle_kernel_url: str | None = None,
    config_yaml: str | None = None,
    experiment_description: str | None = None,
) -> int:
    """Create a new run row; return its id.

    The experiment is upserted by name, so callers do not need to pre-create it.

    Args:
        experiment_name: name of the parent experiment (created if missing).
        arch: architecture identifier (e.g., 'vgg16', 'resnet50', 'none').
        stage: 0 for non-training events, 1 for head-only, 2 for fine-tuning.
        hyperparams: full hyperparameter dict (stored as JSONB).
        fold: 0-based fold index when running cross-validation, else None.
        wandb_run_id, wandb_url: cross-references to the W&B run.
        git_commit: SHA of the commit that produced this run (reproducibility).
        kaggle_kernel_url: URL of the Kaggle kernel that ran the training.
        config_yaml: raw YAML config (stored for reproducibility).
        experiment_description: optional description if experiment is new.

    Returns:
        Primary key (int) of the new run row.
    """
    experiment_id = insert_experiment(experiment_name, experiment_description)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO neurolens.runs (
                    experiment_id, arch, stage, fold,
                    hyperparams, wandb_run_id, wandb_url,
                    git_commit, kaggle_kernel_url, config_yaml
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
            (
                experiment_id,
                arch,
                stage,
                fold,
                Json(hyperparams),
                wandb_run_id,
                wandb_url,
                git_commit,
                kaggle_kernel_url,
                config_yaml,
            ),
        )
        row = cur.fetchone()
        assert row is not None, "INSERT ... RETURNING returned no row"
        return int(row[0])


def log_metric(
    run_id: int,
    epoch: int,
    phase: str,
    name: str,
    value: float,
) -> None:
    """Append one metric point to the metrics table.

    Phase must be one of 'train', 'val', 'test' (enforced by CHECK constraint).
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO neurolens.metrics (run_id, epoch, phase, metric_name, value)
                VALUES (%s, %s, %s, %s, %s)
                """,
            (run_id, epoch, phase, name, value),
        )


def finish_run(
    run_id: int,
    status: str,
    final_metrics: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """Mark a run as finished (completed | failed | cancelled).

    Sets finished_at to NOW() server-side and stores the final metrics.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                UPDATE neurolens.runs
                SET status = %s,
                    final_metrics = %s,
                    error_message = %s,
                    finished_at = NOW()
                WHERE id = %s
                """,
            (
                status,
                Json(final_metrics) if final_metrics else None,
                error_message,
                run_id,
            ),
        )


def insert_prediction(
    run_id: int,
    image_path: str,
    image_filename: str,
    true_label: str,
    predicted_label: str,
    probs: dict[str, float],
    confidence: float,
    inference_time_ms: float | None = None,
) -> int:
    """Insert one prediction row; return its id (used by XAI artifacts).

    is_correct is auto-computed by a GENERATED column — do not pass it.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                INSERT INTO neurolens.predictions (
                    run_id, image_path, image_filename,
                    true_label, predicted_label,
                    probs, confidence, inference_time_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
            (
                run_id,
                image_path,
                image_filename,
                true_label,
                predicted_label,
                Json(probs),
                confidence,
                inference_time_ms,
            ),
        )
        row = cur.fetchone()
        assert row is not None, "INSERT ... RETURNING returned no row"
        return int(row[0])


def insert_predictions_bulk(
    run_id: int,
    rows: list[dict[str, Any]],
    page_size: int = 500,
) -> int:
    """Insert many prediction rows in a single batched INSERT.

    Designed to replace tight loops of ``insert_prediction`` when persisting
    full test-set evaluations (1600 images per fold). Opens ONE connection
    and ONE SSL handshake instead of N per insert — typically 30+ minutes
    faster on remote PostgreSQL.

    Args:
        run_id: parent run identifier.
        rows: list of dicts with keys:
            ``image_path``, ``image_filename``, ``true_label``,
            ``predicted_label``, ``probs`` (dict[str, float]),
            ``confidence``, and optional ``inference_time_ms``.
        page_size: how many rows per server round-trip inside the batch.

    Returns:
        Number of rows inserted (equals ``len(rows)``).
    """
    if not rows:
        return 0

    values = [
        (
            run_id,
            r["image_path"],
            r["image_filename"],
            r["true_label"],
            r["predicted_label"],
            Json(r["probs"]),
            r["confidence"],
            r.get("inference_time_ms"),
        )
        for r in rows
    ]

    with get_connection() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO neurolens.predictions (
                run_id, image_path, image_filename,
                true_label, predicted_label,
                probs, confidence, inference_time_ms
            )
            VALUES %s
            """,
            values,
            page_size=page_size,
        )
    return len(rows)
