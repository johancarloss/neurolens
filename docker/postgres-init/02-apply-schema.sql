-- ============================================================================
-- NeuroLens — PostgreSQL Schema
-- ----------------------------------------------------------------------------
-- Applied to database: neurolens
-- Owner role:          neurolens_writer
-- Reference doc:       docs/private/blueprint/02-data-model.md
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS neurolens;
SET search_path TO neurolens, public;

-- ============================================================================
-- experiments: groups related runs under a common purpose
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE experiments IS
    'Groups related runs under a shared purpose (e.g., "vgg16-vs-resnet50")';

-- ============================================================================
-- runs: a single training execution (one architecture + one config + one fold)
-- ============================================================================
CREATE TABLE IF NOT EXISTS runs (
    id                  SERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    arch                TEXT NOT NULL,            -- 'vgg16' | 'resnet50' | ...
    stage               SMALLINT NOT NULL,        -- 1 (head only) | 2 (fine-tune)
    fold                SMALLINT,                 -- 0..4 if CV, NULL otherwise
    status              TEXT NOT NULL DEFAULT 'running',
    wandb_run_id        TEXT,                     -- cross-reference to W&B
    wandb_url           TEXT,                     -- direct link to W&B UI
    git_commit          TEXT,                     -- SHA of the commit that ran
    kaggle_kernel_url   TEXT,
    config_yaml         TEXT,                     -- raw YAML used (for reprod)
    hyperparams         JSONB NOT NULL,           -- {lr, batch_size, ...}
    final_metrics       JSONB,                    -- {test_acc, f1_macro, ...}
    error_message       TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,

    CONSTRAINT valid_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_stage CHECK (stage IN (0, 1, 2)),  -- 0 reserved for non-training events (hello-world)
    CONSTRAINT valid_fold CHECK (fold IS NULL OR (fold >= 0 AND fold <= 4))
);

COMMENT ON TABLE runs IS
    'Each row = one training execution. status transitions: running -> completed/failed';

-- ============================================================================
-- metrics: time-series per epoch (append-only)
-- ============================================================================
CREATE TABLE IF NOT EXISTS metrics (
    id          BIGSERIAL PRIMARY KEY,
    run_id      INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    epoch       INT NOT NULL,
    phase       TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    logged_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_phase CHECK (phase IN ('train', 'val', 'test'))
);

COMMENT ON TABLE metrics IS
    'Time-series: one row per (run, epoch, phase, metric_name)';

-- ============================================================================
-- predictions: one row per test-set image per run
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    run_id            INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    image_path        TEXT NOT NULL,
    image_filename    TEXT NOT NULL,
    true_label        TEXT NOT NULL,
    predicted_label   TEXT NOT NULL,
    probs             JSONB NOT NULL,
    confidence        DOUBLE PRECISION NOT NULL,
    is_correct        BOOLEAN GENERATED ALWAYS AS (true_label = predicted_label) STORED,
    inference_time_ms DOUBLE PRECISION,

    CONSTRAINT valid_true_label
        CHECK (true_label IN ('glioma', 'meningioma', 'pituitary', 'notumor')),
    CONSTRAINT valid_predicted_label
        CHECK (predicted_label IN ('glioma', 'meningioma', 'pituitary', 'notumor'))
);

COMMENT ON TABLE predictions IS
    'One row per (run, test image). is_correct auto-computed via GENERATED column';

-- ============================================================================
-- xai_artifacts: paths to XAI heatmaps (NOT BLOBs)
-- ============================================================================
CREATE TABLE IF NOT EXISTS xai_artifacts (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    method          TEXT NOT NULL,
    target_class    TEXT NOT NULL,
    artifact_path   TEXT NOT NULL,            -- path on filesystem (NOT blob)
    raw_array_path  TEXT,                     -- optional .npy with raw heatmap
    compute_time_ms DOUBLE PRECISION NOT NULL,
    metadata        JSONB,                    -- {num_samples, target_layer, ...}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_method CHECK (method IN ('gradcam', 'lime', 'shap'))
);

COMMENT ON TABLE xai_artifacts IS
    'Per-prediction XAI artifacts. Banco guarda só caminho; PNG fica em filesystem';

-- ============================================================================
-- xai_comparisons: aggregated metrics across XAI techniques per prediction
-- ============================================================================
CREATE TABLE IF NOT EXISTS xai_comparisons (
    id                      BIGSERIAL PRIMARY KEY,
    prediction_id           BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    -- Pairwise IoU on binarized maps
    iou_gradcam_lime        DOUBLE PRECISION,
    iou_gradcam_shap        DOUBLE PRECISION,
    iou_lime_shap           DOUBLE PRECISION,
    -- LIME stability (std across N runs)
    lime_stability_std      DOUBLE PRECISION,
    lime_num_runs           SMALLINT,
    -- Sparsity (fraction of pixels above threshold)
    sparsity_gradcam        DOUBLE PRECISION,
    sparsity_lime           DOUBLE PRECISION,
    sparsity_shap           DOUBLE PRECISION,
    -- Wall-clock time per technique
    time_ms_gradcam         DOUBLE PRECISION,
    time_ms_lime            DOUBLE PRECISION,
    time_ms_shap            DOUBLE PRECISION,
    -- Metadata
    binarization_threshold  DOUBLE PRECISION DEFAULT 0.5,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE xai_comparisons IS
    '5 quantitative XAI metrics per prediction (concordance, stability, sparsity, time)';

-- ============================================================================
-- Indexes
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_arch_stage ON runs(arch, stage);
CREATE INDEX IF NOT EXISTS idx_runs_status_finished ON runs(status, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_hyperparams_gin
    ON runs USING GIN (hyperparams jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_runs_final_metrics_gin
    ON runs USING GIN (final_metrics jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_metrics_run_epoch ON metrics(run_id, epoch);
CREATE INDEX IF NOT EXISTS idx_metrics_run_phase_name
    ON metrics(run_id, phase, metric_name);

CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_correct ON predictions(run_id, is_correct);
CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(true_label, predicted_label);

CREATE INDEX IF NOT EXISTS idx_xai_prediction_method
    ON xai_artifacts(prediction_id, method);

CREATE INDEX IF NOT EXISTS idx_xai_comparisons_prediction
    ON xai_comparisons(prediction_id);
