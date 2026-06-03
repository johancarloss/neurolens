"""Backwards-compat shim — the real logic lives in ``run_train``.

The Kaggle runner kernel is pushed exactly once (re-pushing detaches its
attached Secrets and Dataset). Its baked-in ``JOB_TYPES`` registry imports
``neurolens.training.run_vgg16`` and calls ``main(config_profile=...)``.

Phase 2 generalized the runner into the arch-agnostic ``run_train`` (the
architecture now comes from ``config.arch``). This module re-exports it so
the immutable kernel keeps working unchanged while the codebase stays honest
about what the runner actually does.

When the kernel is ever re-pushed, point ``JOB_TYPES`` at ``run_train``
directly and delete this file.
"""

from __future__ import annotations

from neurolens.training.run_train import (
    main,
    run_one_fold,
)

__all__ = ["main", "run_one_fold"]
