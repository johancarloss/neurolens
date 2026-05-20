"""Minimal smoke tests for CI.

These tests verify that the package structure is sound and the
runtime environment matches expectations. They do not exercise
any business logic — that comes in dedicated test files.
"""

from __future__ import annotations

import sys


def test_python_version() -> None:
    """Confirm we are running on Python 3.12 as pinned in pyproject.toml."""
    major, minor = sys.version_info[:2]
    assert (major, minor) == (3, 12), (
        f"Expected Python 3.12, got {major}.{minor}. "
        "Update pyproject.toml requires-python if upgrading intentionally."
    )


def test_neurolens_package_imports() -> None:
    """All neurolens subpackages import without error."""
    import neurolens  # noqa: F401
    from neurolens import (  # noqa: F401
        data,
        db,
        models,
        tracking,
        training,
        ui,
        xai,
    )


def test_no_accidental_torch_cuda_dependency_in_tests() -> None:
    """Tests must not require GPU — CI runs on CPU-only ubuntu runners.

    This test imports torch (CPU is fine) and confirms we can build a tensor
    without CUDA. If this fails on CI, it indicates a test added a CUDA-only
    code path.
    """
    import torch

    x = torch.zeros(2, 3)
    assert x.shape == (2, 3)
    assert str(x.device) == "cpu"
