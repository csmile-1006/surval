"""Tests for FrozenImageEncoder. Marked GPU-only."""

import numpy as np
import pytest
import torch

from surval.local_threshold.config import LocalThresholdConfig
from surval.local_threshold.encoder import FrozenImageEncoder


@pytest.fixture(scope="module")
def encoder():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = LocalThresholdConfig(encoder_name="dinov2_vits14")  # smallest variant for speed
    return FrozenImageEncoder(cfg, device="cuda")


@pytest.mark.gpu
def test_determinism(encoder):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)
    f1 = encoder.encode(img).numpy()
    f2 = encoder.encode(img).numpy()
    assert f1.shape == (2, encoder.feature_dim)
    assert f1.dtype == np.float32
    assert np.max(np.abs(f1 - f2)) < 1e-5, f"max abs diff = {np.max(np.abs(f1 - f2))}"


@pytest.mark.gpu
def test_resize_path(encoder):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(1, 180, 320, 3), dtype=np.uint8)
    f = encoder.encode(img).numpy()
    assert f.shape == (1, encoder.feature_dim)
