"""Frozen DINOv2 image encoder.

Loaded via ``torch.hub`` from facebookresearch/dinov2. The encoder is set to
eval mode and all parameters are frozen — the encoder is policy-independent
by construction (see spec §0.3.1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import interpolate

from .config import LocalThresholdConfig

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class FrozenImageEncoder:
    def __init__(self, cfg: LocalThresholdConfig, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(device or cfg.encoder_device)
        if cfg.encoder_name not in _FEATURE_DIMS:
            raise ValueError(f"Unknown encoder_name={cfg.encoder_name!r}. Known: {sorted(_FEATURE_DIMS)}")
        self._feature_dim = _FEATURE_DIMS[cfg.encoder_name]

        model = torch.hub.load("facebookresearch/dinov2", cfg.encoder_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)  # noqa: FBT003 — torch API takes a positional bool
        self.model = model.to(self.device)

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # Accept (B, H, W, 3) uint8/float or (B, 3, H, W) float in [0,1].
        if images.ndim != 4:
            raise ValueError(f"images must be 4D, got shape {tuple(images.shape)}")
        if images.shape[-1] == 3 and images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        size = self.cfg.image_size
        if images.shape[-2] != size or images.shape[-1] != size:
            images = interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        return (images - self._mean) / self._std

    @torch.no_grad()
    @torch.inference_mode()
    def encode(self, images) -> torch.Tensor:
        """Encode a batch of images to features (B, D).

        Accepts numpy arrays or torch tensors. Returns float32 on CPU.
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        x = self._preprocess(images)

        # DINOv2 from torch.hub provides forward_features() returning a dict with
        # "x_norm_clstoken" (B, D) and "x_norm_patchtokens" (B, N, D).
        out = self.model.forward_features(x)
        if self.cfg.encoder_pool == "cls":
            feat = out["x_norm_clstoken"]
        elif self.cfg.encoder_pool == "mean_patch":
            feat = out["x_norm_patchtokens"].mean(dim=1)
        else:
            raise ValueError(f"Unknown encoder_pool={self.cfg.encoder_pool!r}")
        return feat.to(dtype=torch.float32, device="cpu")
