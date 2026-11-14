"""Measurement and signal-prior losses from equations (5)-(7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def total_variation_1d(signals: torch.Tensor) -> torch.Tensor:
    if signals.shape[-1] < 2:
        raise ValueError("1D total variation needs at least two samples.")
    return (signals[..., 1:] - signals[..., :-1]).square().mean()


def total_variation_2d(images: torch.Tensor) -> torch.Tensor:
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("2D total variation needs at least a 2x2 image.")
    vertical = (images[..., 1:, :] - images[..., :-1, :]).square().mean()
    horizontal = (images[..., :, 1:] - images[..., :, :-1]).square().mean()
    return vertical + horizontal


def local_variation_2d(
    images: torch.Tensor, kernel_size: int = 3, stride: int = 2
) -> torch.Tensor:
    """Mean standard deviation of overlapping image patches."""
    if images.ndim == 2:
        images = images.unsqueeze(0)
    if images.ndim != 3:
        raise ValueError("images must have shape [H, W] or [batch, H, W].")
    if images.shape[-2] < kernel_size or images.shape[-1] < kernel_size:
        raise ValueError("Image is smaller than the local-variation window.")

    patches = F.unfold(
        images.unsqueeze(1), kernel_size=kernel_size, stride=stride
    )
    return patches.std(dim=1, unbiased=True).mean()


def cl_objective(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    x_pred: torch.Tensor,
    tv_weight: float,
    lv_weight: float = 0.0,
    image_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total loss and measurement MSE.

    For images, x_pred contains one signal per image column [width, height].
    """
    measurement_loss = F.mse_loss(y_pred, y)
    if image_shape is None:
        if lv_weight:
            raise ValueError("LV is only defined when image_shape is provided.")
        prior = total_variation_1d(x_pred)
        loss = measurement_loss + tv_weight * prior
    else:
        height, width = image_shape
        if tuple(x_pred.shape) != (width, height):
            raise ValueError(
                f"Expected x_pred shape {(width, height)}, got {tuple(x_pred.shape)}."
            )
        image = x_pred.T.reshape(height, width)
        loss = measurement_loss + tv_weight * total_variation_2d(image)
        if lv_weight:
            loss = loss + lv_weight * local_variation_2d(image)
    return loss, measurement_loss
