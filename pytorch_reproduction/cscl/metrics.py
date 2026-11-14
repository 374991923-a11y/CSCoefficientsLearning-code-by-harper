"""NumPy evaluation metrics; no optional computer-vision dependency is needed."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, uniform_filter


def _as_batch(signals: np.ndarray) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[None, :]
    if signals.ndim != 2:
        raise ValueError("Expected one signal or a [batch, n] array.")
    return signals


def mse_per_signal(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference, estimate = _as_batch(reference), _as_batch(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("Reference and estimate shapes must match.")
    return ((reference - estimate) ** 2).mean(axis=1)


def psnr_from_mse(mse: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    mse = np.asarray(mse, dtype=np.float64)
    ratio = np.full_like(mse, np.inf)
    np.divide(data_range, np.sqrt(mse), out=ratio, where=mse > 0)
    return 20 * np.log10(ratio)


def ssim_1d(
    reference: np.ndarray,
    estimate: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
) -> np.ndarray:
    """The per-signal Gaussian SSIM variant in the released CLOMP1D code."""
    reference, estimate = _as_batch(reference), _as_batch(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("Reference and estimate shapes must match.")
    if reference.shape[1] < window_size:
        raise ValueError("Signals must be at least as long as the SSIM window.")

    coordinates = np.arange(window_size) - window_size // 2
    kernel = np.exp(-(coordinates**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    mu1 = convolve1d(reference, kernel, axis=1, mode="reflect")
    mu2 = convolve1d(estimate, kernel, axis=1, mode="reflect")
    sigma1 = convolve1d(reference**2, kernel, axis=1, mode="reflect") - mu1**2
    sigma2 = convolve1d(estimate**2, kernel, axis=1, mode="reflect") - mu2**2
    sigma12 = convolve1d(reference * estimate, kernel, axis=1, mode="reflect") - mu1 * mu2
    dynamic_range = np.ptp(reference, axis=1, keepdims=True)
    c1 = (0.06 * dynamic_range) ** 2
    c2 = (0.10 * dynamic_range) ** 2
    score = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    )
    return score.mean(axis=1)


def pcc_per_signal(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference, estimate = _as_batch(reference), _as_batch(estimate)
    ref_centered = reference - reference.mean(axis=1, keepdims=True)
    est_centered = estimate - estimate.mean(axis=1, keepdims=True)
    numerator = (ref_centered * est_centered).sum(axis=1)
    denominator = np.linalg.norm(ref_centered, axis=1) * np.linalg.norm(
        est_centered, axis=1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def evaluate_1d(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    mse = mse_per_signal(reference, estimate)
    return {
        "ssim": float(np.nanmean(ssim_1d(reference, estimate))),
        "psnr": float(np.nanmean(psnr_from_mse(mse))),
        "mse": float(np.nanmean(mse)),
        "pcc": float(np.nanmean(pcc_per_signal(reference, estimate))),
    }


def ssim_2d(
    reference: np.ndarray,
    estimate: np.ndarray,
    data_range: float = 1.0,
    window_size: int = 7,
) -> float:
    """Standard sliding-window SSIM, matching skimage's default 7x7 setup."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != estimate.shape:
        raise ValueError("Expected two equally shaped 2D images.")
    if min(reference.shape) < window_size:
        raise ValueError("Images must be at least 7x7 for SSIM.")

    filter_args = {"size": window_size, "mode": "reflect"}
    mu1 = uniform_filter(reference, **filter_args)
    mu2 = uniform_filter(estimate, **filter_args)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    covariance_normalization = window_size**2 / (window_size**2 - 1)
    sigma1 = covariance_normalization * (
        uniform_filter(reference**2, **filter_args) - mu1_sq
    )
    sigma2 = covariance_normalization * (
        uniform_filter(estimate**2, **filter_args) - mu2_sq
    )
    sigma12 = covariance_normalization * (
        uniform_filter(reference * estimate, **filter_args) - mu12
    )
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    crop = window_size // 2
    return float(score[crop:-crop, crop:-crop].mean())


def evaluate_2d(
    reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    mse = float(np.mean((reference - estimate) ** 2))
    return {
        "ssim": ssim_2d(reference, estimate, data_range=data_range),
        "psnr": float(psnr_from_mse(mse, data_range=data_range)),
        "mse": mse,
    }
