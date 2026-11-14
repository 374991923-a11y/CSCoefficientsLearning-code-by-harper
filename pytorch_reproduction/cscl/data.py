"""Synthetic data generation matching the released CLOMP1D experiment."""

from __future__ import annotations

import numpy as np
from scipy.fft import idct


def generate_synthetic_1d(
    n: int,
    num_signals: int,
    sparsity_rate: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized signals and their initializing DCT coefficients.

    The coefficient construction follows the authors' released script: dominant
    low-frequency coefficients, a small number of fixed low-frequency terms,
    and weak high-frequency noise. Min-max normalization is performed per
    signal, exactly as in that script. Consequently, the returned coefficients
    describe the pre-normalization signals and are not reconstruction targets.
    """
    if n < 16:
        raise ValueError("n must be at least 16 for the synthetic generator.")
    if num_signals < 1:
        raise ValueError("num_signals must be positive.")
    if not 0 < sparsity_rate <= 1:
        raise ValueError("sparsity_rate must be in (0, 1].")

    k = max(1, int(sparsity_rate * n))
    low_cutoff = min(n, max(k, int(0.9 * n)))
    rng = np.random.default_rng(seed)

    values = 0.5 ** np.linspace(-10, 5, k) + 25
    values /= values.max()
    coefficients = np.zeros((num_signals, n), dtype=np.float64)

    low_indices = np.arange(low_cutoff)
    low_band = np.arange(max(4, n // 10))
    negative_count = k // 5
    fixed_values = 5 * np.array([0.8, 0.75, 0.6, 0.5])

    for batch_index in range(num_signals):
        selected = rng.permutation(low_indices)[:k]
        signed_values = values.copy()
        if negative_count:
            negative = rng.permutation(k)[:negative_count]
            signed_values[negative] *= -1
        coefficients[batch_index, selected] = signed_values
        coefficients[batch_index, 0] = 5

        fixed_positions = rng.permutation(low_band)[:4]
        coefficients[batch_index, fixed_positions] = fixed_values
        if low_cutoff < n:
            coefficients[batch_index, low_cutoff:] = (
                0.01 * rng.random(n - low_cutoff)
            )

    signals = idct(coefficients, axis=-1)
    minima = signals.min(axis=1, keepdims=True)
    ranges = (signals.max(axis=1, keepdims=True) - minima).clip(min=1e-12)
    signals = (signals - minima) / ranges
    return signals.astype(np.float32), coefficients.astype(np.float32)


def generate_synthetic_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Create a smooth test image for dependency-free 2D smoke experiments."""
    if height < 3 or width < 3:
        raise ValueError("Both image dimensions must be at least 3.")
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    y /= max(height - 1, 1)
    x /= max(width - 1, 1)
    image = (
        0.45
        + 0.16 * np.sin(2 * np.pi * x)
        + 0.12 * np.cos(2 * np.pi * y)
        + 0.25 * np.exp(-((x - 0.32) ** 2 + (y - 0.38) ** 2) / 0.025)
        - 0.20 * np.exp(-((x - 0.72) ** 2 + (y - 0.68) ** 2) / 0.015)
    )
    image = (image - image.min()) / np.ptp(image).clip(min=1e-12)
    return image.astype(np.float32)
