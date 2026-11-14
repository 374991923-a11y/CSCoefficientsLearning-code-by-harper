"""Devices, random seeds, and compressed-sensing operators."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.fft import dct


@dataclass
class CSProblem:
    """Tensors defining y = Mx and the normalized dictionary A = MD."""

    y: torch.Tensor
    measurement: torch.Tensor
    dct_basis: torch.Tensor
    dictionary: torch.Tensor
    atom_norms: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def make_dct_basis(n: int) -> torch.Tensor:
    """Return the orthonormal DCT-II synthesis basis used by the paper."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    identity = np.eye(n, dtype=np.float64)
    basis = dct(identity, axis=0, norm="ortho").T
    return torch.from_numpy(basis).to(torch.float32)


def make_gaussian_measurement(
    n: int, sampling_rate: float, seed: int
) -> torch.Tensor:
    """Create a reproducible Gaussian sensing matrix with unit-norm columns."""
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1].")
    m = max(1, int(n * sampling_rate))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((m, n), generator=generator, dtype=torch.float64)
    matrix /= matrix.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return matrix.to(torch.float32)


def build_cs_problem(
    x: torch.Tensor, measurement: torch.Tensor, dct_basis: torch.Tensor
) -> CSProblem:
    """Measure batch signals and normalize dictionary atoms.

    x has shape [batch, n]. The atom norms are retained so learned normalized
    coefficients can be converted back before applying the inverse DCT.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape [batch, n], got {tuple(x.shape)}.")
    if measurement.ndim != 2 or measurement.shape[1] != x.shape[1]:
        raise ValueError("measurement must have shape [m, n].")
    if dct_basis.shape != (x.shape[1], x.shape[1]):
        raise ValueError("dct_basis must have shape [n, n].")

    y = x @ measurement.T
    unnormalized_dictionary = measurement @ dct_basis
    atom_norms = unnormalized_dictionary.norm(dim=0).clamp_min(1e-12)
    dictionary = unnormalized_dictionary / atom_norms
    return CSProblem(
        y=y,
        measurement=measurement,
        dct_basis=dct_basis,
        dictionary=dictionary,
        atom_norms=atom_norms,
    )
