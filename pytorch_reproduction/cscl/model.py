"""Ultra-small CLOMP model: only the sparse coefficients are trainable."""

from __future__ import annotations

import torch
from torch import nn


class CLOMP(nn.Module):
    """Coefficients Learning on Orthogonal Matching Pursuit.

    A, D and atom_norms are fixed buffers. The parameter count is exactly
    batch_size * n, or n for a single signal.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        dct_basis: torch.Tensor,
        atom_norms: torch.Tensor,
        batch_size: int,
    ) -> None:
        super().__init__()
        if dictionary.ndim != 2:
            raise ValueError("dictionary must have shape [m, n].")
        self.m, self.n = dictionary.shape
        self.batch_size = batch_size
        self.register_buffer("dictionary", dictionary.clone())
        self.register_buffer("dct_basis", dct_basis.clone())
        self.register_buffer("atom_norms", atom_norms.clone())
        self.coefficients = nn.Parameter(
            torch.empty((batch_size, self.n), dtype=dictionary.dtype)
        )
        nn.init.xavier_uniform_(self.coefficients)

    @property
    def trainable_parameter_count(self) -> int:
        return self.coefficients.numel()

    def forward(
        self, residual: torch.Tensor, support: torch.Tensor, threshold: float = 0.7
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject residual atoms and estimate measurements with masked s."""
        correlation = (residual @ self.dictionary).abs()
        normalizer = correlation.amax(dim=1, keepdim=True).clamp_min(1e-12)
        selected = correlation / normalizer > threshold
        support = support | selected
        sparse_coefficients = self.coefficients * support.to(self.coefficients.dtype)
        y_pred = sparse_coefficients @ self.dictionary.T
        return support, y_pred

    def sparse_coefficients(self, support: torch.Tensor) -> torch.Tensor:
        return self.coefficients * support.to(self.coefficients.dtype)

    def reconstruct(self, support: torch.Tensor) -> torch.Tensor:
        # A_normalized = A_original / atom_norm. Undo that scaling before Ds.
        physical_coefficients = self.sparse_coefficients(support) / self.atom_norms
        return physical_coefficients @ self.dct_basis.T

    def reconstruct_unmasked(self) -> torch.Tensor:
        """Dense reconstruction from all coefficients (released-code behavior).

        The authors' script applies the prior loss and produces the final
        output from the coefficients BEFORE the support mask; the mask only
        gates the measurement-loss branch. This method reproduces that.
        """
        physical_coefficients = self.coefficients / self.atom_norms
        return physical_coefficients @ self.dct_basis.T
