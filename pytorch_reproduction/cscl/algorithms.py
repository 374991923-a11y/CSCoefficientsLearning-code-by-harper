"""CLOMP and the OMP/IHT reference algorithms."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .losses import cl_objective
from .model import CLOMP
from .utils import synchronize


@dataclass
class CLOMPConfig:
    iterations: int = 200
    threshold: float = 0.7
    learning_rate: float = 2.0
    min_learning_rate: float = 0.01
    scheduler_steps: int = 100
    tv_weight: float = 0.5
    lv_weight: float = 0.0
    stop_threshold: float = 1e-5
    min_iterations: int = 82
    plateau_delta: float | None = None
    stop_on_total_loss: bool = False
    log_every: int = 0
    # False: paper equations (4)/(Fig. 2b) - prior and output use Mask*s.
    # True: released-code behavior - prior and output use unmasked s; the
    # mask only gates the measurement branch (dense TV-regularized output).
    unmasked_prior: bool = False

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("Learning rates must be positive/non-negative.")
        if self.scheduler_steps < 1:
            raise ValueError("scheduler_steps must be positive.")
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be positive.")


@dataclass
class ReconstructionResult:
    signal: torch.Tensor
    coefficients: torch.Tensor
    support: torch.Tensor
    losses: list[float]
    residual_losses: list[float]
    iterations: int
    elapsed_seconds: float


def reconstruct_clomp(
    problem,
    config: CLOMPConfig,
    image_shape: tuple[int, int] | None = None,
) -> ReconstructionResult:
    """Optimize coefficients independently for the current measurements only.

    No dataset, pretrained weights, or gradient update shared across experiments
    is involved. ``problem`` is a :class:`cscl.utils.CSProblem` instance.
    """
    config.validate()
    y = problem.y
    batch_size, n = y.shape[0], problem.dictionary.shape[1]
    model = CLOMP(
        problem.dictionary,
        problem.dct_basis,
        problem.atom_norms,
        batch_size,
    ).to(y.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=config.scheduler_steps, power=1.0
    )
    support = torch.zeros(
        (batch_size, n), dtype=torch.bool, device=y.device
    )
    residual = y
    losses: list[float] = []
    residual_losses: list[float] = []

    synchronize(y.device)
    started = time.perf_counter()
    iterations_run = 0
    for epoch in range(config.iterations):
        support, y_pred = model(residual, support, config.threshold)
        if config.unmasked_prior:
            x_pred = model.reconstruct_unmasked()
        else:
            x_pred = model.reconstruct(support)
        loss, measurement_loss = cl_objective(
            y,
            y_pred,
            x_pred,
            tv_weight=config.tv_weight,
            lv_weight=config.lv_weight,
            image_shape=image_shape,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        for group in optimizer.param_groups:
            group["lr"] = max(config.min_learning_rate, group["lr"])

        loss_value = loss.detach().item()
        residual_value = measurement_loss.detach().item()
        losses.append(loss_value)
        residual_losses.append(residual_value)
        residual = (y - y_pred).detach()
        iterations_run = epoch + 1

        if config.log_every and (epoch % config.log_every == 0):
            print(
                f"CLOMP iteration {epoch + 1:4d}/{config.iterations}: "
                f"loss={loss_value:.6e}, mse={residual_value:.6e}, "
                f"support={support.sum().item()}"
            )

        if iterations_run >= config.min_iterations:
            stopping_loss = loss_value if config.stop_on_total_loss else residual_value
            reached_threshold = stopping_loss < config.stop_threshold
            reached_plateau = (
                config.plateau_delta is not None
                and len(losses) > 1
                and abs(losses[-2] - losses[-1]) < config.plateau_delta
            )
            if reached_threshold or reached_plateau:
                break

    with torch.no_grad():
        raw_coefficients = (
            model.coefficients
            if config.unmasked_prior
            else model.sparse_coefficients(support)
        )
        coefficients = raw_coefficients / model.atom_norms
        signal = coefficients @ problem.dct_basis.T
    synchronize(y.device)
    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        signal=signal.detach(),
        coefficients=coefficients.detach(),
        support=support.detach(),
        losses=losses,
        residual_losses=residual_losses,
        iterations=iterations_run,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def reconstruct_omp(
    problem,
    threshold: float = 0.7,
    iterations: int = 200,
    stop_threshold: float = 1e-5,
) -> ReconstructionResult:
    """Multi-index OMP baseline used in the authors' released scripts."""
    y, dictionary = problem.y, problem.dictionary
    batch_size, n = y.shape[0], dictionary.shape[1]
    residual = y.clone()
    support = torch.zeros((batch_size, n), dtype=torch.bool, device=y.device)
    coefficients = torch.zeros((batch_size, n), dtype=y.dtype, device=y.device)
    losses: list[float] = []

    synchronize(y.device)
    started = time.perf_counter()
    iterations_run = 0
    for iteration in range(iterations):
        correlation = (residual @ dictionary).abs()
        normalizer = correlation.amax(dim=1, keepdim=True).clamp_min(1e-12)
        support |= correlation / normalizer > threshold

        for batch_index in range(batch_size):
            selected = support[batch_index]
            if not selected.any():
                continue
            atoms = dictionary[:, selected]
            weights = torch.linalg.pinv(atoms) @ y[batch_index]
            coefficients[batch_index, selected] = weights
            residual[batch_index] = y[batch_index] - atoms @ weights

        loss_value = residual.square().mean().item()
        losses.append(loss_value)
        iterations_run = iteration + 1
        if loss_value < stop_threshold:
            break

    coefficients = coefficients / problem.atom_norms
    signal = coefficients @ problem.dct_basis.T
    synchronize(y.device)
    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        signal=signal,
        coefficients=coefficients,
        support=support,
        losses=losses.copy(),
        residual_losses=losses,
        iterations=iterations_run,
        elapsed_seconds=elapsed,
    )


@torch.no_grad()
def reconstruct_iht(
    problem,
    iterations: int = 200,
    stop_threshold: float = 1e-5,
    step_size: float = 0.1,
    sparsity: int | None = None,
) -> ReconstructionResult:
    """Batched iterative hard thresholding; default K=m/5 as in the paper code."""
    y, dictionary = problem.y, problem.dictionary
    batch_size, n = y.shape[0], dictionary.shape[1]
    m = y.shape[1]
    k = max(1, m // 5) if sparsity is None else sparsity
    if not 1 <= k <= n:
        raise ValueError(f"sparsity must be in [1, {n}].")

    residual = y.clone()
    coefficients = torch.zeros((batch_size, n), dtype=y.dtype, device=y.device)
    support = torch.zeros((batch_size, n), dtype=torch.bool, device=y.device)
    losses: list[float] = []

    synchronize(y.device)
    started = time.perf_counter()
    iterations_run = 0
    for iteration in range(iterations):
        coefficients = coefficients + step_size * (residual @ dictionary)
        indices = coefficients.abs().topk(k, dim=1, largest=True, sorted=False).indices
        support = torch.zeros_like(support).scatter_(1, indices, True)
        coefficients = coefficients * support.to(coefficients.dtype)
        residual = y - coefficients @ dictionary.T

        loss_value = residual.square().mean().item()
        losses.append(loss_value)
        iterations_run = iteration + 1
        if loss_value < stop_threshold:
            break

    coefficients = coefficients / problem.atom_norms
    signal = coefficients @ problem.dct_basis.T
    synchronize(y.device)
    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        signal=signal,
        coefficients=coefficients,
        support=support,
        losses=losses.copy(),
        residual_losses=losses,
        iterations=iterations_run,
        elapsed_seconds=elapsed,
    )
