from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscl.algorithms import CLOMPConfig, reconstruct_clomp, reconstruct_iht, reconstruct_omp
from cscl.data import generate_synthetic_1d
from cscl.losses import local_variation_2d, total_variation_1d, total_variation_2d
from cscl.metrics import evaluate_1d, evaluate_2d
from cscl.model import CLOMP
from cscl.utils import (
    CSProblem,
    build_cs_problem,
    make_dct_basis,
    make_gaussian_measurement,
)


class CoreTests(unittest.TestCase):
    def test_dct_basis_is_orthonormal(self):
        basis = make_dct_basis(32)
        error = (basis.T @ basis - torch.eye(32)).abs().max()
        self.assertLess(error.item(), 1e-6)

    def test_synthetic_data_is_deterministic_and_normalized(self):
        first, _ = generate_synthetic_1d(64, 3, 0.2, seed=7)
        second, _ = generate_synthetic_1d(64, 3, 0.2, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 64))
        np.testing.assert_allclose(first.min(axis=1), 0.0, atol=1e-7)
        np.testing.assert_allclose(first.max(axis=1), 1.0, atol=1e-7)

    def test_normalized_dictionary_round_trip(self):
        n = 16
        basis = make_dct_basis(n)
        measurement = make_gaussian_measurement(n, 0.5, seed=3)
        physical = torch.zeros((2, n))
        physical[0, 0], physical[0, 3] = 1.0, -0.4
        physical[1, 1], physical[1, 5] = 0.7, 0.2
        x = physical @ basis.T
        problem = build_cs_problem(x, measurement, basis)
        normalized = physical * problem.atom_norms
        torch.testing.assert_close(normalized @ problem.dictionary.T, problem.y)
        torch.testing.assert_close(
            (normalized / problem.atom_norms) @ basis.T, x, rtol=1e-5, atol=1e-6
        )

    def test_model_has_only_n_parameters_per_signal_and_support_grows(self):
        basis = make_dct_basis(16)
        measurement = make_gaussian_measurement(16, 0.5, seed=1)
        x = torch.rand((2, 16))
        problem = build_cs_problem(x, measurement, basis)
        model = CLOMP(problem.dictionary, basis, problem.atom_norms, batch_size=2)
        self.assertEqual(model.trainable_parameter_count, 32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 32)
        support = torch.zeros((2, 16), dtype=torch.bool)
        first, _ = model(problem.y, support)
        second, _ = model(problem.y, first)
        self.assertTrue(torch.all(second | ~first).item())
        self.assertTrue(model.dictionary.requires_grad is False)

    def test_clomp_optimization_is_finite_and_reduces_loss(self):
        signals, _ = generate_synthetic_1d(32, 2, 0.2, seed=11)
        basis = make_dct_basis(32)
        measurement = make_gaussian_measurement(32, 0.5, seed=12)
        problem = build_cs_problem(torch.from_numpy(signals), measurement, basis)
        result = reconstruct_clomp(
            problem,
            CLOMPConfig(
                iterations=25,
                learning_rate=0.05,
                min_learning_rate=0.005,
                scheduler_steps=25,
                tv_weight=0.1,
                min_iterations=100,
            ),
        )
        self.assertEqual(result.signal.shape, signals.shape)
        self.assertTrue(np.isfinite(result.losses).all())
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertEqual(result.iterations, 25)

    def test_omp_recovers_identity_problem(self):
        identity = torch.eye(8)
        coefficients = torch.zeros((1, 8))
        coefficients[0, 1], coefficients[0, 6] = 1.0, -0.5
        problem = CSProblem(coefficients, identity, identity, identity, torch.ones(8))
        result = reconstruct_omp(problem, iterations=2, stop_threshold=1e-12)
        torch.testing.assert_close(result.signal, coefficients, rtol=1e-5, atol=1e-6)

    def test_iht_recovers_single_identity_coefficient(self):
        identity = torch.eye(8)
        coefficients = torch.zeros((1, 8))
        coefficients[0, 4] = 0.8
        problem = CSProblem(coefficients, identity, identity, identity, torch.ones(8))
        result = reconstruct_iht(
            problem, iterations=3, stop_threshold=1e-12, step_size=1.0, sparsity=1
        )
        torch.testing.assert_close(result.signal, coefficients, rtol=1e-5, atol=1e-6)

    def test_prior_losses_are_differentiable(self):
        signal = torch.randn((2, 16), requires_grad=True)
        image = torch.rand((8, 10), requires_grad=True)
        loss = (
            total_variation_1d(signal)
            + total_variation_2d(image)
            + local_variation_2d(image)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(signal.grad)
        self.assertIsNotNone(image.grad)

    def test_metrics_perfect_reconstruction(self):
        signal = np.sin(np.linspace(0, 4, 32))[None]
        metrics_1d = evaluate_1d(signal, signal)
        self.assertAlmostEqual(metrics_1d["ssim"], 1.0, places=6)
        self.assertAlmostEqual(metrics_1d["pcc"], 1.0, places=6)
        self.assertTrue(np.isinf(metrics_1d["psnr"]))
        image = np.random.default_rng(4).random((16, 16))
        metrics_2d = evaluate_2d(image, image)
        self.assertAlmostEqual(metrics_2d["ssim"], 1.0, places=6)
        self.assertEqual(metrics_2d["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
