#!/usr/bin/env python3
"""Reproduce the synthetic 1D CLOMP/OMP/IHT experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cscl.algorithms import (
    CLOMPConfig,
    reconstruct_clomp,
    reconstruct_iht,
    reconstruct_omp,
)
from cscl.data import generate_synthetic_1d
from cscl.metrics import evaluate_1d
from cscl.utils import (
    build_cs_problem,
    make_dct_basis,
    make_gaussian_measurement,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Signal length")
    parser.add_argument("--num-signals", type=int, default=1000)
    parser.add_argument("--sampling-rate", type=float, default=0.15)
    parser.add_argument("--sparsity-rate", type=float, default=0.4)
    parser.add_argument("--methods", default="clomp,omp,iht")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--min-learning-rate", type=float, default=0.01)
    parser.add_argument("--tv-weight", type=float, default=0.5)
    parser.add_argument("--stop-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--unmasked-prior",
        action="store_true",
        help="Prior and final output use unmasked coefficients (released-code behavior)",
    )
    parser.add_argument("--iht-sparsity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [name.strip().lower() for name in args.methods.split(",") if name.strip()]
    unknown = set(methods) - {"clomp", "omp", "iht"}
    if unknown or not methods:
        raise ValueError(f"Invalid methods: {sorted(unknown)}")

    output_dir = args.output_dir or Path(
        "pytorch_reproduction/runs"
    ) / f"1d_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    signals_np, _ = generate_synthetic_1d(
        args.n, args.num_signals, args.sparsity_rate, args.seed
    )
    signals = torch.from_numpy(signals_np).to(device)
    measurement = make_gaussian_measurement(args.n, args.sampling_rate, args.seed + 1).to(device)
    dct_basis = make_dct_basis(args.n).to(device)
    problem = build_cs_problem(signals, measurement, dct_basis)

    config = CLOMPConfig(
        iterations=args.iterations,
        threshold=args.threshold,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        tv_weight=args.tv_weight,
        stop_threshold=args.stop_threshold,
        log_every=args.log_every,
        unmasked_prior=args.unmasked_prior,
    )
    summary: dict = {
        "config": {
            **vars(args),
            "output_dir": str(output_dir),
            "resolved_device": str(device),
            "clomp": asdict(config),
            "trainable_parameters": args.num_signals * args.n,
        },
        "results": {},
    }
    histories: dict = {}
    predictions: dict[str, np.ndarray] = {"ground_truth": signals_np}

    for method in methods:
        print(f"\n=== {method.upper()} on {device} ===")
        if method == "clomp":
            result = reconstruct_clomp(problem, config)
        elif method == "omp":
            result = reconstruct_omp(
                problem,
                threshold=args.threshold,
                iterations=args.iterations,
                stop_threshold=args.stop_threshold,
            )
        else:
            result = reconstruct_iht(
                problem,
                iterations=args.iterations,
                stop_threshold=args.stop_threshold,
                sparsity=args.iht_sparsity,
            )

        prediction = result.signal.cpu().numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        metrics = evaluate_1d(signals_np, prediction)
        summary["results"][method] = {
            **metrics,
            "iterations": result.iterations,
            "elapsed_seconds": result.elapsed_seconds,
            "mean_support_size": float(result.support.sum(dim=1).float().mean()),
        }
        histories[method] = {
            "loss": result.losses,
            "measurement_mse": result.residual_losses,
        }
        predictions[method] = prediction
        print(
            f"SSIM={metrics['ssim']:.6f}  PSNR={metrics['psnr']:.3f} dB  "
            f"MSE={metrics['mse']:.6e}  PCC={metrics['pcc']:.6f}  "
            f"time={result.elapsed_seconds:.3f}s"
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    with (output_dir / "losses.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"method": method, **values} for method, values in summary["results"].items()]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_predictions:
        np.savez_compressed(output_dir / "predictions.npz", **predictions)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
