#!/usr/bin/env python3
"""Run column-wise 2D image reconstruction with CLOMP, OMP, and IHT."""

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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cscl.algorithms import (
    CLOMPConfig,
    reconstruct_clomp,
    reconstruct_iht,
    reconstruct_omp,
)
from cscl.data import generate_synthetic_image
from cscl.metrics import evaluate_2d
from cscl.utils import (
    build_cs_problem,
    make_dct_basis,
    make_gaussian_measurement,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="Image file or .npy")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--sampling-rate", type=float, default=0.3)
    parser.add_argument("--methods", default="clomp,omp,iht")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--baseline-iterations", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--min-learning-rate", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.2)
    parser.add_argument("--lv-weight", type=float, default=0.1)
    parser.add_argument("--stop-threshold", type=float, default=1e-4)
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


def load_image(path: Path | None, height: int | None, width: int | None) -> np.ndarray:
    if (height is None) != (width is None):
        raise ValueError("Set both --height and --width, or neither.")
    if path is None:
        return generate_synthetic_image(height or 64, width or 64)

    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Install Pillow to decode image files, or use .npy.") from error
        with Image.open(path) as source:
            image = np.asarray(source.convert("L"))

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        if image.shape[-1] < 3:
            raise ValueError("Expected grayscale or RGB/RGBA input.")
        image = (
            0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        )
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")
    if min(image.shape) < 3:
        raise ValueError("Both image dimensions must be at least 3.")

    if height is not None and tuple(image.shape) != (height, width):
        tensor = torch.from_numpy(image)[None, None]
        image = F.interpolate(
            tensor, size=(height, width), mode="bilinear", align_corners=False
        )[0, 0].numpy()
    image = image - image.min()
    return image / max(float(image.max()), 1e-12)


def main() -> None:
    args = parse_args()
    methods = [name.strip().lower() for name in args.methods.split(",") if name.strip()]
    unknown = set(methods) - {"clomp", "omp", "iht"}
    if unknown or not methods:
        raise ValueError(f"Invalid methods: {sorted(unknown)}")

    output_dir = args.output_dir or Path(
        "pytorch_reproduction/runs"
    ) / f"2d_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    image = load_image(args.image, args.height, args.width)
    height, width = image.shape
    # Every column is an independent length-H signal, sensed by the same M.
    signals = torch.from_numpy(image.T.copy()).to(device)
    measurement = make_gaussian_measurement(height, args.sampling_rate, args.seed + 1).to(device)
    dct_basis = make_dct_basis(height).to(device)
    problem = build_cs_problem(signals, measurement, dct_basis)

    config = CLOMPConfig(
        iterations=args.iterations,
        threshold=args.threshold,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        tv_weight=args.tv_weight,
        lv_weight=args.lv_weight,
        stop_threshold=args.stop_threshold,
        plateau_delta=1e-7,
        stop_on_total_loss=True,
        log_every=args.log_every,
        unmasked_prior=args.unmasked_prior,
    )
    summary: dict = {
        "config": {
            **vars(args),
            "image": str(args.image) if args.image else "synthetic",
            "output_dir": str(output_dir),
            "resolved_device": str(device),
            "image_shape": [height, width],
            "clomp": asdict(config),
            "trainable_parameters": height * width,
        },
        "results": {},
    }
    histories: dict = {}
    predictions: dict[str, np.ndarray] = {"ground_truth": image}

    for method in methods:
        print(f"\n=== {method.upper()} on {device} ===")
        if method == "clomp":
            result = reconstruct_clomp(problem, config, image_shape=(height, width))
        elif method == "omp":
            result = reconstruct_omp(
                problem,
                threshold=args.threshold,
                iterations=args.baseline_iterations,
                stop_threshold=args.stop_threshold,
            )
        else:
            result = reconstruct_iht(
                problem,
                iterations=args.baseline_iterations,
                stop_threshold=args.stop_threshold,
                sparsity=args.iht_sparsity,
            )

        prediction = result.signal.cpu().numpy().T
        prediction = np.clip(prediction, 0.0, 1.0)
        metrics = evaluate_2d(image, prediction, data_range=1.0)
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
            f"MSE={metrics['mse']:.6e}  time={result.elapsed_seconds:.3f}s"
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
