#!/usr/bin/env python3
"""Render loss curves and reconstruction comparisons from saved runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Validated categorical palette (dataviz reference, light mode).
METHOD_COLORS = {
    "clomp": "#2a78d6",  # slot 1 blue
    "omp": "#eb6834",    # slot 2 orange
    "iht": "#1baf7a",    # slot 3 aqua (low contrast -> direct labels required)
}
METHOD_ORDER = ["clomp", "omp", "iht"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "axes.axisbelow": True,
        }
    )


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (run_dir / "losses.json").open(encoding="utf-8") as handle:
        losses = json.load(handle)
    predictions = np.load(run_dir / "predictions.npz")
    return summary, losses, predictions


def metric_label(metrics: dict) -> str:
    return f"SSIM {metrics['ssim']:.3f} · PSNR {metrics['psnr']:.1f} dB"


def settings_text(config: dict, kind: str) -> str:
    if kind == "1d":
        return (
            f"n={config['n']}, signals={config['num_signals']}, "
            f"Sr={config['sampling_rate']}, Kr={config['sparsity_rate']}, "
            f"CLOMP iterations={config['clomp']['iterations']}"
        )
    return (
        f"image {config['image_shape'][0]}x{config['image_shape'][1]}, "
        f"Sr={config['sampling_rate']}, "
        f"CLOMP iterations={config['clomp']['iterations']}"
    )


def plot_loss_curves(losses: dict, summary: dict, kind: str, out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in losses]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    units = (
        f"{summary['config']['clomp']['iterations']} CLOMP iterations"
        if kind == "1d"
        else f"{summary['config']['clomp']['iterations']} CLOMP / "
        f"{summary['config']['baseline_iterations']} baseline iterations"
    )

    for method in methods:
        steps = np.arange(1, len(losses[method]["loss"]) + 1)
        axes[0].semilogy(
            steps,
            losses[method]["loss"],
            color=METHOD_COLORS[method],
            linewidth=2.0,
            solid_capstyle="round",
            label=method.upper(),
        )
        axes[1].semilogy(
            steps,
            losses[method]["measurement_mse"],
            color=METHOD_COLORS[method],
            linewidth=2.0,
            solid_capstyle="round",
        )

    for ax, title in zip(axes, ["Total loss", "Measurement MSE"]):
        ax.set_xlabel("Iteration")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(left=1)
        ax.margins(x=0.01)

    fig.suptitle(
        f"{kind.upper()} reconstruction loss  ·  {settings_text(summary['config'], kind)}",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.905,
        units,
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    axes[0].legend(frameon=False, loc="upper right", fontsize=9, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_1d_reconstruction(
    predictions: np.ndarray, summary: dict, out_path: Path, num_rows: int = 3
) -> None:
    methods = [m for m in METHOD_ORDER if m in predictions.files]
    ground_truth = predictions["ground_truth"]
    num_rows = min(num_rows, ground_truth.shape[0])

    fig, axes = plt.subplots(num_rows, 1, figsize=(9.6, 2.1 * num_rows), sharex=False)
    axes = np.atleast_1d(axes)
    for row in range(num_rows):
        ax = axes[row]
        ax.plot(
            ground_truth[row],
            color=INK,
            linewidth=1.7,
            alpha=0.85,
            label="Ground truth",
        )
        for method in methods:
            ax.plot(
                predictions[method][row],
                color=METHOD_COLORS[method],
                linewidth=2.0,
                alpha=0.95,
                label=method.upper(),
            )
        ax.set_ylabel(f"Signal {row + 1}", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.01)
        if row == 0:
            ax.legend(
                frameon=False,
                loc="upper right",
                fontsize=9,
                ncols=4,
                handlelength=1.6,
                columnspacing=1.2,
            )

    metrics = summary["results"]
    label = " · ".join(
        f"{m.upper()} {metric_label(metrics[m])}" for m in methods
    )
    fig.suptitle(
        f"1D reconstruction overlay (signals 1-{num_rows})",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.945,
        f"Metrics averaged over all {ground_truth.shape[0]} signals: {label}",
        ha="left",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_reconstruction(
    predictions: np.ndarray, summary: dict, out_path: Path
) -> None:
    methods = [m for m in METHOD_ORDER if m in predictions.files]
    panels = [("ground_truth", "Ground truth", None)] + [
        (m, m.upper(), summary["results"][m]) for m in methods
    ]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(2.9 * len(panels), 3.4)
    )
    axes = np.atleast_1d(axes)
    for ax, (key, title, metrics) in zip(axes, panels):
        ax.imshow(predictions[key], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BASELINE)
            spine.set_linewidth(1.0)
        ax.set_title(title, loc="left", fontsize=11, pad=8)
        if metrics:
            ax.text(
                0.0,
                -0.12,
                metric_label(metrics),
                transform=ax.transAxes,
                fontsize=9,
                color=INK_2,
                va="top",
            )

    fig.suptitle(
        f"2D reconstruction  ·  {settings_text(summary['config'], '2d')}",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-1d", type=Path, default=repo_root / "pytorch_reproduction/runs/fig_1d_smoke"
    )
    parser.add_argument(
        "--run-2d", type=Path, default=repo_root / "pytorch_reproduction/runs/fig_2d_64"
    )
    parser.add_argument("--outdir", type=Path, default=repo_root / "figures")
    args = parser.parse_args()

    apply_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run_1d.exists():
        summary, losses, predictions = load_run(args.run_1d)
        plot_loss_curves(losses, summary, "1d", args.outdir / "1d_loss_curves.png")
        plot_1d_reconstruction(predictions, summary, args.outdir / "1d_reconstructions.png")
        print(f"1D figures written from {args.run_1d}")
    else:
        print(f"Skip 1D: {args.run_1d} not found")

    if args.run_2d.exists():
        summary, losses, predictions = load_run(args.run_2d)
        plot_loss_curves(losses, summary, "2d", args.outdir / "2d_loss_curves.png")
        plot_2d_reconstruction(predictions, summary, args.outdir / "2d_reconstruction.png")
        print(f"2D figures written from {args.run_2d}")
    else:
        print(f"Skip 2D: {args.run_2d} not found")


if __name__ == "__main__":
    main()
