#!/usr/bin/env python3
"""Plot contractible-loop diagnostics from the released data tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


MODEL_ORDER = ("BHZ", "SSH", "BBH")
PANEL_LABELS = {"BHZ": "(a)", "SSH": "(b)", "BBH": "(c)"}
ORIENTATION_RADIUS = 0.532442


def add_halo(text):
    text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="black")])
    return text


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def circle_points(radius: float, n: int = 500):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return 1.0 + radius * np.cos(theta), 1.0 + radius * np.sin(theta)


def heatmap_grid(path: Path):
    rows = read_rows(path)
    xs = sorted({float(row["theta1_over_pi"]) for row in rows})
    ys = sorted({float(row["theta2_over_pi"]) for row in rows})
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    grid = np.empty((len(ys), len(xs)))
    for row in rows:
        x = float(row["theta1_over_pi"])
        y = float(row["theta2_over_pi"])
        grid[y_index[y], x_index[x]] = float(row["N_loc"])
    return grid


def plot_overlay(data_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.88, wspace=0.44)

    radii = [0.20, 0.40, 0.60, 0.80, 0.95]
    for ax, model in zip(axes, MODEL_ORDER):
        z = heatmap_grid(data_dir / f"heatmap_Nloc_{model}_n81.csv")
        im = ax.imshow(z, extent=(0, 2, 0, 2), origin="lower", aspect="equal", interpolation="bicubic")
        for radius in radii:
            x, y = circle_points(radius)
            ax.plot(x, y, color="white", lw=1.0, ls="--", alpha=0.68)
        x, y = circle_points(ORIENTATION_RADIUS)
        ax.plot(x, y, color="white", lw=2.0)
        ax.plot([1.0], [1.0], marker="o", color="white", ms=4.0)
        add_halo(ax.text(1.05, 1.06, r"$c_0=(1,1)$", color="white", fontsize=8.5))
        add_halo(ax.text(1.50, 1.65, r"$C_0(\rho)$", color="white", fontsize=10.0))
        ax.text(-0.13, 1.04, PANEL_LABELS[model], transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.set_title(model, fontsize=11)
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_ylabel(r"$\theta_2/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cb.set_label(r"$\mathcal{N}_{\rm loc}$", rotation=270, labelpad=13)

    fig.savefig(output_dir / "fig_contractible_loops.pdf")
    fig.savefig(output_dir / "fig_contractible_loops.png", dpi=300)
    plt.close(fig)


def plot_radius_scan(data_dir: Path, output_dir: Path) -> None:
    rows = read_rows(data_dir / "contractible_center11_radius_scan.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    for model in MODEL_ORDER:
        sub = [row for row in rows if row["model"] == model]
        sub.sort(key=lambda row: float(row["radius_over_pi"]))
        x = [float(row["radius_over_pi"]) for row in sub]
        y = [float(row["D_loc_strict"]) for row in sub]
        ax.plot(x, y, marker="o", ms=3.0, lw=1.35, label=model)
    ax.axvline(ORIENTATION_RADIUS, color="0.35", lw=1.0, ls="--")
    ax.text(ORIENTATION_RADIUS + 0.015, 0.94 * ax.get_ylim()[1], r"$\rho=0.532$", fontsize=9, va="top")
    ax.set_xlabel(r"radius $\rho$")
    ax.set_ylabel(r"$D_{\rm loc}$")
    ax.legend(frameon=False)
    fig.savefig(output_dir / "fig_contractible_radius_scan.pdf")
    fig.savefig(output_dir / "fig_contractible_radius_scan.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot contractible-loop figures from released data tables.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing CSV data tables")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_overlay(args.data, args.out)
    plot_radius_scan(args.data, args.out)


if __name__ == "__main__":
    main()
