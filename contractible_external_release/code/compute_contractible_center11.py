#!/usr/bin/env python3
"""Compute contractible-loop radius scans for the released models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


MODEL_ORDER = ("BHZ", "SSH", "BBH")
CENTER_THETA_OVER_PI = (1.0, 1.0)
DEFAULT_ORIENTATION_RADIUS = 0.532442

SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=complex,
)

FIELDNAMES = [
    "model",
    "orientation",
    "center_theta1_over_pi",
    "center_theta2_over_pi",
    "radius_over_pi",
    "n_points",
    "D_loc_strict",
    "D_loc_swap",
    "ep_exact",
    "arg_det",
    "max_abs_phase",
    "eigphase_1",
    "eigphase_2",
    "eigphase_3",
    "eigphase_4",
    "schmidt_1",
    "schmidt_2",
    "schmidt_3",
    "schmidt_4",
]


def load_release_modules(release_dir: Path):
    sys.path.insert(0, str(release_dir))
    source = (release_dir / "compute_grid_datasets.py").read_text()
    stop = "print('Computing additional data tables...')"
    module = types.ModuleType("contractible_release_definitions")
    module.__file__ = str(release_dir / "compute_grid_datasets.py")
    sys.modules[module.__name__] = module
    exec(source.split(stop)[0], module.__dict__)
    return module, module.core


def operator_schmidt_singular_values(unitary: np.ndarray) -> np.ndarray:
    reshaped = unitary.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    return np.linalg.svd(reshaped, compute_uv=False)


def operator_entanglement(unitary: np.ndarray) -> float:
    singular_values = operator_schmidt_singular_values(unitary)
    return float(1.0 - np.sum(singular_values**4) / 16.0)


def exact_entangling_power(unitary: np.ndarray) -> float:
    swap_entanglement = operator_entanglement(SWAP)
    return float((4.0 / 9.0) * (operator_entanglement(unitary) + operator_entanglement(unitary @ SWAP) - swap_entanglement))


def loop_points(radius: float, n_points: int, orientation: str) -> list[tuple[float, float]]:
    theta = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=True)
    x0, y0 = CENTER_THETA_OVER_PI
    points = [(x0 + radius * math.cos(t), y0 + radius * math.sin(t)) for t in theta]
    if orientation == "-":
        points = list(reversed(points))
    return points


def local_frames(release_module, model: str, points: list[tuple[float, float]]):
    if model == "BHZ":
        parameters = [(0.0, math.pi * x, math.pi * y) for x, y in points]
        return release_module.loop_frames(parameters, release_module.bhz_data)[0]
    if model == "SSH":
        parameters = [(math.pi * x, math.pi * y) for x, y in points]
        return release_module.loop_frames(parameters, release_module.ssh_data)[0]
    if model == "BBH":
        parameters = [(math.pi * x, math.pi * y) for x, y in points]
        return release_module.loop_frames(parameters, release_module.bbh_data)[0]
    raise ValueError(f"Unknown model: {model}")


def holonomy(release_module, core_module, model: str, radius: float, orientation: str, n_points: int) -> np.ndarray:
    frames = local_frames(release_module, model, loop_points(radius, n_points, orientation))
    return core_module.berry_holonomy(frames)


def unitary_metrics(core_module, unitary: np.ndarray) -> dict[str, float]:
    distance, _ = core_module.best_local_procrustes(unitary, include_swap=False, n_restart=24, seed=17)
    swap_distance, _ = core_module.best_local_procrustes(unitary, include_swap=True, n_restart=24, seed=17)
    eigenphases = np.array(sorted(np.angle(np.linalg.eigvals(unitary))))
    schmidt = operator_schmidt_singular_values(unitary)
    return {
        "D_loc_strict": float(distance),
        "D_loc_swap": float(swap_distance),
        "ep_exact": exact_entangling_power(unitary),
        "arg_det": float(np.angle(np.linalg.det(unitary))),
        "max_abs_phase": float(np.max(np.abs(eigenphases))),
        "eigphase_1": float(eigenphases[0]),
        "eigphase_2": float(eigenphases[1]),
        "eigphase_3": float(eigenphases[2]),
        "eigphase_4": float(eigenphases[3]),
        "schmidt_1": float(schmidt[0]),
        "schmidt_2": float(schmidt[1]),
        "schmidt_3": float(schmidt[2]),
        "schmidt_4": float(schmidt[3]),
    }


def default_radii() -> list[float]:
    coarse = np.arange(0.05, 0.951, 0.05)
    bbh_low = np.arange(0.175, 0.351, 0.005)
    mid = np.array([0.515, 0.520, 0.525, 0.530, DEFAULT_ORIENTATION_RADIUS, 0.535, 0.540, 0.545])
    return sorted({round(float(r), 6) for r in np.concatenate([coarse, bbh_low, mid])})


def radius_scan(release_module, core_module, radii: list[float], n_points: int) -> list[dict[str, float]]:
    rows = []
    for model in MODEL_ORDER:
        for radius in radii:
            unitary = holonomy(release_module, core_module, model, radius, "+", n_points)
            rows.append(
                {
                    "model": model,
                    "orientation": "+",
                    "center_theta1_over_pi": CENTER_THETA_OVER_PI[0],
                    "center_theta2_over_pi": CENTER_THETA_OVER_PI[1],
                    "radius_over_pi": float(radius),
                    "n_points": int(n_points),
                    **unitary_metrics(core_module, unitary),
                }
            )
    return rows


def orientation_check(release_module, core_module, radius: float, n_points: int) -> list[dict[str, float]]:
    rows = []
    for model in MODEL_ORDER:
        unitaries = {}
        start = len(rows)
        for orientation in ("+", "-"):
            unitary = holonomy(release_module, core_module, model, radius, orientation, n_points)
            unitaries[orientation] = unitary
            rows.append(
                {
                    "model": model,
                    "orientation": orientation,
                    "center_theta1_over_pi": CENTER_THETA_OVER_PI[0],
                    "center_theta2_over_pi": CENTER_THETA_OVER_PI[1],
                    "radius_over_pi": float(radius),
                    "n_points": int(n_points),
                    **unitary_metrics(core_module, unitary),
                }
            )
        inverse_norm = float(np.linalg.norm(unitaries["-"] - unitaries["+"].conj().T, ord="fro"))
        for row in rows[start:]:
            row["inverse_consistency_norm"] = inverse_norm
    return rows


def write_csv(path: Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(radius_rows: list[dict[str, float]], orientation_rows: list[dict[str, float]], radius: float, output_path: Path) -> None:
    bbh_rows = [row for row in radius_rows if row["model"] == "BBH"]
    largest = max(bbh_rows, key=lambda row: row["D_loc_strict"])
    summary = {
        "loop_family": "(theta1,theta2)/pi = (1,1) + rho (cos t, sin t)",
        "center_theta_over_pi": list(CENTER_THETA_OVER_PI),
        "orientation_check_radius_over_pi": float(radius),
        "n_points_per_loop": int(orientation_rows[0]["n_points"]),
        "models": list(MODEL_ORDER),
        "bbh_largest_D_loc_in_scan": largest,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute contractible-loop diagnostics for the released models.")
    parser.add_argument("--release", type=Path, required=True, help="Path to entanglement_obstruction_release")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-loop", type=int, default=121, help="Number of points per closed loop")
    parser.add_argument(
        "--orientation-radius",
        type=float,
        default=DEFAULT_ORIENTATION_RADIUS,
        help="Radius used for the orientation-reversal check",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    release_module, core_module = load_release_modules(args.release)

    scan_rows = radius_scan(release_module, core_module, default_radii(), args.n_loop)
    write_csv(args.out / "contractible_center11_radius_scan.csv", scan_rows, FIELDNAMES)

    orientation_rows = orientation_check(release_module, core_module, args.orientation_radius, args.n_loop)
    write_csv(
        args.out / "contractible_center11_orientation_check.csv",
        orientation_rows,
        FIELDNAMES + ["inverse_consistency_norm"],
    )

    write_summary(scan_rows, orientation_rows, args.orientation_radius, args.out / "contractible_center11_summary.json")


if __name__ == "__main__":
    main()
