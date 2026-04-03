#!/usr/bin/env python3
"""Compute supporting BHZ and BBH data tables used by the figures."""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

import core_models as core

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
TAU0 = I2
TAU1 = SX
TAU2 = SY
TAU3 = SZ


def kron(a, b):
    return np.kron(a, b)


def delta_metrics_from_compressed(oa_tilde: np.ndarray, ob_tilde: np.ndarray) -> Tuple[float, float]:
    a = 0.5 * (oa_tilde + oa_tilde.conj().T)
    b = 0.5 * (ob_tilde + ob_tilde.conj().T)
    evals_a, vecs_a = np.linalg.eigh(a)
    order = np.argsort(evals_a)
    evals_a = evals_a[order]
    vecs_a = vecs_a[:, order]
    delta_a = float(evals_a[2] - evals_a[1])
    deltas_b = []
    for idxs in [range(2), range(2, 4)]:
        sub = vecs_a[:, idxs]
        b_sub = sub.conj().T @ b @ sub
        evals_b = np.linalg.eigvalsh(0.5 * (b_sub + b_sub.conj().T))
        deltas_b.append(float(evals_b[1] - evals_b[0]))
    return delta_a, min(deltas_b)


def frame_overlap_min(f0: np.ndarray, f1: np.ndarray) -> float:
    return float(np.min(np.abs(np.diag(f0.conj().T @ f1))))


def loop_frames(loop_points: Sequence[Tuple[float, ...]], data_fn: Callable[..., Dict[str, object]]):
    frames: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    overlaps: List[float] = []
    prev: np.ndarray | None = None
    for pt in loop_points:
        datum = data_fn(*[float(x) for x in pt])
        frame = datum['frame']
        if prev is not None:
            overlaps.append(frame_overlap_min(prev, frame))
        frames.append(frame)
        records.append(datum)
        prev = frame
    overlaps.append(frame_overlap_min(frames[-1], frames[0]))
    return frames, records, overlaps


def loop_quality_table(records, overlaps, extra_cols):
    rows = []
    n = len(records)
    ts = np.linspace(0.0, 1.0, n)
    for t, rec, ov in zip(ts, records, overlaps):
        row = {
            's': float(t),
            'gap': float(rec['gap']),
            'joint': float(rec['joint']),
            'deltaA': float(rec['deltaA']),
            'deltaB': float(rec['deltaB']),
            'frame_overlap': float(ov),
        }
        for col in extra_cols:
            row[col] = float(rec[col])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BBH disorder window
# ---------------------------------------------------------------------------

def compute_bbh_disorder_window() -> pd.DataFrame:
    cfg = core.BBHConfig()
    oa = core.bbh_pos_observable(cfg, 'x')
    ob = core.bbh_pos_observable(cfg, 'y')
    refs = core.bbh_refs(cfg)

    def build_bbh_fixed(theta_x: float, theta_y: float, W: float, eta: np.ndarray) -> np.ndarray:
        h = core.build_bbh(cfg, theta_x, theta_y, disorder=0.0)
        for y in range(cfg.Ny):
            for x in range(cfg.Nx):
                sl = slice(core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny), core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
                h[sl, sl] += W * float(eta[y, x]) * np.eye(4)
        return 0.5 * (h + h.conj().T)

    rows = []
    for W in [0.00, 0.03, 0.06, 0.09]:
        vals = []
        gaps = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            eta = rng.random((cfg.Ny, cfg.Nx)) - 0.5
            ts = np.linspace(0.0, 2.0 * math.pi, 81)
            frames = []
            gap_min = np.inf
            for t in ts:
                h = build_bbh_fixed(float(t), float(t), W, eta)
                evals, evecs = np.linalg.eigh(h)
                order = np.argsort(np.abs(evals))
                evals = evals[order]
                evecs = evecs[:, order]
                v = evecs[:, :4]
                abs_sorted = np.sort(np.abs(evals))
                gap = float(abs_sorted[4] - abs_sorted[3])
                oa_t, ob_t = core.compress_observables(v, oa, ob)
                frame, _ = core.joint_diagonalize(v, oa_t, ob_t)
                frame = core.fix_column_phases(frame, refs)
                frames.append(frame)
                gap_min = min(gap_min, gap)
            u = core.berry_holonomy(frames)
            vals.append(float(core.best_local_procrustes(u, include_swap=False, seed=seed)[0]))
            gaps.append(float(gap_min))
        rows.append({
            'W': float(W),
            'D_loc_mean': float(np.mean(vals)),
            'D_loc_std': float(np.std(vals, ddof=1)),
            'gap_mean': float(np.mean(gaps)),
            'gap_std': float(np.std(gaps, ddof=1)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BHZ additional curves
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BHZConfig:
    Ly: int = 10
    M: float = 1.0
    B: float = 1.0
    A: float = 1.0
    lamR: float = 0.2
    h_e: float = 0.6


bhz_cfg = BHZConfig()


def bhz_h(kx: float, th_t: float, th_b: float, cfg: BHZConfig = bhz_cfg):
    dim = 4 * cfg.Ly
    h = np.zeros((dim, dim), dtype=complex)
    h0 = (cfg.M - 4.0 * cfg.B + 2.0 * cfg.B * math.cos(kx)) * kron(TAU3, TAU0)
    h0 += cfg.A * math.sin(kx) * kron(TAU1, SZ)
    h0 += cfg.lamR * math.sin(kx) * kron(TAU1, SY)
    ty = cfg.B * kron(TAU3, TAU0) - 0.5j * cfg.A * kron(TAU2, TAU0) + 0.5j * cfg.lamR * kron(TAU1, SX)
    def sl(y: int):
        return slice(4 * y, 4 * y + 4)
    for y in range(cfg.Ly):
        h[sl(y), sl(y)] += h0
    for y in range(cfg.Ly - 1):
        h[sl(y), sl(y + 1)] += ty
        h[sl(y + 1), sl(y)] += ty.conj().T
    edge_top = cfg.h_e * (math.cos(th_t) * kron(TAU0, SX) + math.sin(th_t) * kron(TAU0, SY))
    edge_bottom = cfg.h_e * (math.cos(th_b) * kron(TAU0, SX) + math.sin(th_b) * kron(TAU0, SY))
    h[sl(0), sl(0)] += edge_top
    h[sl(cfg.Ly - 1), sl(cfg.Ly - 1)] += edge_bottom
    return 0.5 * (h + h.conj().T)


def bhz_side_observable(cfg: BHZConfig = bhz_cfg):
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=float)
    ys = np.linspace(-1.0, 1.0, cfg.Ly)
    for y, val in enumerate(ys):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = val * np.eye(4)
    return out


def bhz_spin_observable(cfg: BHZConfig = bhz_cfg):
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=complex)
    blk = kron(TAU0, SZ)
    for y in range(cfg.Ly):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = blk
    return out


def bhz_refs(cfg: BHZConfig = bhz_cfg):
    return [1, 0, 4 * (cfg.Ly - 1) + 1, 4 * (cfg.Ly - 1) + 0]


bhz_oa = bhz_side_observable(bhz_cfg)
bhz_ob = bhz_spin_observable(bhz_cfg)
bhz_refsites = bhz_refs(bhz_cfg)


def bhz_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(kx: float, theta_t: float, theta_b: float):
        h = bhz_h(kx, theta_t, theta_b, cfg)
        evals, evecs = np.linalg.eigh(h)
        idx = np.argsort(np.abs(evals))[:4]
        idx = idx[np.argsort(evals[idx])]
        v = evecs[:, idx]
        abs_sorted = np.sort(np.abs(evals))
        gap = float(abs_sorted[4] - abs_sorted[3])
        oa_t, ob_t = core.compress_observables(v, oa, ob)
        frame, _ = core.joint_diagonalize(v, oa_t, ob_t)
        frame = core.fix_column_phases(frame, refs)
        delta_a, delta_b = delta_metrics_from_compressed(oa_t, ob_t)
        weights = np.sum(np.abs(v) ** 2, axis=1)
        edge2 = (np.sum(weights[:8]) + np.sum(weights[-8:])) / 4.0
        return {
            'frame': frame,
            'gap': gap,
            'joint': core.jointness_metric(oa_t, ob_t),
            'deltaA': delta_a,
            'deltaB': delta_b,
            'edge_weight_2row': float(edge2),
            'oa_tilde': oa_t,
            'ob_tilde': ob_t,
        }
    return _data


bhz_data = bhz_data_factory(bhz_cfg, bhz_oa, bhz_ob, bhz_refsites)


def bhz_loop(kind: str, kx: float = 0.0, data_fn=bhz_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    pts = []
    for t in ts:
        if kind == 'top':
            pts.append((kx, float(t), 0.0))
        elif kind == 'bottom':
            pts.append((kx, 0.0, float(t)))
        elif kind == 'diag':
            pts.append((kx, float(t), float(t)))
        elif kind == 'anti':
            pts.append((kx, float(t), float(-t)))
        else:
            raise ValueError(kind)
    return loop_frames(pts, data_fn)


def compute_bhz_phase_diagnostics() -> None:
    top_frames, _, _ = bhz_loop('top', 0.0, n=121)
    diag_frames, _, _ = bhz_loop('diag', 0.0, n=121)
    anti_frames, anti_records, anti_overlaps = bhz_loop('anti', 0.0, n=121)

    u_top = core.berry_holonomy(top_frames)
    u_diag = core.berry_holonomy(diag_frames)
    u_anti = core.berry_holonomy(anti_frames)

    pd.DataFrame(np.angle(np.diag(u_top)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_top_loop_phases.csv')
    pd.DataFrame(np.angle(np.diag(u_diag)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_corotating_loop_phases.csv')
    pd.DataFrame(np.angle(np.diag(u_anti)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_counterrotating_loop_phases.csv')

    bhz_anti_quality = loop_quality_table(anti_records, anti_overlaps, ['edge_weight_2row'])
    bhz_anti_quality.to_csv(DATA / 'bhz_counterrotating_loop_quality.csv', index=False)

    with open(DATA / 'manuscript_summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    width_df = pd.DataFrame(summary['bhz']['width_scaling'])
    width_df.to_csv(DATA / 'bhz_width_scaling.csv', index=False)


def main() -> None:
    print('Computing supporting figure datasets...')
    disorder_path = DATA / 'bbh_disorder_scan.csv'
    if not disorder_path.exists():
        print('  - BBH disorder window ...')
        compute_bbh_disorder_window().to_csv(disorder_path, index=False)
    else:
        print('  - BBH disorder scan already present; skipping')

    needed_bhz = [
        DATA / 'bhz_top_loop_phases.csv',
        DATA / 'bhz_corotating_loop_phases.csv',
        DATA / 'bhz_counterrotating_loop_phases.csv',
        DATA / 'bhz_counterrotating_loop_quality.csv',
        DATA / 'bhz_width_scaling.csv',
    ]
    if not all(p.exists() for p in needed_bhz):
        print('  - BHZ phase tables and diagnostics ...')
        compute_bhz_phase_diagnostics()
    else:
        print('  - BHZ phase tables already present; skipping')
    print('Done.')


if __name__ == '__main__':
    main()
