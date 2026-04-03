#!/usr/bin/env python3
"""Compute additional grid-based datasets and loop diagnostics."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import expm

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


def entangling_power_stats(u: np.ndarray, n_samples: int = 2048, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_samples):
        psi = np.kron(core.random_qubit_state(rng), core.random_qubit_state(rng))
        out = u @ psi
        vals.append(core.linear_entropy_from_state(out))
    arr = np.maximum(np.array(vals, dtype=float), 0.0)
    return {
        'mean': float(np.mean(arr)),
        'stderr': float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
        'n_samples': int(n_samples),
    }


def holonomy_metrics(u: np.ndarray, seed: int = 0, ep_samples: int = 2048) -> Dict[str, object]:
    d_strict, payload_strict = core.best_local_procrustes(u, include_swap=False, n_restart=24, seed=seed)
    schmidt = core.operator_schmidt_values(u)
    ep = entangling_power_stats(u, n_samples=ep_samples, seed=seed)
    eigph = np.angle(np.linalg.eigvals(u))
    return {
        'D_loc_strict': float(d_strict),
        'ep_mean': float(ep['mean']),
        'ep_stderr': float(ep['stderr']),
        'ep_samples': int(ep['n_samples']),
        'schmidt': [float(x) for x in schmidt],
        'arg_det': float(np.angle(np.linalg.det(u))),
        'max_abs_phase': float(np.max(np.abs(eigph))),
        'nearest_local_gate': payload_strict['nearest_local_gate'],
    }


def unitary_hermitian_generator(u: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(u)
    phases = np.angle(vals)
    k = vecs @ np.diag(phases) @ np.linalg.inv(vecs)
    return 0.5 * (k + k.conj().T)


def projector_axis_fit(u: np.ndarray, projector: str) -> Dict[str, object]:
    sign = +1.0 if projector in {'left', 'top'} else -1.0
    k = unitary_hermitian_generator(u)
    PAULIS = {'I': I2, 'X': SX, 'Y': SY, 'Z': SZ}
    def coeff(a,b):
        return float(np.real_if_close(0.25*np.trace(np.kron(PAULIS[a], PAULIS[b]).conj().T @ k)))
    n = np.array([
        coeff('I','X') + sign * coeff('Z','X'),
        coeff('I','Y') + sign * coeff('Z','Y'),
        coeff('I','Z') + sign * coeff('Z','Z'),
    ], dtype=float)
    phi = float(np.linalg.norm(n))
    axis = n / max(phi, 1e-15)
    p = (I2 + sign * SZ) / 2.0
    k_fit = phi * np.kron(p, axis[0] * SX + axis[1] * SY + axis[2] * SZ)
    u_fit = expm(-1j * k_fit)
    return {
        'phi': phi,
        'axis_x': float(axis[0]),
        'axis_y': float(axis[1]),
        'axis_z': float(axis[2]),
        'generator_distance': float(np.linalg.norm(k - k_fit, ord='fro')),
        'unitary_distance': float(np.linalg.norm(u - u_fit, ord='fro')),
    }


def periodic_torus_grid(vals_a: np.ndarray, vals_b: np.ndarray, data_fn: Callable[[float, float], Dict[str, object]]):
    na, nb = len(vals_a), len(vals_b)
    frames: List[List[np.ndarray]] = [[None for _ in range(nb)] for _ in range(na)]  # type: ignore
    out = {k: np.zeros((na, nb), dtype=float) for k in ['gap','joint','deltaA','deltaB']}
    extras: Dict[str, np.ndarray] = {}
    for i, a in enumerate(vals_a):
        for j, b in enumerate(vals_b):
            d = data_fn(float(a), float(b))
            frames[i][j] = d['frame']
            out['gap'][i, j] = float(d['gap'])
            out['joint'][i, j] = float(d['joint'])
            out['deltaA'][i, j] = float(d['deltaA'])
            out['deltaB'][i, j] = float(d['deltaB'])
            for key, val in d.items():
                if key in {'frame','gap','joint','deltaA','deltaB','oa_tilde','ob_tilde'}:
                    continue
                if np.isscalar(val):
                    extras.setdefault(key, np.zeros((na, nb), dtype=float))[i, j] = float(np.real(val))
    da = float(vals_a[1] - vals_a[0])
    db = float(vals_b[1] - vals_b[0])
    nloc = np.zeros((na, nb), dtype=float)
    for i in range(na):
        for j in range(nb):
            nloc_a = core.link_entangling_connection(frames[i][j], frames[(i+1)%na][j], da)
            nloc_b = core.link_entangling_connection(frames[i][j], frames[i][(j+1)%nb], db)
            nloc[i, j] = 0.5 * (nloc_a + nloc_b)
    out['nloc'] = nloc
    out.update(extras)
    return out


def save_long_grid_data_file(path: Path, xvals: np.ndarray, yvals: np.ndarray, arrays: Dict[str, np.ndarray], xname: str, yname: str) -> None:
    rows = []
    for i, xv in enumerate(xvals):
        for j, yv in enumerate(yvals):
            row = {xname: float(xv), yname: float(yv)}
            for key, arr in arrays.items():
                row[key] = float(arr[i, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)

# SSH
ssh_cfg = core.SSHConfig()
ssh_oa = core.ssh_edge_observable(ssh_cfg)
ssh_ob = core.ssh_spin_observable(ssh_cfg)
ssh_refs = core.ssh_refs(ssh_cfg)

def ssh_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(theta_l: float, theta_r: float):
        h = core.build_spinful_ssh(cfg, theta_l, theta_r)
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
        edge_weight = 0.0
        for spin in (0, 1):
            edge_weight += weights[core.ssh_index(0, 0, spin, cfg.N)]
            edge_weight += weights[core.ssh_index(cfg.N - 1, 1, spin, cfg.N)]
        edge_weight /= 4.0
        return {'frame': frame, 'gap': gap, 'joint': core.jointness_metric(oa_t, ob_t), 'deltaA': delta_a, 'deltaB': delta_b, 'edge_weight': float(edge_weight), 'oa_tilde': oa_t, 'ob_tilde': ob_t}
    return _data
ssh_data = ssh_data_factory(ssh_cfg, ssh_oa, ssh_ob, ssh_refs)

def ssh_loop(kind: str, data_fn=ssh_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    if kind == 'left': pts = [(float(t), 0.0) for t in ts]
    elif kind == 'right': pts = [(0.0, float(t)) for t in ts]
    elif kind == 'diag': pts = [(float(t), float(t)) for t in ts]
    elif kind == 'anti': pts = [(float(t), float(-t)) for t in ts]
    else: raise ValueError(kind)
    return loop_frames(pts, data_fn)

# BBH
bbh_cfg = core.BBHConfig()
bbh_oa = core.bbh_pos_observable(bbh_cfg, 'x')
bbh_ob = core.bbh_pos_observable(bbh_cfg, 'y')
bbh_refs = core.bbh_refs(bbh_cfg)

def bbh_corner_weight(v, cfg=bbh_cfg):
    weights = np.sum(np.abs(v) ** 2, axis=1)
    total = 0.0
    corners = [(0, 0), (0, cfg.Ny - 1), (cfg.Nx - 1, 0), (cfg.Nx - 1, cfg.Ny - 1)]
    for x, y in corners:
        sl = slice(core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny), core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
        total += float(np.sum(weights[sl]))
    return total / 4.0

def bbh_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(theta_x: float, theta_y: float):
        h = core.build_bbh(cfg, theta_x, theta_y)
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
        delta_a, delta_b = delta_metrics_from_compressed(oa_t, ob_t)
        return {'frame': frame, 'gap': gap, 'joint': core.jointness_metric(oa_t, ob_t), 'deltaA': delta_a, 'deltaB': delta_b, 'corner_weight': bbh_corner_weight(v, cfg), 'oa_tilde': oa_t, 'ob_tilde': ob_t}
    return _data
bbh_data = bbh_data_factory(bbh_cfg, bbh_oa, bbh_ob, bbh_refs)

def bbh_loop(kind: str, data_fn=bbh_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    if kind == 'x': pts = [(float(t), 0.0) for t in ts]
    elif kind == 'y': pts = [(0.0, float(t)) for t in ts]
    elif kind == 'diag': pts = [(float(t), float(t)) for t in ts]
    elif kind == 'anti': pts = [(float(t), float(-t)) for t in ts]
    else: raise ValueError(kind)
    return loop_frames(pts, data_fn)

def bbh_average_density(theta_x: float=0.0, theta_y: float=0.0, cfg=bbh_cfg):
    h = core.build_bbh(cfg, theta_x, theta_y)
    evals, evecs = np.linalg.eigh(h)
    order = np.argsort(np.abs(evals))
    v = evecs[:, order[:4]]
    w = np.sum(np.abs(v)**2, axis=1)/4.0
    dens = np.zeros((cfg.Ny, cfg.Nx))
    for y in range(cfg.Ny):
        for x in range(cfg.Nx):
            sl = slice(core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny), core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny)+4)
            dens[cfg.Ny-1-y, x] = np.sum(w[sl])
    return dens

# BHZ for continuous eta scan
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
    def sl(y: int): return slice(4*y, 4*y+4)
    for y in range(cfg.Ly): h[sl(y), sl(y)] += h0
    for y in range(cfg.Ly-1):
        h[sl(y), sl(y+1)] += ty
        h[sl(y+1), sl(y)] += ty.conj().T
    edge_top = cfg.h_e * (math.cos(th_t) * kron(TAU0, SX) + math.sin(th_t) * kron(TAU0, SY))
    edge_bottom = cfg.h_e * (math.cos(th_b) * kron(TAU0, SX) + math.sin(th_b) * kron(TAU0, SY))
    h[sl(0), sl(0)] += edge_top
    h[sl(cfg.Ly-1), sl(cfg.Ly-1)] += edge_bottom
    return 0.5 * (h + h.conj().T)

def bhz_side_observable(cfg: BHZConfig = bhz_cfg):
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=float)
    ys = np.linspace(-1.0, 1.0, cfg.Ly)
    for y, val in enumerate(ys): out[4*y:4*y+4, 4*y:4*y+4] = val * np.eye(4)
    return out

def bhz_spin_observable(cfg: BHZConfig = bhz_cfg):
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=complex)
    blk = kron(TAU0, SZ)
    for y in range(cfg.Ly): out[4*y:4*y+4, 4*y:4*y+4] = blk
    return out

def bhz_refs(cfg: BHZConfig = bhz_cfg):
    return [1, 0, 4*(cfg.Ly-1)+1, 4*(cfg.Ly-1)+0]

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
        return {'frame': frame, 'gap': gap, 'joint': core.jointness_metric(oa_t, ob_t), 'deltaA': delta_a, 'deltaB': delta_b, 'edge_weight_2row': float(edge2), 'oa_tilde': oa_t, 'ob_tilde': ob_t}
    return _data
bhz_data = bhz_data_factory(bhz_cfg, bhz_oa, bhz_ob, bhz_refsites)

def bhz_loop(kind: str, kx: float = 0.0, data_fn=bhz_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    pts = []
    for t in ts:
        if kind == 'top': pts.append((kx, float(t), 0.0))
        elif kind == 'bottom': pts.append((kx, 0.0, float(t)))
        elif kind == 'diag': pts.append((kx, float(t), float(t)))
        elif kind == 'anti': pts.append((kx, float(t), float(-t)))
        else: raise ValueError(kind)
    return loop_frames(pts, data_fn)


def single_winding_eta_points(eta: float, n_main: int = 61, n_close: int = 21):
    pts = []
    for t in np.linspace(0.0, 2.0 * math.pi, n_main):
        pts.append((float(t % (2.0 * math.pi)), float((eta * t) % (2.0 * math.pi))))
    y_end = (eta * 2.0 * math.pi) % (2.0 * math.pi)
    if abs(y_end) > 1e-12 and abs(y_end - 2.0 * math.pi) > 1e-12:
        if y_end <= math.pi:
            yvals = np.linspace(y_end, 0.0, n_close)
            for y in yvals[1:]: pts.append((0.0, float(y)))
        else:
            yvals = np.linspace(y_end, 2.0 * math.pi, n_close // 2 + 1)
            for y in yvals[1:]: pts.append((0.0, float(y % (2.0 * math.pi))))
            for _ in range(max(0, n_close // 2 - 1)): pts.append((0.0, 0.0))
    return pts


def loop_quality_table(records, overlaps, extra_cols):
    rows = []
    n = len(records)
    ts = np.linspace(0.0, 1.0, n)
    for t, rec, ov in zip(ts, records, overlaps):
        row = {'s': float(t), 'gap': float(rec['gap']), 'joint': float(rec['joint']), 'deltaA': float(rec['deltaA']), 'deltaB': float(rec['deltaB']), 'frame_overlap': float(ov)}
        for col in extra_cols: row[col] = float(rec[col])
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_loop_scan_2d(model: str, data_fn, axis_loop, n_main: int, n_close: int, eta_grid: np.ndarray):
    rows = []
    for eta in eta_grid:
        frames, records, overlaps = loop_frames(single_winding_eta_points(float(eta), n_main=n_main, n_close=n_close), data_fn)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        rows.append({'model': model, 'eta_numeric': float(eta), 'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']), 'ep_stderr': float(hm['ep_stderr']), 'gap_min': float(min(float(r['gap']) for r in records)), 'joint_max': float(max(float(r['joint']) for r in records)), 'overlap_min': float(min(overlaps))})
    frames, records, overlaps = axis_loop()
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=1024)
    rows.append({'model': model, 'eta_numeric': float('nan'), 'eta_axis2': True, 'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']), 'ep_stderr': float(hm['ep_stderr']), 'gap_min': float(min(float(r['gap']) for r in records)), 'joint_max': float(max(float(r['joint']) for r in records)), 'overlap_min': float(min(overlaps))})
    return pd.DataFrame(rows)

print('Computing additional data tables...')
# High-resolution torus grids
ssh_thetas = np.linspace(0.0, 2.0 * math.pi, 81, endpoint=False)
ssh_grid = periodic_torus_grid(ssh_thetas, ssh_thetas, ssh_data)
save_long_grid_data_file(DATA / 'ssh_torus_grid.csv', ssh_thetas, ssh_thetas, {k: ssh_grid[k] for k in ['gap','joint','deltaA','deltaB','edge_weight','nloc']}, 'theta_L','theta_R')

bbh_thetas = np.linspace(0.0, 2.0 * math.pi, 61, endpoint=False)
bbh_grid = periodic_torus_grid(bbh_thetas, bbh_thetas, bbh_data)
save_long_grid_data_file(DATA / 'bbh_torus_grid.csv', bbh_thetas, bbh_thetas, {k: bbh_grid[k] for k in ['gap','joint','deltaA','deltaB','corner_weight','nloc']}, 'theta_x','theta_y')

# Quality curves and metrics
ssh_left_frames, ssh_left_records, ssh_left_overlaps = ssh_loop('left', n=121)
ssh_right_frames, ssh_right_records, ssh_right_overlaps = ssh_loop('right', n=121)
ssh_diag_frames, ssh_diag_records, ssh_diag_overlaps = ssh_loop('diag', n=121)
ssh_anti_frames, ssh_anti_records, ssh_anti_overlaps = ssh_loop('anti', n=121)
ssh_left_quality = loop_quality_table(ssh_left_records, ssh_left_overlaps, ['edge_weight'])
ssh_left_quality.to_csv(DATA / 'ssh_left_loop_quality.csv', index=False)

u_ssh_left = core.berry_holonomy(ssh_left_frames)
u_ssh_right = core.berry_holonomy(ssh_right_frames)
u_ssh_diag = core.berry_holonomy(ssh_diag_frames)
u_ssh_anti = core.berry_holonomy(ssh_anti_frames)
ssh_left_metrics = holonomy_metrics(u_ssh_left, seed=17, ep_samples=4096)
ssh_right_metrics = holonomy_metrics(u_ssh_right, seed=17, ep_samples=4096)
ssh_diag_metrics = holonomy_metrics(u_ssh_diag, seed=17, ep_samples=4096)
ssh_anti_metrics = holonomy_metrics(u_ssh_anti, seed=17, ep_samples=4096)
ssh_fit = projector_axis_fit(u_ssh_left, 'left')
pd.DataFrame(np.abs(u_ssh_left), index=['L↓','L↑','R↓','R↑'], columns=['L↓','L↑','R↓','R↑']).to_csv(DATA / 'ssh_left_loop_holonomy_matrix.csv')

ssh_size_rows = []
for N in [12,14,16,18,20]:
    cfg_n = replace(ssh_cfg, N=N)
    oa_n = core.ssh_edge_observable(cfg_n)
    ob_n = core.ssh_spin_observable(cfg_n)
    refs_n = core.ssh_refs(cfg_n)
    data_n = ssh_data_factory(cfg_n, oa_n, ob_n, refs_n)
    frames, records, overlaps = ssh_loop('left', data_fn=data_n, n=121)
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=1024)
    ssh_size_rows.append({'N': N, 'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']), 'gap_min': float(min(float(r['gap']) for r in records)), 'joint_max': float(max(float(r['joint']) for r in records))})
pd.DataFrame(ssh_size_rows).to_csv(DATA / 'ssh_size_scaling.csv', index=False)

# BBH quality, density, loop metrics
bbh_diag_frames, bbh_diag_records, bbh_diag_overlaps = bbh_loop('diag', n=121)
bbh_x_frames, bbh_x_records, bbh_x_overlaps = bbh_loop('x', n=121)
bbh_y_frames, bbh_y_records, bbh_y_overlaps = bbh_loop('y', n=121)
bbh_anti_frames, bbh_anti_records, bbh_anti_overlaps = bbh_loop('anti', n=121)
bbh_diag_quality = loop_quality_table(bbh_diag_records, bbh_diag_overlaps, ['corner_weight'])
bbh_diag_quality.to_csv(DATA / 'bbh_diagonal_loop_quality.csv', index=False)
pd.DataFrame(bbh_average_density(0.0,0.0)).to_csv(DATA / 'bbh_corner_density_reference.csv', index=False)

u_bbh_x = core.berry_holonomy(bbh_x_frames)
u_bbh_y = core.berry_holonomy(bbh_y_frames)
u_bbh_diag = core.berry_holonomy(bbh_diag_frames)
u_bbh_anti = core.berry_holonomy(bbh_anti_frames)
bbh_x_metrics = holonomy_metrics(u_bbh_x, seed=17, ep_samples=4096)
bbh_y_metrics = holonomy_metrics(u_bbh_y, seed=17, ep_samples=4096)
bbh_diag_metrics = holonomy_metrics(u_bbh_diag, seed=17, ep_samples=4096)
bbh_anti_metrics = holonomy_metrics(u_bbh_anti, seed=17, ep_samples=4096)

# Continuous eta scan
eta_grid = np.linspace(-2.0, 2.0, 81)
loop_family = pd.concat([
    continuous_loop_scan_2d('SSH', ssh_data, lambda: ssh_loop('right', n=121), n_main=61, n_close=21, eta_grid=eta_grid),
    continuous_loop_scan_2d('BBH', bbh_data, lambda: bbh_loop('y', n=121), n_main=41, n_close=15, eta_grid=eta_grid),
    continuous_loop_scan_2d('BHZ', lambda a,b: bhz_data(0.0, a, b), lambda: bhz_loop('bottom', 0.0, n=121), n_main=61, n_close=21, eta_grid=eta_grid),
], ignore_index=True)
loop_family.to_csv(DATA / 'continuous_loop_scan.csv', index=False)

# Summary update
with open(DATA / 'manuscript_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)
summary['ssh']['loops']['C_L'] = ssh_left_metrics
summary['ssh']['loops']['C_R'] = ssh_right_metrics
summary['ssh']['loops']['C_diag'] = ssh_diag_metrics
summary['ssh']['loops']['C_anti'] = ssh_anti_metrics
summary['ssh']['left_fit'] = ssh_fit
summary['ssh']['left_quality'] = {
    'gap_min': float(ssh_left_quality['gap'].min()),
    'joint_max': float(ssh_left_quality['joint'].max()),
    'deltaA_min': float(ssh_left_quality['deltaA'].min()),
    'deltaB_min': float(ssh_left_quality['deltaB'].min()),
    'edge_weight_min': float(ssh_left_quality['edge_weight'].min()),
    'overlap_min': float(ssh_left_quality['frame_overlap'].min()),
}
summary['bbh']['loops']['C_x'] = bbh_x_metrics
summary['bbh']['loops']['C_y'] = bbh_y_metrics
summary['bbh']['loops']['C_diag'] = bbh_diag_metrics
summary['bbh']['loops']['C_anti'] = bbh_anti_metrics
summary['bbh']['diag_quality'] = {
    'gap_min': float(bbh_diag_quality['gap'].min()),
    'joint_max': float(bbh_diag_quality['joint'].max()),
    'deltaA_min': float(bbh_diag_quality['deltaA'].min()),
    'deltaB_min': float(bbh_diag_quality['deltaB'].min()),
    'corner_weight_min': float(bbh_diag_quality['corner_weight'].min()),
    'overlap_min': float(bbh_diag_quality['frame_overlap'].min()),
}
summary['loop_family_scan'] = {
    'eta_window': [-2.0, 2.0],
    'n_eta': int(len(eta_grid)),
    'axis2_label': 'second-axis control',
    'peaks': {}
}
for model in ['SSH','BBH','BHZ']:
    sub = loop_family[(loop_family['model']==model) & (loop_family['eta_numeric'].notna())].copy()
    i_max = int(sub['D_loc'].idxmax())
    i_min = int(sub['D_loc'].idxmin())
    summary['loop_family_scan']['peaks'][model] = {
        'eta_at_max_Dloc': float(loop_family.loc[i_max,'eta_numeric']),
        'max_Dloc': float(loop_family.loc[i_max,'D_loc']),
        'eta_at_min_Dloc': float(loop_family.loc[i_min,'eta_numeric']),
        'min_Dloc': float(loop_family.loc[i_min,'D_loc']),
        'axis2_Dloc': float(loop_family[(loop_family['model']==model) & (loop_family['eta_numeric'].isna())]['D_loc'].iloc[0]),
    }
with open(DATA / 'manuscript_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print('Additional data tables complete.')
