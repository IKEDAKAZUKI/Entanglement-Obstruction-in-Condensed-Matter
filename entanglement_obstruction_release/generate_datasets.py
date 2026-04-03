#!/usr/bin/env python3
"""Generate the principal data tables used by the manuscript."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

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


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def fmt_float(x: float) -> float:
    return float(np.real_if_close(x))


def delta_metrics_from_compressed(oa_tilde: np.ndarray, ob_tilde: np.ndarray) -> Tuple[float, float]:
    a = 0.5 * (oa_tilde + oa_tilde.conj().T)
    b = 0.5 * (ob_tilde + ob_tilde.conj().T)
    evals_a, vecs_a = np.linalg.eigh(a)
    order = np.argsort(evals_a)
    evals_a = evals_a[order]
    vecs_a = vecs_a[:, order]
    delta_a = float(evals_a[2] - evals_a[1])
    deltas_b: List[float] = []
    for idxs in [range(2), range(2, 4)]:
        sub = vecs_a[:, idxs]
        b_sub = sub.conj().T @ b @ sub
        evals_b = np.linalg.eigvalsh(0.5 * (b_sub + b_sub.conj().T))
        deltas_b.append(float(evals_b[1] - evals_b[0]))
    return delta_a, min(deltas_b)


def frame_overlap_min(f0: np.ndarray, f1: np.ndarray) -> float:
    return float(np.min(np.abs(np.diag(f0.conj().T @ f1))))


def loop_frames(loop_points: Sequence[Tuple[float, ...]], data_fn: Callable[..., Dict[str, object]]) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    frames: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    overlaps: List[float] = []
    prev: np.ndarray | None = None
    for pt in loop_points:
        datum = data_fn(*[float(x) for x in pt])
        frame = datum['frame']  # type: ignore[index]
        if prev is not None:
            overlaps.append(frame_overlap_min(prev, frame))
        frames.append(frame)
        records.append(datum)
        prev = frame
    overlaps.append(frame_overlap_min(frames[-1], frames[0]))
    return frames, records, overlaps


def pure_state_concurrence(psi: np.ndarray) -> float:
    a, b, c, d = psi
    return float(2.0 * abs(a * d - b * c))


def entangling_power_stats(u: np.ndarray, n_samples: int = 4096, seed: int = 0) -> Dict[str, float]:
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


def holonomy_metrics(u: np.ndarray, seed: int = 0, ep_samples: int = 4096) -> Dict[str, object]:
    d_strict, payload_strict = core.best_local_procrustes(u, include_swap=False, n_restart=24, seed=seed)
    d_swap, _ = core.best_local_procrustes(u, include_swap=True, n_restart=24, seed=seed)
    schmidt = core.operator_schmidt_values(u)
    ep = entangling_power_stats(u, n_samples=ep_samples, seed=seed)
    eigph = np.angle(np.linalg.eigvals(u))
    return {
        'D_loc_strict': float(d_strict),
        'D_loc_swap': float(d_swap),
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


PAULIS = {
    'I': I2,
    'X': SX,
    'Y': SY,
    'Z': SZ,
}


def pauli_coefficients(k: np.ndarray) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for a_name, a in PAULIS.items():
        for b_name, b in PAULIS.items():
            coeff = 0.25 * np.trace(np.kron(a, b).conj().T @ k)
            out[(a_name, b_name)] = float(np.real_if_close(coeff))
    return out


def projector_axis_fit(u: np.ndarray, projector: str) -> Dict[str, object]:
    sign = +1.0 if projector in {'left', 'top'} else -1.0
    k = unitary_hermitian_generator(u)
    coeff = pauli_coefficients(k)
    n = np.array([
        coeff[('I', 'X')] + sign * coeff[('Z', 'X')],
        coeff[('I', 'Y')] + sign * coeff[('Z', 'Y')],
        coeff[('I', 'Z')] + sign * coeff[('Z', 'Z')],
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


def single_term_fit(u: np.ndarray, label: str) -> Dict[str, float]:
    k = unitary_hermitian_generator(u)
    coeff = pauli_coefficients(k)
    if label == 'Iz':
        phi = coeff[('I', 'Z')]
        k_fit = phi * np.kron(I2, SZ)
    elif label == 'Zz':
        phi = coeff[('Z', 'Z')]
        k_fit = phi * np.kron(SZ, SZ)
    elif label == 'PTz':
        phi = coeff[('I', 'Z')] + coeff[('Z', 'Z')]
        k_fit = phi * np.kron((I2 + SZ) / 2.0, SZ)
    else:
        raise ValueError(label)
    u_fit = expm(-1j * k_fit)
    return {
        'phi': float(phi),
        'generator_distance': float(np.linalg.norm(k - k_fit, ord='fro')),
        'unitary_distance': float(np.linalg.norm(u - u_fit, ord='fro')),
    }


def chern_number_from_frames(frames: List[List[np.ndarray]], cols: Sequence[int] | None = None) -> float:
    na = len(frames)
    nb = len(frames[0])
    total = 0.0
    for i in range(na):
        for j in range(nb):
            f00 = frames[i][j]
            f10 = frames[(i + 1) % na][j]
            f11 = frames[(i + 1) % na][(j + 1) % nb]
            f01 = frames[i][(j + 1) % nb]
            if cols is not None:
                f00 = f00[:, cols]
                f10 = f10[:, cols]
                f11 = f11[:, cols]
                f01 = f01[:, cols]
            def link(a: np.ndarray, b: np.ndarray) -> complex:
                m = a.conj().T @ b
                det = np.linalg.det(m)
                return det / max(abs(det), 1e-15)
            u1 = link(f00, f10)
            u2 = link(f10, f11)
            u3 = link(f01, f11)
            u4 = link(f00, f01)
            total += np.angle(u1 * u2 / (u3 * u4))
    return float(total / (2.0 * math.pi))


def periodic_torus_grid(vals_a: np.ndarray, vals_b: np.ndarray, data_fn: Callable[[float, float], Dict[str, object]]) -> Tuple[Dict[str, np.ndarray], List[List[np.ndarray]]]:
    na = len(vals_a)
    nb = len(vals_b)
    frames: List[List[np.ndarray]] = [[None for _ in range(nb)] for _ in range(na)]  # type: ignore
    gap = np.zeros((na, nb), dtype=float)
    joint = np.zeros((na, nb), dtype=float)
    delta_a = np.zeros((na, nb), dtype=float)
    delta_b = np.zeros((na, nb), dtype=float)
    extras: Dict[str, np.ndarray] = {}

    for i, a in enumerate(vals_a):
        for j, b in enumerate(vals_b):
            d = data_fn(float(a), float(b))
            frames[i][j] = d['frame']  # type: ignore[index]
            gap[i, j] = float(d['gap'])
            joint[i, j] = float(d['joint'])
            delta_a[i, j] = float(d['deltaA'])
            delta_b[i, j] = float(d['deltaB'])
            for key, val in d.items():
                if key in {'frame', 'gap', 'joint', 'deltaA', 'deltaB', 'oa_tilde', 'ob_tilde'}:
                    continue
                if np.isscalar(val):
                    if key not in extras:
                        extras[key] = np.zeros((na, nb), dtype=float)
                    extras[key][i, j] = float(np.real(val))

    da = float(vals_a[1] - vals_a[0])
    db = float(vals_b[1] - vals_b[0])
    nloc = np.zeros((na, nb), dtype=float)
    for i in range(na):
        for j in range(nb):
            nloc_a = core.link_entangling_connection(frames[i][j], frames[(i + 1) % na][j], da)
            nloc_b = core.link_entangling_connection(frames[i][j], frames[i][(j + 1) % nb], db)
            nloc[i, j] = 0.5 * (nloc_a + nloc_b)

    out = {'gap': gap, 'joint': joint, 'deltaA': delta_a, 'deltaB': delta_b, 'nloc': nloc}
    out.update(extras)
    return out, frames


def save_long_grid_data_file(path: Path, xvals: np.ndarray, yvals: np.ndarray, arrays: Dict[str, np.ndarray], xname: str, yname: str) -> None:
    rows = []
    for i, xv in enumerate(xvals):
        for j, yv in enumerate(yvals):
            row = {xname: float(xv), yname: float(yv)}
            for key, arr in arrays.items():
                row[key] = float(arr[i, j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# SSH benchmark and controls
# -----------------------------------------------------------------------------
ssh_cfg = core.SSHConfig()
ssh_oa = core.ssh_edge_observable(ssh_cfg)
ssh_ob = core.ssh_spin_observable(ssh_cfg)
ssh_refs = core.ssh_refs(ssh_cfg)


def ssh_edge_observable_cubic(cfg: core.SSHConfig = ssh_cfg) -> np.ndarray:
    dim = 4 * cfg.N
    out = np.zeros((dim, dim), dtype=float)
    xs = np.linspace(-1.0, 1.0, cfg.N) ** 3
    for n, x in enumerate(xs):
        for sub in (0, 1):
            for spin in (0, 1):
                i = core.ssh_index(n, sub, spin, cfg.N)
                out[i, i] = x
    return out


def ssh_spin_observable_rot(alpha: float = 0.25, cfg: core.SSHConfig = ssh_cfg) -> np.ndarray:
    dim = 4 * cfg.N
    out = np.zeros((dim, dim), dtype=complex)
    mat = math.cos(alpha) * SZ + math.sin(alpha) * SX
    for n in range(cfg.N):
        for sub in (0, 1):
            for a in range(2):
                for b in range(2):
                    i = core.ssh_index(n, sub, a, cfg.N)
                    j = core.ssh_index(n, sub, b, cfg.N)
                    out[i, j] = mat[a, b]
    return out


ssh_oa_variant = ssh_edge_observable_cubic(ssh_cfg)
ssh_ob_variant = ssh_spin_observable_rot(cfg=ssh_cfg)


def ssh_data_factory(cfg: core.SSHConfig, oa: np.ndarray, ob: np.ndarray, refs: Sequence[int]) -> Callable[[float, float], Dict[str, object]]:
    @lru_cache(maxsize=None)
    def _data(theta_l: float, theta_r: float) -> Dict[str, object]:
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
        return {
            'frame': frame,
            'gap': gap,
            'joint': core.jointness_metric(oa_t, ob_t),
            'deltaA': delta_a,
            'deltaB': delta_b,
            'edge_weight': float(edge_weight),
            'oa_tilde': oa_t,
            'ob_tilde': ob_t,
        }
    return _data


ssh_data = ssh_data_factory(ssh_cfg, ssh_oa, ssh_ob, ssh_refs)
ssh_data_edge_variant = ssh_data_factory(ssh_cfg, ssh_oa_variant, ssh_ob, ssh_refs)
ssh_data_spin_variant = ssh_data_factory(ssh_cfg, ssh_oa, ssh_ob_variant, ssh_refs)
ssh_data_combined_variant = ssh_data_factory(ssh_cfg, ssh_oa_variant, ssh_ob_variant, ssh_refs)


# -----------------------------------------------------------------------------
# BBH benchmark and controls
# -----------------------------------------------------------------------------
bbh_cfg = core.BBHConfig()
bbh_oa = core.bbh_pos_observable(bbh_cfg, 'x')
bbh_ob = core.bbh_pos_observable(bbh_cfg, 'y')
bbh_refs = core.bbh_refs(bbh_cfg)


def bbh_pos_observable_cubic(axis: str, cfg: core.BBHConfig = bbh_cfg) -> np.ndarray:
    dim = 4 * cfg.Nx * cfg.Ny
    out = np.zeros((dim, dim), dtype=float)
    xs = np.linspace(-1.0, 1.0, cfg.Nx) ** 3
    ys = np.linspace(-1.0, 1.0, cfg.Ny) ** 3
    for y in range(cfg.Ny):
        for x in range(cfg.Nx):
            val = xs[x] if axis == 'x' else ys[y]
            sl = slice(core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny), core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
            out[sl, sl] = val * np.eye(4)
    return out


bbh_oa_variant = bbh_pos_observable_cubic('x', bbh_cfg)
bbh_ob_variant = bbh_pos_observable_cubic('y', bbh_cfg)


def bbh_corner_weight(v: np.ndarray, cfg: core.BBHConfig = bbh_cfg) -> float:
    weights = np.sum(np.abs(v) ** 2, axis=1)
    total = 0.0
    corners = [(0, 0), (0, cfg.Ny - 1), (cfg.Nx - 1, 0), (cfg.Nx - 1, cfg.Ny - 1)]
    for x, y in corners:
        sl = slice(core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny), core.bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
        total += float(np.sum(weights[sl]))
    return total / 4.0


def bbh_data_factory(cfg: core.BBHConfig, oa: np.ndarray, ob: np.ndarray, refs: Sequence[int]) -> Callable[[float, float], Dict[str, object]]:
    @lru_cache(maxsize=None)
    def _data(theta_x: float, theta_y: float) -> Dict[str, object]:
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
        return {
            'frame': frame,
            'gap': gap,
            'joint': core.jointness_metric(oa_t, ob_t),
            'deltaA': delta_a,
            'deltaB': delta_b,
            'corner_weight': bbh_corner_weight(v, cfg),
            'oa_tilde': oa_t,
            'ob_tilde': ob_t,
        }
    return _data


bbh_data = bbh_data_factory(bbh_cfg, bbh_oa, bbh_ob, bbh_refs)
bbh_data_variant = bbh_data_factory(bbh_cfg, bbh_oa_variant, bbh_ob_variant, bbh_refs)


# -----------------------------------------------------------------------------
# BHZ benchmark and controls
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BHZConfig:
    Ly: int = 10
    M: float = 1.0
    B: float = 1.0
    A: float = 1.0
    lamR: float = 0.2
    h_e: float = 0.6


bhz_cfg = BHZConfig()


def bhz_h(kx: float, th_t: float, th_b: float, cfg: BHZConfig = bhz_cfg) -> np.ndarray:
    dim = 4 * cfg.Ly
    h = np.zeros((dim, dim), dtype=complex)
    h0 = (cfg.M - 4.0 * cfg.B + 2.0 * cfg.B * math.cos(kx)) * kron(TAU3, TAU0)
    h0 += cfg.A * math.sin(kx) * kron(TAU1, SZ)
    h0 += cfg.lamR * math.sin(kx) * kron(TAU1, SY)
    ty = cfg.B * kron(TAU3, TAU0) - 0.5j * cfg.A * kron(TAU2, TAU0) + 0.5j * cfg.lamR * kron(TAU1, SX)

    def sl(y: int) -> slice:
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


def bhz_side_observable(cfg: BHZConfig = bhz_cfg) -> np.ndarray:
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=float)
    ys = np.linspace(-1.0, 1.0, cfg.Ly)
    for y, val in enumerate(ys):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = val * np.eye(4)
    return out


def bhz_side_observable_tanh(beta: float = 2.0, cfg: BHZConfig = bhz_cfg) -> np.ndarray:
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=float)
    ys = np.tanh(beta * np.linspace(-1.0, 1.0, cfg.Ly))
    for y, val in enumerate(ys):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = val * np.eye(4)
    return out


def bhz_spin_observable(cfg: BHZConfig = bhz_cfg) -> np.ndarray:
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=complex)
    blk = kron(TAU0, SZ)
    for y in range(cfg.Ly):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = blk
    return out


def bhz_spin_observable_rot(alpha: float = 0.25, cfg: BHZConfig = bhz_cfg) -> np.ndarray:
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=complex)
    blk = math.cos(alpha) * kron(TAU0, SZ) + math.sin(alpha) * kron(TAU0, SX)
    for y in range(cfg.Ly):
        out[4 * y:4 * y + 4, 4 * y:4 * y + 4] = blk
    return out


def bhz_refs(cfg: BHZConfig = bhz_cfg) -> List[int]:
    return [1, 0, 4 * (cfg.Ly - 1) + 1, 4 * (cfg.Ly - 1) + 0]


bhz_oa = bhz_side_observable(bhz_cfg)
bhz_ob = bhz_spin_observable(bhz_cfg)
bhz_oa_variant = bhz_side_observable_tanh(cfg=bhz_cfg)
bhz_ob_variant = bhz_spin_observable_rot(cfg=bhz_cfg)
bhz_refsites = bhz_refs(bhz_cfg)


def bhz_data_factory(cfg: BHZConfig, oa: np.ndarray, ob: np.ndarray, refs: Sequence[int]) -> Callable[[float, float, float], Dict[str, object]]:
    @lru_cache(maxsize=None)
    def _data(kx: float, theta_t: float, theta_b: float) -> Dict[str, object]:
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
        edge1 = (np.sum(weights[:4]) + np.sum(weights[-4:])) / 4.0
        return {
            'frame': frame,
            'gap': gap,
            'joint': core.jointness_metric(oa_t, ob_t),
            'deltaA': delta_a,
            'deltaB': delta_b,
            'edge_weight_2row': float(edge2),
            'edge_weight_1row': float(edge1),
            'oa_tilde': oa_t,
            'ob_tilde': ob_t,
        }
    return _data


bhz_data = bhz_data_factory(bhz_cfg, bhz_oa, bhz_ob, bhz_refsites)
bhz_data_side_variant = bhz_data_factory(bhz_cfg, bhz_oa_variant, bhz_ob, bhz_refsites)
bhz_data_spin_variant = bhz_data_factory(bhz_cfg, bhz_oa, bhz_ob_variant, bhz_refsites)
bhz_data_combined_variant = bhz_data_factory(bhz_cfg, bhz_oa_variant, bhz_ob_variant, bhz_refsites)


# -----------------------------------------------------------------------------
# Loop factories
# -----------------------------------------------------------------------------

def ssh_loop(kind: str, data_fn: Callable[[float, float], Dict[str, object]] = ssh_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    if kind == 'left':
        pts = [(float(t), 0.0) for t in ts]
    elif kind == 'right':
        pts = [(0.0, float(t)) for t in ts]
    elif kind == 'diag':
        pts = [(float(t), float(t)) for t in ts]
    elif kind == 'anti':
        pts = [(float(t), float(-t)) for t in ts]
    else:
        raise ValueError(kind)
    return loop_frames(pts, data_fn)



def bbh_loop(kind: str, data_fn: Callable[[float, float], Dict[str, object]] = bbh_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    if kind == 'x':
        pts = [(float(t), 0.0) for t in ts]
    elif kind == 'y':
        pts = [(0.0, float(t)) for t in ts]
    elif kind == 'diag':
        pts = [(float(t), float(t)) for t in ts]
    elif kind == 'anti':
        pts = [(float(t), float(-t)) for t in ts]
    else:
        raise ValueError(kind)
    return loop_frames(pts, data_fn)



def bhz_loop(kind: str, kx: float = 0.0, data_fn: Callable[[float, float, float], Dict[str, object]] = bhz_data, n: int = 121):
    ts = np.linspace(0.0, 2.0 * math.pi, n)
    pts: List[Tuple[float, float, float]] = []
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



def single_winding_eta_points(eta: float, n_main: int = 81, n_close: int = 31) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for t in np.linspace(0.0, 2.0 * math.pi, n_main):
        pts.append((float(t % (2.0 * math.pi)), float((eta * t) % (2.0 * math.pi))))
    y_end = (eta * 2.0 * math.pi) % (2.0 * math.pi)
    if abs(y_end) > 1e-12 and abs(y_end - 2.0 * math.pi) > 1e-12:
        if y_end <= math.pi:
            yvals = np.linspace(y_end, 0.0, n_close)
            for y in yvals[1:]:
                pts.append((0.0, float(y)))
        else:
            yvals = np.linspace(y_end, 2.0 * math.pi, n_close // 2 + 1)
            for y in yvals[1:]:
                pts.append((0.0, float(y % (2.0 * math.pi))))
            for _ in range(max(0, n_close // 2 - 1)):
                pts.append((0.0, 0.0))
    return pts


ETA_VALUES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def continuous_loop_scan_2d(model: str, data_fn: Callable[[float, float], Dict[str, object]], axis_loop: Callable[[], Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]], n_main: int, n_close: int) -> pd.DataFrame:
    rows = []
    for eta in ETA_VALUES:
        frames, records, overlaps = loop_frames(single_winding_eta_points(eta, n_main=n_main, n_close=n_close), data_fn)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        rows.append({
            'model': model,
            'eta_label': f'{eta:g}',
            'eta_numeric': eta,
            'D_loc': float(hm['D_loc_strict']),
            'ep_mean': float(hm['ep_mean']),
            'ep_stderr': float(hm['ep_stderr']),
            'gap_min': float(min(float(r['gap']) for r in records)),
            'joint_max': float(max(float(r['joint']) for r in records)),
            'overlap_min': float(min(overlaps)),
        })
    frames, records, overlaps = axis_loop()
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=1024)
    rows.append({
        'model': model,
        'eta_label': r'$\infty$',
        'eta_numeric': 9.0,
        'D_loc': float(hm['D_loc_strict']),
        'ep_mean': float(hm['ep_mean']),
        'ep_stderr': float(hm['ep_stderr']),
        'gap_min': float(min(float(r['gap']) for r in records)),
        'joint_max': float(max(float(r['joint']) for r in records)),
        'overlap_min': float(min(overlaps)),
    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Compute torus grids
# -----------------------------------------------------------------------------
print('Computing SSH torus grid...')
ssh_thetas = np.linspace(0.0, 2.0 * math.pi, 61, endpoint=False)
ssh_grid, ssh_frames_grid = periodic_torus_grid(ssh_thetas, ssh_thetas, ssh_data)

print('Computing BBH torus grid...')
bbh_thetas = np.linspace(0.0, 2.0 * math.pi, 31, endpoint=False)
bbh_grid, bbh_frames_grid = periodic_torus_grid(bbh_thetas, bbh_thetas, bbh_data)

print('Computing BHZ torus grid...')
bhz_thetas = np.linspace(0.0, 2.0 * math.pi, 61, endpoint=False)
bhz_grid, bhz_frames_grid = periodic_torus_grid(bhz_thetas, bhz_thetas, lambda a, b: bhz_data(0.0, a, b))

print('Computing BHZ annulus grid...')
ann_k = np.linspace(0.05, 0.35, 31)
ann_theta = np.linspace(0.0, 2.0 * math.pi, 81, endpoint=False)
ann_gap = np.zeros((len(ann_k), len(ann_theta)), dtype=float)
ann_joint = np.zeros_like(ann_gap)
ann_deltaA = np.zeros_like(ann_gap)
ann_deltaB = np.zeros_like(ann_gap)
ann_edge2 = np.zeros_like(ann_gap)
ann_frames: List[List[np.ndarray]] = [[None for _ in range(len(ann_theta))] for _ in range(len(ann_k))]  # type: ignore
for i, kval in enumerate(ann_k):
    for j, th in enumerate(ann_theta):
        d = bhz_data(float(kval), float(th), float(-th))
        ann_gap[i, j] = float(d['gap'])
        ann_joint[i, j] = float(d['joint'])
        ann_deltaA[i, j] = float(d['deltaA'])
        ann_deltaB[i, j] = float(d['deltaB'])
        ann_edge2[i, j] = float(d['edge_weight_2row'])
        ann_frames[i][j] = d['frame']
dtheta_ann = float(ann_theta[1] - ann_theta[0])
ann_nloc_theta = np.zeros_like(ann_gap)
for i in range(len(ann_k)):
    for j in range(len(ann_theta)):
        ann_nloc_theta[i, j] = core.link_entangling_connection(ann_frames[i][j], ann_frames[i][(j + 1) % len(ann_theta)], dtheta_ann)


# -----------------------------------------------------------------------------
# Representative loops and quality curves
# -----------------------------------------------------------------------------
print('Computing representative loop data...')

def loop_quality_table(records: List[Dict[str, object]], overlaps: List[float], extra_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    n = len(records)
    ts = np.linspace(0.0, 1.0, n)
    for i, (t, rec, ov) in enumerate(zip(ts, records, overlaps)):
        row = {
            's': float(t),
            'gap': float(rec['gap']),
            'joint': float(rec['joint']),
            'deltaA': float(rec['deltaA']),
            'deltaB': float(rec['deltaB']),
            'frame_overlap': float(ov),
        }
        for key in extra_cols:
            row[key] = float(rec[key])
        rows.append(row)
    return pd.DataFrame(rows)

ssh_left_frames, ssh_left_records, ssh_left_overlaps = ssh_loop('left')
ssh_right_frames, ssh_right_records, _ = ssh_loop('right')
ssh_diag_frames, ssh_diag_records, _ = ssh_loop('diag')
u_ssh_left = core.berry_holonomy(ssh_left_frames)
u_ssh_right = core.berry_holonomy(ssh_right_frames)
u_ssh_diag = core.berry_holonomy(ssh_diag_frames)
ssh_left_quality = loop_quality_table(ssh_left_records, ssh_left_overlaps, ['edge_weight'])

bbh_x_frames, bbh_x_records, _ = bbh_loop('x')
bbh_y_frames, bbh_y_records, _ = bbh_loop('y')
bbh_diag_frames, bbh_diag_records, bbh_diag_overlaps = bbh_loop('diag')
bbh_anti_frames, bbh_anti_records, _ = bbh_loop('anti')
u_bbh_x = core.berry_holonomy(bbh_x_frames)
u_bbh_y = core.berry_holonomy(bbh_y_frames)
u_bbh_diag = core.berry_holonomy(bbh_diag_frames)
u_bbh_anti = core.berry_holonomy(bbh_anti_frames)
bbh_diag_quality = loop_quality_table(bbh_diag_records, bbh_diag_overlaps, ['corner_weight'])

bhz_top_frames, bhz_top_records, _ = bhz_loop('top', 0.0)
bhz_bottom_frames, bhz_bottom_records, _ = bhz_loop('bottom', 0.0)
bhz_diag_frames, bhz_diag_records, _ = bhz_loop('diag', 0.0)
bhz_anti_frames, bhz_anti_records, bhz_anti_overlaps = bhz_loop('anti', 0.0)
u_bhz_top = core.berry_holonomy(bhz_top_frames)
u_bhz_bottom = core.berry_holonomy(bhz_bottom_frames)
u_bhz_diag = core.berry_holonomy(bhz_diag_frames)
u_bhz_anti = core.berry_holonomy(bhz_anti_frames)
bhz_anti_quality = loop_quality_table(bhz_anti_records, bhz_anti_overlaps, ['edge_weight_2row'])

# Width scaling for BHZ at kx = 0.
print('Computing BHZ width scaling...')
width_rows = []
for Ly in [6, 8, 10, 12]:
    cfg_w = replace(bhz_cfg, Ly=Ly)
    oa_w = bhz_side_observable(cfg_w)
    ob_w = bhz_spin_observable(cfg_w)
    refs_w = bhz_refs(cfg_w)
    data_w = bhz_data_factory(cfg_w, oa_w, ob_w, refs_w)
    for kind in ['top', 'diag', 'anti']:
        frames, records, overlaps = bhz_loop(kind, 0.0, data_fn=data_w, n=121)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        width_rows.append({
            'Ly': Ly,
            'loop': kind,
            'D_loc': float(hm['D_loc_strict']),
            'ep_mean': float(hm['ep_mean']),
            'gap_min': float(min(float(r['gap']) for r in records)),
            'joint_max': float(max(float(r['joint']) for r in records)),
        })
width_df = pd.DataFrame(width_rows)

# SSH finite-size scaling.
print('Computing SSH finite-size scaling...')
ssh_size_rows = []
for N in [12, 14, 16, 18]:
    cfg_n = replace(ssh_cfg, N=N)
    oa_n = core.ssh_edge_observable(cfg_n)
    ob_n = core.ssh_spin_observable(cfg_n)
    refs_n = core.ssh_refs(cfg_n)
    data_n = ssh_data_factory(cfg_n, oa_n, ob_n, refs_n)
    frames, records, overlaps = ssh_loop('left', data_fn=data_n, n=121)
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=1024)
    ssh_size_rows.append({
        'N': N,
        'D_loc': float(hm['D_loc_strict']),
        'ep_mean': float(hm['ep_mean']),
        'gap_min': float(min(float(r['gap']) for r in records)),
        'joint_max': float(max(float(r['joint']) for r in records)),
    })
ssh_size_df = pd.DataFrame(ssh_size_rows)

# -----------------------------------------------------------------------------
# Loop-family scan
# -----------------------------------------------------------------------------
print('Computing single-winding loop-family scan...')
loop_family_df = pd.concat([
    continuous_loop_scan_2d('SSH', ssh_data, lambda: ssh_loop('right', data_fn=ssh_data, n=121), n_main=81, n_close=31),
    continuous_loop_scan_2d('BBH', bbh_data, lambda: bbh_loop('y', data_fn=bbh_data, n=121), n_main=61, n_close=21),
    continuous_loop_scan_2d('BHZ', lambda a, b: bhz_data(0.0, a, b), lambda: bhz_loop('bottom', 0.0, data_fn=bhz_data, n=121), n_main=81, n_close=31),
], ignore_index=True)


# -----------------------------------------------------------------------------
# Observable robustness tests
# -----------------------------------------------------------------------------
print('Computing observable-robustness tests...')
rob_rows: List[Dict[str, object]] = []
for variant, data_fn in [
    ('reference', ssh_data),
    ('edge_variant', ssh_data_edge_variant),
    ('spin_variant', ssh_data_spin_variant),
    ('combined_variant', ssh_data_combined_variant),
]:
    for kind in ['left', 'diag']:
        frames, records, overlaps = ssh_loop(kind, data_fn=data_fn, n=121)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        rob_rows.append({
            'model': 'SSH', 'variant': variant, 'loop': kind,
            'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']),
            'gap_min': float(min(float(r['gap']) for r in records)),
            'joint_max': float(max(float(r['joint']) for r in records)),
        })
for variant, data_fn in [('reference', bbh_data), ('cubic_variant', bbh_data_variant)]:
    for kind in ['x', 'diag']:
        frames, records, overlaps = bbh_loop(kind, data_fn=data_fn, n=121)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        rob_rows.append({
            'model': 'BBH', 'variant': variant, 'loop': kind,
            'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']),
            'gap_min': float(min(float(r['gap']) for r in records)),
            'joint_max': float(max(float(r['joint']) for r in records)),
        })
for variant, data_fn in [
    ('reference', bhz_data),
    ('side_variant', bhz_data_side_variant),
    ('spin_variant', bhz_data_spin_variant),
    ('combined_variant', bhz_data_combined_variant),
]:
    for kind in ['diag', 'anti']:
        frames, records, overlaps = bhz_loop(kind, 0.0, data_fn=data_fn, n=121)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=1024)
        rob_rows.append({
            'model': 'BHZ', 'variant': variant, 'loop': kind,
            'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']),
            'gap_min': float(min(float(r['gap']) for r in records)),
            'joint_max': float(max(float(r['joint']) for r in records)),
        })
robustness_df = pd.DataFrame(rob_rows)


# -----------------------------------------------------------------------------
# Effective 4x4 mechanism data
# -----------------------------------------------------------------------------
print('Computing effective-generator decompositions...')
gen_rows = []
for name, u in [
    ('SSH_left', u_ssh_left),
    ('SSH_diag', u_ssh_diag),
    ('BBH_diag', u_bbh_diag),
    ('BHZ_top', u_bhz_top),
    ('BHZ_diag', u_bhz_diag),
    ('BHZ_anti', u_bhz_anti),
]:
    k = unitary_hermitian_generator(u)
    coeff = pauli_coefficients(k)
    for (a, b), c in coeff.items():
        gen_rows.append({'loop': name, 'pauli_left': a, 'pauli_right': b, 'coefficient': float(c)})
generator_df = pd.DataFrame(gen_rows)

fit_rows = []
fit_rows.append({'loop': 'SSH_left', 'fit': 'P_L⊗(n·σ)', **projector_axis_fit(u_ssh_left, 'left')})
fit_rows.append({'loop': 'BHZ_top', 'fit': 'P_T⊗Z', **projector_axis_fit(u_bhz_top, 'top')})
fit_rows.append({'loop': 'BHZ_diag', 'fit': 'I⊗Z', **single_term_fit(u_bhz_diag, 'Iz')})
fit_rows.append({'loop': 'BHZ_anti', 'fit': 'Z⊗Z', **single_term_fit(u_bhz_anti, 'Zz')})
fit_df = pd.DataFrame(fit_rows)


# -----------------------------------------------------------------------------
# Annulus k-curves and readout witness
# -----------------------------------------------------------------------------
print('Computing BHZ annulus fixed-kx loop curves and readout witness...')
psi_pp = np.kron(np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0), np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0))
ann_rows = []
for kval in np.linspace(float(ann_k[0]), float(ann_k[-1]), 17):
    top_frames, top_records, _ = bhz_loop('top', float(kval), data_fn=bhz_data, n=121)
    diag_frames, diag_records, _ = bhz_loop('diag', float(kval), data_fn=bhz_data, n=121)
    anti_frames, anti_records, _ = bhz_loop('anti', float(kval), data_fn=bhz_data, n=121)
    u_top = core.berry_holonomy(top_frames)
    u_diag = core.berry_holonomy(diag_frames)
    u_anti = core.berry_holonomy(anti_frames)
    mt = holonomy_metrics(u_top, seed=17, ep_samples=1024)
    md = holonomy_metrics(u_diag, seed=17, ep_samples=1024)
    ma = holonomy_metrics(u_anti, seed=17, ep_samples=1024)
    anti_fit = single_term_fit(u_anti, 'Zz')
    ann_rows.append({
        'kx': float(kval),
        'D_top': float(mt['D_loc_strict']),
        'D_diag': float(md['D_loc_strict']),
        'D_anti': float(ma['D_loc_strict']),
        'ep_top': float(mt['ep_mean']),
        'ep_diag': float(md['ep_mean']),
        'ep_anti': float(ma['ep_mean']),
        'anti_gap_min': float(min(float(r['gap']) for r in anti_records)),
        'anti_joint_max': float(max(float(r['joint']) for r in anti_records)),
        'anti_deltaA_min': float(min(float(r['deltaA']) for r in anti_records)),
        'anti_deltaB_min': float(min(float(r['deltaB']) for r in anti_records)),
        'anti_edge2_min': float(min(float(r['edge_weight_2row']) for r in anti_records)),
        'anti_fit_ZZ': float(anti_fit['unitary_distance']),
        'anti_phi_ZZ': float(anti_fit['phi']),
        'Cout_top_plusplus': pure_state_concurrence(u_top @ psi_pp),
        'Cout_diag_plusplus': pure_state_concurrence(u_diag @ psi_pp),
        'Cout_anti_plusplus': pure_state_concurrence(u_anti @ psi_pp),
    })
ann_curves = pd.DataFrame(ann_rows)


# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
print('Computing control scans...')
# SSH explicit B scan.
ssh_b_rows = []
for Bval in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
    cfg_b = replace(ssh_cfg, B=Bval)
    oa_b = core.ssh_edge_observable(cfg_b)
    ob_b = core.ssh_spin_observable(cfg_b)
    refs_b = core.ssh_refs(cfg_b)
    data_b = ssh_data_factory(cfg_b, oa_b, ob_b, refs_b)
    frames, records, overlaps = ssh_loop('left', data_fn=data_b, n=121)
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=2048)
    ssh_b_rows.append({
        'B': float(Bval),
        'D_loc': float(hm['D_loc_strict']),
        'ep_mean': float(hm['ep_mean']),
        'ep_stderr': float(hm['ep_stderr']),
        'gap_min': float(min(float(r['gap']) for r in records)),
        'joint_max': float(max(float(r['joint']) for r in records)),
    })
ssh_B_df = pd.DataFrame(ssh_b_rows)
small = ssh_B_df[ssh_B_df['B'] <= 0.10]
coef_D = np.polyfit((small['B'].to_numpy()) ** 2, small['D_loc'].to_numpy(), 1)
coef_ep = np.polyfit((small['B'].to_numpy()) ** 4, small['ep_mean'].to_numpy(), 1)

# BBH disorder window.
disorder_rows = []
for W in [0.00, 0.03, 0.06, 0.09]:
    vals = []
    gaps = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ts = np.linspace(0.0, 2.0 * math.pi, 81)
        frames = []
        gap_min = np.inf
        for t in ts:
            h = core.build_bbh(bbh_cfg, float(t), float(t), disorder=W, rng=rng)
            evals, evecs = np.linalg.eigh(h)
            order = np.argsort(np.abs(evals))
            evals = evals[order]
            evecs = evecs[:, order]
            v = evecs[:, :4]
            abs_sorted = np.sort(np.abs(evals))
            gap = float(abs_sorted[4] - abs_sorted[3])
            oa_t, ob_t = core.compress_observables(v, bbh_oa, bbh_ob)
            frame, _ = core.joint_diagonalize(v, oa_t, ob_t)
            frame = core.fix_column_phases(frame, bbh_refs)
            frames.append(frame)
            gap_min = min(gap_min, gap)
        u = core.berry_holonomy(frames)
        vals.append(float(core.best_local_procrustes(u, include_swap=False, seed=seed)[0]))
        gaps.append(float(gap_min))
    disorder_rows.append({
        'W': float(W),
        'D_loc_mean': float(np.mean(vals)),
        'D_loc_std': float(np.std(vals, ddof=1)),
        'gap_mean': float(np.mean(gaps)),
        'gap_std': float(np.std(gaps, ddof=1)),
    })
disorder_df = pd.DataFrame(disorder_rows)

# Trivial controls.
trivial_rows = []
# SSH trivial dimerization.
ssh_triv_cfg = replace(ssh_cfg, t1=1.0, t2=0.55)
ssh_triv = ssh_data_factory(ssh_triv_cfg, core.ssh_edge_observable(ssh_triv_cfg), core.ssh_spin_observable(ssh_triv_cfg), core.ssh_refs(ssh_triv_cfg))
vals = [ssh_triv(0.0, 0.0)]
trivial_rows.append({
    'model': 'SSH_trivial',
    'local_weight': float(vals[0]['edge_weight']),
    'joint_max': float(vals[0]['joint']),
    'gap_min': float(vals[0]['gap']),
})
# BBH non-HOTI.
bbh_triv_cfg = replace(bbh_cfg, gx=0.8, gy=0.8)
bbh_triv = bbh_data_factory(bbh_triv_cfg, core.bbh_pos_observable(bbh_triv_cfg, 'x'), core.bbh_pos_observable(bbh_triv_cfg, 'y'), core.bbh_refs(bbh_triv_cfg))
vals = [bbh_triv(0.0, 0.0)]
trivial_rows.append({
    'model': 'BBH_trivial',
    'local_weight': float(vals[0]['corner_weight']),
    'joint_max': float(vals[0]['joint']),
    'gap_min': float(vals[0]['gap']),
})
trivial_df = pd.DataFrame(trivial_rows)


# -----------------------------------------------------------------------------
# Save raw data tables
# -----------------------------------------------------------------------------
print('Saving raw data tables...')
save_long_grid_data_file(DATA / 'ssh_torus_grid.csv', ssh_thetas, ssh_thetas, {k: ssh_grid[k] for k in ['gap', 'joint', 'deltaA', 'deltaB', 'edge_weight', 'nloc']}, 'theta_L', 'theta_R')
save_long_grid_data_file(DATA / 'bbh_torus_grid.csv', bbh_thetas, bbh_thetas, {k: bbh_grid[k] for k in ['gap', 'joint', 'deltaA', 'deltaB', 'corner_weight', 'nloc']}, 'theta_x', 'theta_y')
save_long_grid_data_file(DATA / 'bhz_torus_grid.csv', bhz_thetas, bhz_thetas, {k: bhz_grid[k] for k in ['gap', 'joint', 'deltaA', 'deltaB', 'edge_weight_2row', 'nloc']}, 'theta_T', 'theta_B')
save_long_grid_data_file(DATA / 'bhz_annulus_grid.csv', ann_k, ann_theta, {'gap': ann_gap, 'joint': ann_joint, 'deltaA': ann_deltaA, 'deltaB': ann_deltaB, 'edge_weight_2row': ann_edge2, 'nloc_theta': ann_nloc_theta}, 'kx', 'vartheta')
ssh_left_quality.to_csv(DATA / 'ssh_left_loop_quality.csv', index=False)
bbh_diag_quality.to_csv(DATA / 'bbh_diagonal_loop_quality.csv', index=False)
bhz_anti_quality.to_csv(DATA / 'bhz_counterrotating_loop_quality.csv', index=False)
width_df.to_csv(DATA / 'bhz_width_scaling.csv', index=False)
ssh_size_df.to_csv(DATA / 'ssh_size_scaling.csv', index=False)
loop_family_df.to_csv(DATA / 'continuous_loop_scan.csv', index=False)
robustness_df.to_csv(DATA / 'observable_robustness_scan.csv', index=False)
generator_df.to_csv(DATA / 'fitted_generator_matrices.csv', index=False)
fit_df.to_csv(DATA / 'fitted_generator_coefficients.csv', index=False)
ann_curves.to_csv(DATA / 'bhz_annulus_k_slices.csv', index=False)
ssh_B_df.to_csv(DATA / 'ssh_small_field_scan.csv', index=False)
disorder_df.to_csv(DATA / 'bbh_disorder_scan.csv', index=False)
trivial_df.to_csv(DATA / 'control_diagnostics.csv', index=False)
pd.DataFrame(np.abs(u_ssh_left), index=['L↓', 'L↑', 'R↓', 'R↑'], columns=['L↓', 'L↑', 'R↓', 'R↑']).to_csv(DATA / 'ssh_left_loop_holonomy_matrix.csv')
pd.DataFrame(np.angle(np.diag(u_bhz_top)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_top_loop_phases.csv')
pd.DataFrame(np.angle(np.diag(u_bhz_diag)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_corotating_loop_phases.csv')
pd.DataFrame(np.angle(np.diag(u_bhz_anti)), index=['T↓', 'T↑', 'B↓', 'B↑'], columns=['phase']).to_csv(DATA / 'bhz_counterrotating_loop_phases.csv')

# -----------------------------------------------------------------------------
# Summary JSON
# -----------------------------------------------------------------------------
print('Writing summary file...')
ssh_left_metrics = holonomy_metrics(u_ssh_left, seed=17, ep_samples=4096)
ssh_right_metrics = holonomy_metrics(u_ssh_right, seed=17, ep_samples=4096)
ssh_diag_metrics = holonomy_metrics(u_ssh_diag, seed=17, ep_samples=4096)

bbh_x_metrics = holonomy_metrics(u_bbh_x, seed=17, ep_samples=4096)
bbh_y_metrics = holonomy_metrics(u_bbh_y, seed=17, ep_samples=4096)
bbh_diag_metrics = holonomy_metrics(u_bbh_diag, seed=17, ep_samples=4096)
bbh_anti_metrics = holonomy_metrics(u_bbh_anti, seed=17, ep_samples=4096)

bhz_top_metrics = holonomy_metrics(u_bhz_top, seed=17, ep_samples=4096)
bhz_bottom_metrics = holonomy_metrics(u_bhz_bottom, seed=17, ep_samples=4096)
bhz_diag_metrics = holonomy_metrics(u_bhz_diag, seed=17, ep_samples=4096)
bhz_anti_metrics = holonomy_metrics(u_bhz_anti, seed=17, ep_samples=4096)

summary = {
    'ssh': {
        'gap_min': float(np.min(ssh_grid['gap'])),
        'joint_max': float(np.max(ssh_grid['joint'])),
        'deltaA_min': float(np.min(ssh_grid['deltaA'])),
        'deltaB_min': float(np.min(ssh_grid['deltaB'])),
        'edge_weight_min': float(np.min(ssh_grid['edge_weight'])),
        'loops': {
            'C_L': ssh_left_metrics,
            'C_R': ssh_right_metrics,
            'C_diag': ssh_diag_metrics,
        },
        'left_quality': {
            'gap_min': float(ssh_left_quality['gap'].min()),
            'joint_max': float(ssh_left_quality['joint'].max()),
            'deltaA_min': float(ssh_left_quality['deltaA'].min()),
            'deltaB_min': float(ssh_left_quality['deltaB'].min()),
            'edge_weight_min': float(ssh_left_quality['edge_weight'].min()),
            'overlap_min': float(ssh_left_quality['frame_overlap'].min()),
        },
        'mechanism_fit': projector_axis_fit(u_ssh_left, 'left'),
        'B_scan': ssh_b_rows,
        'small_B_fit': {
            'D_loc_coeff': float(coef_D[0]),
            'D_loc_intercept': float(coef_D[1]),
            'ep_coeff': float(coef_ep[0]),
            'ep_intercept': float(coef_ep[1]),
            'fit_window_max_B': 0.10,
        },
    },
    'bbh': {
        'gap_min': float(np.min(bbh_grid['gap'])),
        'joint_max': float(np.max(bbh_grid['joint'])),
        'deltaA_min': float(np.min(bbh_grid['deltaA'])),
        'deltaB_min': float(np.min(bbh_grid['deltaB'])),
        'corner_weight_min': float(np.min(bbh_grid['corner_weight'])),
        'loops': {
            'C_x': bbh_x_metrics,
            'C_y': bbh_y_metrics,
            'C_diag': bbh_diag_metrics,
            'C_anti': bbh_anti_metrics,
        },
        'diag_quality': {
            'gap_min': float(bbh_diag_quality['gap'].min()),
            'joint_max': float(bbh_diag_quality['joint'].max()),
            'deltaA_min': float(bbh_diag_quality['deltaA'].min()),
            'deltaB_min': float(bbh_diag_quality['deltaB'].min()),
            'corner_weight_min': float(bbh_diag_quality['corner_weight'].min()),
            'overlap_min': float(bbh_diag_quality['frame_overlap'].min()),
        },
        'disorder_window': disorder_rows,
    },
    'bhz': {
        'gap_min': float(np.min(bhz_grid['gap'])),
        'joint_max': float(np.max(bhz_grid['joint'])),
        'deltaA_min': float(np.min(bhz_grid['deltaA'])),
        'deltaB_min': float(np.min(bhz_grid['deltaB'])),
        'edge_weight_2row_min': float(np.min(bhz_grid['edge_weight_2row'])),
        'loops': {
            'C_T': bhz_top_metrics,
            'C_B': bhz_bottom_metrics,
            'C_+': bhz_diag_metrics,
            'C_-': bhz_anti_metrics,
        },
        'anti_quality': {
            'gap_min': float(bhz_anti_quality['gap'].min()),
            'joint_max': float(bhz_anti_quality['joint'].max()),
            'deltaA_min': float(bhz_anti_quality['deltaA'].min()),
            'deltaB_min': float(bhz_anti_quality['deltaB'].min()),
            'edge_weight_2row_min': float(bhz_anti_quality['edge_weight_2row'].min()),
            'overlap_min': float(bhz_anti_quality['frame_overlap'].min()),
        },
        'mechanism_fits': {
            'C_T': projector_axis_fit(u_bhz_top, 'top'),
            'C_+': single_term_fit(u_bhz_diag, 'Iz'),
            'C_-': single_term_fit(u_bhz_anti, 'Zz'),
        },
        'width_scaling': width_rows,
    },
    'bhz_annulus': {
        'k_window': [float(ann_k[0]), float(ann_k[-1])],
        'gap_min': float(np.min(ann_gap)),
        'joint_max': float(np.max(ann_joint)),
        'deltaA_min': float(np.min(ann_deltaA)),
        'deltaB_min': float(np.min(ann_deltaB)),
        'edge_weight_2row_min': float(np.min(ann_edge2)),
        'nloc_theta_max': float(np.max(ann_nloc_theta)),
        'D_top_min': float(ann_curves['D_top'].min()),
        'D_top_max': float(ann_curves['D_top'].max()),
        'D_diag_min': float(ann_curves['D_diag'].min()),
        'D_diag_max': float(ann_curves['D_diag'].max()),
        'D_anti_min': float(ann_curves['D_anti'].min()),
        'D_anti_max': float(ann_curves['D_anti'].max()),
        'ZZ_fit_min': float(ann_curves['anti_fit_ZZ'].min()),
        'ZZ_fit_max': float(ann_curves['anti_fit_ZZ'].max()),
        'phi_ZZ_min': float(ann_curves['anti_phi_ZZ'].min()),
        'phi_ZZ_max': float(ann_curves['anti_phi_ZZ'].max()),
        'Cout_anti_min': float(ann_curves['Cout_anti_plusplus'].min()),
        'Cout_anti_max': float(ann_curves['Cout_anti_plusplus'].max()),
        'Cout_diag_max': float(ann_curves['Cout_diag_plusplus'].max()),
    },
    'loop_family_scan': {
        'eta_values': ETA_VALUES + ['inf'],
    },
    'observable_robustness': {
        'csv': 'observable_robustness_scan.csv',
    },
    'control_diagnostics': trivial_rows,
    'berry_comparison': {
        'quartet_chern': {
            'SSH': chern_number_from_frames(ssh_frames_grid),
            'BBH': chern_number_from_frames(bbh_frames_grid),
            'BHZ': chern_number_from_frames(bhz_frames_grid),
        },
        'block_chern': {
            'SSH_A': chern_number_from_frames(ssh_frames_grid, cols=[0, 1]),
            'SSH_B': chern_number_from_frames(ssh_frames_grid, cols=[2, 3]),
            'BBH_A': chern_number_from_frames(bbh_frames_grid, cols=[0, 1]),
            'BBH_B': chern_number_from_frames(bbh_frames_grid, cols=[2, 3]),
            'BHZ_A': chern_number_from_frames(bhz_frames_grid, cols=[0, 1]),
            'BHZ_B': chern_number_from_frames(bhz_frames_grid, cols=[2, 3]),
        },
    },
}

with open(DATA / 'manuscript_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('Data tables written.')
