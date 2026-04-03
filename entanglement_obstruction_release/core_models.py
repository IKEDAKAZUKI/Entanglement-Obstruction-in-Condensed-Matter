#!/usr/bin/env python3
"""Shared model, quartet, and holonomy utilities for the release scripts."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import polar

# -----------------------------------------------------------------------------
# Basic matrices
# -----------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, SX, SY, SZ]

TAU0 = I2
TAU1 = SX
TAU2 = SY
TAU3 = SZ


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


# BBH Clifford matrices in a standard convention.
GAMMA1 = -kron(TAU2, SX)
GAMMA2 = -kron(TAU2, SY)
GAMMA3 = -kron(TAU2, SZ)
GAMMA4 = kron(TAU1, I2)
GAMMA0 = kron(TAU3, I2)

# Additional onsite matrices used for local boundary / corner fields.
BBH_M1 = kron(TAU0, SZ)
BBH_M2 = kron(TAU3, SZ)
BBH_M3 = kron(TAU0, SX)
BBH_M4 = kron(TAU0, SY)


# -----------------------------------------------------------------------------
# Generic linear algebra helpers
# -----------------------------------------------------------------------------

def unitary_part(m: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(m)
    return u @ vh


def antihermitian_log(u: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(u)
    phases = np.angle(vals)
    logu = vecs @ np.diag(1j * phases) @ np.linalg.inv(vecs)
    return 0.5 * (logu - logu.conj().T)


def operator_schmidt_values(u: np.ndarray) -> np.ndarray:
    """Operator Schmidt values for a two-qubit gate in the computational basis."""
    reshaped = u.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    return np.linalg.svd(reshaped, compute_uv=False)


def operator_schmidt_rank(u: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.sum(operator_schmidt_values(u) > tol))


def random_qubit_state(rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=2) + 1j * rng.normal(size=2)
    z /= np.linalg.norm(z)
    return z


def linear_entropy_from_state(psi: np.ndarray) -> float:
    rho = np.outer(psi, psi.conj())
    rho_a = np.zeros((2, 2), dtype=complex)
    # partial trace over second qubit
    for a in range(2):
        for ap in range(2):
            for b in range(2):
                rho_a[a, ap] += rho[2 * a + b, 2 * ap + b]
    return float(1.0 - np.real(np.trace(rho_a @ rho_a)))


def entangling_power_mc(
    u: np.ndarray,
    n_samples: int = 512,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        psi = np.kron(random_qubit_state(rng), random_qubit_state(rng))
        out = u @ psi
        values.append(linear_entropy_from_state(out))
    return float(np.mean(values))


def local_antihermitian_basis() -> List[np.ndarray]:
    basis = [1j * kron(I2, I2)]
    basis += [1j * kron(p, I2) for p in (SX, SY, SZ)]
    basis += [1j * kron(I2, p) for p in (SX, SY, SZ)]
    out: List[np.ndarray] = []
    for x in basis:
        y = x.copy()
        for z in out:
            y -= np.trace(z.conj().T @ y) * z
        nrm = math.sqrt(float(np.real(np.trace(y.conj().T @ y))))
        out.append(y / nrm)
    return out


LOCAL_AH_BASIS = local_antihermitian_basis()


def proj_local_lie(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    for b in LOCAL_AH_BASIS:
        coeff = np.trace(b.conj().T @ a)
        out += coeff * b
    return out


def best_local_procrustes(
    u: np.ndarray,
    n_restart: int = 16,
    include_swap: bool = True,
    seed: int = 0,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Approximate min ||U - (A⊗B) P||_F over local unitaries and optional qubit swap.

    The optimization is small enough that alternating polar decompositions with a handful of
    random restarts provide stable fits for the present calculations.
    """
    rng = np.random.default_rng(seed)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    perms = [np.eye(4, dtype=complex), swap] if include_swap else [np.eye(4, dtype=complex)]

    best_dist = np.inf
    best_payload: Dict[str, np.ndarray] = {}

    for p in perms:
        up = u @ p.conj().T
        up4 = up.reshape(2, 2, 2, 2)
        for _ in range(n_restart):
            seed_mat = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            b, _ = polar(seed_mat)
            a = np.eye(2, dtype=complex)
            prev = None
            for _ in range(64):
                m = np.zeros((2, 2), dtype=complex)
                for i in range(2):
                    for j in range(2):
                        # Contract over the second-qubit operator indices.
                        m[i, j] = np.sum(up4[i, :, j, :] * b.conj())
                a, _ = polar(m)

                n = np.zeros((2, 2), dtype=complex)
                for i in range(2):
                    for j in range(2):
                        n[i, j] = np.sum(up4[:, i, :, j] * a.conj())
                b, _ = polar(n)

                v = np.kron(a, b)
                dist = float(np.linalg.norm(up - v, ord="fro"))
                if prev is not None and abs(prev - dist) < 1e-12:
                    break
                prev = dist

            if dist < best_dist:
                best_dist = dist
                best_payload = {
                    "A": a,
                    "B": b,
                    "P": p,
                    "nearest_local_gate": np.kron(a, b) @ p,
                }

    return best_dist, best_payload


# -----------------------------------------------------------------------------
# Generic projector / frame extraction
# -----------------------------------------------------------------------------

def spectral_projector_from_states(v: np.ndarray) -> np.ndarray:
    return v @ v.conj().T


def compress_observables(v: np.ndarray, oa: np.ndarray, ob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return v.conj().T @ oa @ v, v.conj().T @ ob @ v


def split_two_plus_two(evals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(evals)
    evals = evals[order]
    left = order[:2]
    right = order[2:]
    return left, right


def joint_diagonalize(
    v: np.ndarray,
    oa_tilde: np.ndarray,
    ob_tilde: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract a local product frame by sequential spectral splitting.

    First split the quartet into a 2+2 decomposition using the compressed O_A, then split
    each two-dimensional block using the compressed O_B.
    """
    a = 0.5 * (oa_tilde + oa_tilde.conj().T)
    b = 0.5 * (ob_tilde + ob_tilde.conj().T)

    evals_a, vecs_a = np.linalg.eigh(a)
    left_idx, right_idx = split_two_plus_two(evals_a)

    quartet_cols: List[np.ndarray] = []
    labels: List[Tuple[int, int]] = []

    for idxs, side in [(left_idx, -1), (right_idx, +1)]:
        sub = vecs_a[:, idxs]
        b_sub = sub.conj().T @ b @ sub
        evals_b, vecs_b = np.linalg.eigh(0.5 * (b_sub + b_sub.conj().T))
        order_b = np.argsort(evals_b)
        vecs_b = vecs_b[:, order_b]
        for j, spin in enumerate((-1, +1)):
            quartet_cols.append(sub @ vecs_b[:, j])
            labels.append((side, spin))

    q = np.column_stack(quartet_cols)
    f = v @ q
    return f, labels


def fix_column_phases(frame: np.ndarray, refs: Sequence[int]) -> np.ndarray:
    out = frame.copy()
    for col, ref in enumerate(refs):
        amp = out[ref, col]
        if abs(amp) < 1e-14:
            ref = int(np.argmax(np.abs(out[:, col])))
            amp = out[ref, col]
        if abs(amp) > 1e-14:
            out[:, col] *= np.exp(-1j * np.angle(amp))
    return out


def jointness_metric(oa_tilde: np.ndarray, ob_tilde: np.ndarray) -> float:
    a = 0.5 * (oa_tilde + oa_tilde.conj().T)
    b = 0.5 * (ob_tilde + ob_tilde.conj().T)
    da = float(np.ptp(np.linalg.eigvalsh(a)))
    db = float(np.ptp(np.linalg.eigvalsh(b)))
    denom = max(da * db, 1e-14)
    return float(np.linalg.norm(a @ b - b @ a, ord="fro") / denom)


def link_unitary(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    return unitary_part(f0.conj().T @ f1)


def link_entangling_connection(f0: np.ndarray, f1: np.ndarray, dtheta: float) -> float:
    u = link_unitary(f0, f1)
    a = antihermitian_log(u) / dtheta
    return float(np.linalg.norm(a - proj_local_lie(a), ord="fro"))


def berry_holonomy(frames: Sequence[np.ndarray]) -> np.ndarray:
    u = np.eye(4, dtype=complex)
    for i in range(len(frames) - 1):
        u = u @ link_unitary(frames[i], frames[i + 1])
    return u


def holonomy_entangling_metrics(
    u: np.ndarray,
    *,
    n_samples: int = 512,
    seed: int = 0,
    n_restart: int = 16,
) -> Dict[str, object]:
    dist, payload = best_local_procrustes(u, n_restart=n_restart, seed=seed)
    svals = operator_schmidt_values(u)
    return {
        "D_loc": float(dist),
        "operator_schmidt_values": [float(x) for x in svals],
        "operator_schmidt_rank": operator_schmidt_rank(u),
        "entangling_power": entangling_power_mc(u, n_samples=n_samples, seed=seed),
        "best_local_gate": payload["nearest_local_gate"],
    }


# -----------------------------------------------------------------------------
# Spinful SSH / Rice-Mele-like edge quartet benchmark
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SSHConfig:
    N: int = 16
    t1: float = 0.55
    t2: float = 1.0
    delta: float = 0.0
    lam1: float = 0.20
    lam2: float = 0.15
    B: float = 0.30



def ssh_index(n: int, sub: int, spin: int, N: int) -> int:
    return ((n * 2 + sub) * 2 + spin)



def ssh_edge_observable(cfg: SSHConfig) -> np.ndarray:
    dim = 4 * cfg.N
    out = np.zeros((dim, dim), dtype=float)
    xs = np.linspace(-1.0, 1.0, cfg.N)
    for n, x in enumerate(xs):
        for sub in (0, 1):
            for spin in (0, 1):
                i = ssh_index(n, sub, spin, cfg.N)
                out[i, i] = x
    return out



def ssh_spin_observable(cfg: SSHConfig) -> np.ndarray:
    dim = 4 * cfg.N
    out = np.zeros((dim, dim), dtype=complex)
    for n in range(cfg.N):
        for sub in (0, 1):
            for spin in (0, 1):
                i = ssh_index(n, sub, spin, cfg.N)
                out[i, i] = 1.0 if spin == 0 else -1.0
    return out



def ssh_refs(cfg: SSHConfig) -> List[int]:
    # Ordered as [left/down, left/up, right/down, right/up]
    return [
        ssh_index(0, 0, 1, cfg.N),
        ssh_index(0, 0, 0, cfg.N),
        ssh_index(cfg.N - 1, 1, 1, cfg.N),
        ssh_index(cfg.N - 1, 1, 0, cfg.N),
    ]



def build_spinful_ssh(
    cfg: SSHConfig,
    theta_l: float,
    theta_r: float,
    *,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    dim = 4 * cfg.N
    h = np.zeros((dim, dim), dtype=complex)

    for n in range(cfg.N):
        for sub in (0, 1):
            sign = +1.0 if sub == 0 else -1.0
            for spin in (0, 1):
                i = ssh_index(n, sub, spin, cfg.N)
                h[i, i] += sign * cfg.delta
                if disorder:
                    h[i, i] += disorder * (rng.random() - 0.5)

    for n in range(cfg.N):
        # intracell hopping t1 I + i lam1 sigma_y
        block_intra = cfg.t1 * I2 + 1j * cfg.lam1 * SY
        for a in range(2):
            for b in range(2):
                i = ssh_index(n, 0, a, cfg.N)
                j = ssh_index(n, 1, b, cfg.N)
                h[i, j] += block_intra[a, b]
                h[j, i] += np.conjugate(block_intra[a, b])

        if n < cfg.N - 1:
            block_inter = cfg.t2 * I2 + 1j * cfg.lam2 * SX
            for a in range(2):
                for b in range(2):
                    i = ssh_index(n, 1, a, cfg.N)
                    j = ssh_index(n + 1, 0, b, cfg.N)
                    h[i, j] += block_inter[a, b]
                    h[j, i] += np.conjugate(block_inter[a, b])

    z_left = cfg.B * (math.cos(theta_l) * SX + math.sin(theta_l) * SY)
    z_right = cfg.B * (math.cos(theta_r) * SX + math.sin(theta_r) * SY)
    for a in range(2):
        for b in range(2):
            h[ssh_index(0, 0, a, cfg.N), ssh_index(0, 0, b, cfg.N)] += z_left[a, b]
            h[ssh_index(cfg.N - 1, 1, a, cfg.N), ssh_index(cfg.N - 1, 1, b, cfg.N)] += z_right[a, b]

    return 0.5 * (h + h.conj().T)



def ssh_quartet_data(
    cfg: SSHConfig,
    theta_l: float,
    theta_r: float,
    oa: np.ndarray,
    ob: np.ndarray,
    refs: Sequence[int],
    *,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Dict[str, object]:
    h = build_spinful_ssh(cfg, theta_l, theta_r, disorder=disorder, rng=rng)
    evals, evecs = np.linalg.eigh(h)
    idx = np.argsort(np.abs(evals))[:4]
    idx = idx[np.argsort(evals[idx])]
    vals = evals[idx]
    v = evecs[:, idx]
    gap = float(np.sort(np.abs(evals))[4] - np.sort(np.abs(evals))[3])

    oa_tilde, ob_tilde = compress_observables(v, oa, ob)
    frame, labels = joint_diagonalize(v, oa_tilde, ob_tilde)
    frame = fix_column_phases(frame, refs)

    return {
        "energies": [float(x) for x in vals],
        "frame": frame,
        "gap": gap,
        "jointness": jointness_metric(oa_tilde, ob_tilde),
        "oa_eigs": [float(x) for x in np.linalg.eigvalsh(0.5 * (oa_tilde + oa_tilde.conj().T))],
        "ob_eigs": [float(x) for x in np.linalg.eigvalsh(0.5 * (ob_tilde + ob_tilde.conj().T))],
        "labels": labels,
    }


# -----------------------------------------------------------------------------
# BBH corner quartet benchmark
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BBHConfig:
    Nx: int = 5
    Ny: int = 5
    gx: float = 0.30
    gy: float = 0.30
    lx: float = 1.00
    ly: float = 1.00
    delta: float = 0.0
    mx: float = 0.60
    my: float = 0.60
    pattern: str = "cross"



def bbh_cell_index(x: int, y: int, Nx: int, Ny: int) -> int:
    return 4 * (x + Nx * y)



def bbh_pos_observable(cfg: BBHConfig, axis: str) -> np.ndarray:
    dim = 4 * cfg.Nx * cfg.Ny
    out = np.zeros((dim, dim), dtype=float)
    xs = np.linspace(-1.0, 1.0, cfg.Nx)
    ys = np.linspace(-1.0, 1.0, cfg.Ny)
    for y in range(cfg.Ny):
        for x in range(cfg.Nx):
            val = xs[x] if axis == "x" else ys[y]
            sl = slice(bbh_cell_index(x, y, cfg.Nx, cfg.Ny), bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
            out[sl, sl] = val * np.eye(4)
    return out



def bbh_refs(cfg: BBHConfig) -> List[int]:
    return [
        bbh_cell_index(0, 0, cfg.Nx, cfg.Ny) + 1,
        bbh_cell_index(0, cfg.Ny - 1, cfg.Nx, cfg.Ny) + 2,
        bbh_cell_index(cfg.Nx - 1, 0, cfg.Nx, cfg.Ny) + 3,
        bbh_cell_index(cfg.Nx - 1, cfg.Ny - 1, cfg.Nx, cfg.Ny) + 0,
    ]



def build_bbh(
    cfg: BBHConfig,
    theta_x: float,
    theta_y: float,
    *,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    dim = 4 * cfg.Nx * cfg.Ny
    h = np.zeros((dim, dim), dtype=complex)

    onsite = cfg.gx * GAMMA4 + cfg.gy * GAMMA2 + cfg.delta * GAMMA0
    hopx = 0.5 * (cfg.lx * GAMMA4 - 1j * cfg.lx * GAMMA3)
    hopy = 0.5 * (cfg.ly * GAMMA2 - 1j * cfg.ly * GAMMA1)
    bx = math.cos(theta_x) * BBH_M1 + math.sin(theta_x) * BBH_M2
    by = math.cos(theta_y) * BBH_M3 + math.sin(theta_y) * BBH_M4

    for y in range(cfg.Ny):
        y_sign = -1 if y == 0 else (+1 if y == cfg.Ny - 1 else 0)
        for x in range(cfg.Nx):
            x_sign = -1 if x == 0 else (+1 if x == cfg.Nx - 1 else 0)
            sl = slice(bbh_cell_index(x, y, cfg.Nx, cfg.Ny), bbh_cell_index(x, y, cfg.Nx, cfg.Ny) + 4)
            h[sl, sl] += onsite
            if disorder:
                h[sl, sl] += disorder * (rng.random() - 0.5) * np.eye(4)

            if x_sign != 0 and y_sign != 0:
                if cfg.pattern == "cross":
                    h[sl, sl] += cfg.mx * (x_sign * y_sign) * bx
                    h[sl, sl] += cfg.my * (x_sign * y_sign) * by
                elif cfg.pattern == "separable":
                    h[sl, sl] += cfg.mx * x_sign * bx
                    h[sl, sl] += cfg.my * y_sign * by
                else:
                    h[sl, sl] += cfg.mx * x_sign * bx
                    h[sl, sl] += cfg.my * (x_sign * y_sign) * by

            if x < cfg.Nx - 1:
                sl2 = slice(bbh_cell_index(x + 1, y, cfg.Nx, cfg.Ny), bbh_cell_index(x + 1, y, cfg.Nx, cfg.Ny) + 4)
                h[sl, sl2] += hopx
                h[sl2, sl] += hopx.conj().T
            if y < cfg.Ny - 1:
                sl2 = slice(bbh_cell_index(x, y + 1, cfg.Nx, cfg.Ny), bbh_cell_index(x, y + 1, cfg.Nx, cfg.Ny) + 4)
                h[sl, sl2] += hopy
                h[sl2, sl] += hopy.conj().T

    return 0.5 * (h + h.conj().T)



def bbh_quartet_data(
    cfg: BBHConfig,
    theta_x: float,
    theta_y: float,
    oa: np.ndarray,
    ob: np.ndarray,
    refs: Sequence[int],
    *,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Dict[str, object]:
    h = build_bbh(cfg, theta_x, theta_y, disorder=disorder, rng=rng)
    evals, evecs = np.linalg.eigh(h)
    idx = np.argsort(np.abs(evals))[:4]
    idx = idx[np.argsort(np.abs(evals[idx]))]
    vals = evals[idx]
    v = evecs[:, idx]
    abs_sorted = np.sort(np.abs(evals))
    gap = float(abs_sorted[4] - abs_sorted[3])

    oa_tilde, ob_tilde = compress_observables(v, oa, ob)
    frame, labels = joint_diagonalize(v, oa_tilde, ob_tilde)
    frame = fix_column_phases(frame, refs)

    return {
        "energies": [float(x) for x in vals],
        "frame": frame,
        "gap": gap,
        "jointness": jointness_metric(oa_tilde, ob_tilde),
        "oa_eigs": [float(x) for x in np.linalg.eigvalsh(0.5 * (oa_tilde + oa_tilde.conj().T))],
        "ob_eigs": [float(x) for x in np.linalg.eigvalsh(0.5 * (ob_tilde + ob_tilde.conj().T))],
        "labels": labels,
    }


# -----------------------------------------------------------------------------
# Generic torus-grid analysis
# -----------------------------------------------------------------------------

def fundamental_grid(
    n_grid: int,
) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=True)



def analyze_grid(
    thetas_a: np.ndarray,
    thetas_b: np.ndarray,
    data_fn: Callable[[float, float], Dict[str, object]],
) -> Dict[str, object]:
    n_a = len(thetas_a)
    n_b = len(thetas_b)
    frames: List[List[np.ndarray]] = [[None for _ in range(n_b)] for _ in range(n_a)]  # type: ignore
    gaps = np.zeros((n_a, n_b), dtype=float)
    joint = np.zeros((n_a, n_b), dtype=float)

    for i, ta in enumerate(thetas_a):
        for j, tb in enumerate(thetas_b):
            datum = data_fn(float(ta), float(tb))
            frames[i][j] = datum["frame"]
            gaps[i, j] = float(datum["gap"])
            joint[i, j] = float(datum["jointness"])

    d_a = float(thetas_a[1] - thetas_a[0])
    d_b = float(thetas_b[1] - thetas_b[0])
    nloc_a = np.zeros((n_a - 1, n_b), dtype=float)
    nloc_b = np.zeros((n_a, n_b - 1), dtype=float)

    for i in range(n_a - 1):
        for j in range(n_b):
            nloc_a[i, j] = link_entangling_connection(frames[i][j], frames[i + 1][j], d_a)
    for i in range(n_a):
        for j in range(n_b - 1):
            nloc_b[i, j] = link_entangling_connection(frames[i][j], frames[i][j + 1], d_b)

    nloc = np.zeros((n_a, n_b), dtype=float)
    nloc[:-1, :] += nloc_a
    nloc[1:, :] += nloc_a
    nloc[:, :-1] += nloc_b
    nloc[:, 1:] += nloc_b
    counts = np.zeros((n_a, n_b), dtype=float)
    counts[:-1, :] += 1.0
    counts[1:, :] += 1.0
    counts[:, :-1] += 1.0
    counts[:, 1:] += 1.0
    nloc /= np.maximum(counts, 1.0)

    return {
        "theta_a": thetas_a,
        "theta_b": thetas_b,
        "frames": frames,
        "gap_map": gaps,
        "joint_map": joint,
        "nloc_map": nloc,
        "nloc_a": nloc_a,
        "nloc_b": nloc_b,
    }



def path_holonomy_metrics(
    frames: Sequence[np.ndarray],
    *,
    seed: int = 0,
    n_restart: int = 16,
    n_samples: int = 512,
) -> Dict[str, object]:
    u = berry_holonomy(frames)
    metrics = holonomy_entangling_metrics(u, seed=seed, n_restart=n_restart, n_samples=n_samples)
    metrics["U"] = u
    return metrics


# -----------------------------------------------------------------------------
# Convenience wrappers for the two benchmarks
# -----------------------------------------------------------------------------

def ssh_benchmark(
    cfg: SSHConfig,
    *,
    n_grid: int = 17,
    n_loop: int = 41,
) -> Dict[str, object]:
    oa = ssh_edge_observable(cfg)
    ob = ssh_spin_observable(cfg)
    refs = ssh_refs(cfg)

    def datum(thl: float, thr: float) -> Dict[str, object]:
        return ssh_quartet_data(cfg, thl, thr, oa, ob, refs)

    theta = fundamental_grid(n_grid)
    grid = analyze_grid(theta, theta, datum)

    loop_theta = fundamental_grid(n_loop)
    frames_left = [datum(float(t), 0.0)["frame"] for t in loop_theta]
    frames_right = [datum(0.0, float(t))["frame"] for t in loop_theta]
    frames_diag = [datum(float(t), float(t))["frame"] for t in loop_theta]

    return {
        "config": asdict(cfg),
        "grid": grid,
        "loops": {
            "left_cycle": path_holonomy_metrics(frames_left, seed=11),
            "right_cycle": path_holonomy_metrics(frames_right, seed=12),
            "diag_cycle": path_holonomy_metrics(frames_diag, seed=13),
        },
    }



def bbh_benchmark(
    cfg: BBHConfig,
    *,
    n_grid: int = 13,
    n_loop: int = 25,
) -> Dict[str, object]:
    oa = bbh_pos_observable(cfg, "x")
    ob = bbh_pos_observable(cfg, "y")
    refs = bbh_refs(cfg)

    def datum(thx: float, thy: float) -> Dict[str, object]:
        return bbh_quartet_data(cfg, thx, thy, oa, ob, refs)

    theta = fundamental_grid(n_grid)
    grid = analyze_grid(theta, theta, datum)

    loop_theta = fundamental_grid(n_loop)
    frames_x = [datum(float(t), 0.0)["frame"] for t in loop_theta]
    frames_y = [datum(0.0, float(t))["frame"] for t in loop_theta]
    frames_diag = [datum(float(t), float(t))["frame"] for t in loop_theta]

    return {
        "config": asdict(cfg),
        "grid": grid,
        "loops": {
            "x_cycle": path_holonomy_metrics(frames_x, seed=21),
            "y_cycle": path_holonomy_metrics(frames_y, seed=22),
            "diag_cycle": path_holonomy_metrics(frames_diag, seed=23),
        },
    }


# -----------------------------------------------------------------------------
# Scaling studies
# -----------------------------------------------------------------------------

def ssh_size_scaling(
    sizes: Sequence[int],
    *,
    n_loop: int = 31,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    ts = fundamental_grid(n_loop)
    for N in sizes:
        cfg = SSHConfig(N=N)
        oa = ssh_edge_observable(cfg)
        ob = ssh_spin_observable(cfg)
        refs = ssh_refs(cfg)
        frames = [ssh_quartet_data(cfg, float(t), 0.0, oa, ob, refs)["frame"] for t in ts]
        metrics = path_holonomy_metrics(frames, seed=100 + N, n_restart=12, n_samples=256)
        gaps = [ssh_quartet_data(cfg, float(t), 0.0, oa, ob, refs)["gap"] for t in np.linspace(0.0, 2.0 * math.pi, 9)]
        out.append(
            {
                "N": N,
                "D_loc": metrics["D_loc"],
                "entangling_power": metrics["entangling_power"],
                "operator_schmidt_rank": metrics["operator_schmidt_rank"],
                "gap_min": float(np.min(gaps)),
            }
        )
    return out



def bbh_disorder_scaling(
    cfg: BBHConfig,
    strengths: Sequence[float],
    *,
    n_realizations: int = 4,
    n_loop: int = 21,
) -> List[Dict[str, object]]:
    ts = fundamental_grid(n_loop)
    oa = bbh_pos_observable(cfg, "x")
    ob = bbh_pos_observable(cfg, "y")
    refs = bbh_refs(cfg)
    out: List[Dict[str, object]] = []

    for w in strengths:
        rows = []
        for r in range(n_realizations):
            rng = np.random.default_rng(7000 + 97 * r + int(1000 * w))

            def datum(th: float) -> Dict[str, object]:
                return bbh_quartet_data(cfg, th, th, oa, ob, refs, disorder=w, rng=rng)

            frames = [datum(float(t))["frame"] for t in ts]
            metrics = path_holonomy_metrics(frames, seed=300 + r + int(100 * w), n_restart=10, n_samples=192)
            gaps = [datum(float(t))["gap"] for t in np.linspace(0.0, 2.0 * math.pi, 7)]
            rows.append(
                {
                    "D_loc": metrics["D_loc"],
                    "entangling_power": metrics["entangling_power"],
                    "operator_schmidt_rank": metrics["operator_schmidt_rank"],
                    "gap_min": float(np.min(gaps)),
                }
            )
        out.append(
            {
                "W": float(w),
                "D_loc_mean": float(np.mean([r["D_loc"] for r in rows])),
                "D_loc_std": float(np.std([r["D_loc"] for r in rows])),
                "entangling_power_mean": float(np.mean([r["entangling_power"] for r in rows])),
                "entangling_power_std": float(np.std([r["entangling_power"] for r in rows])),
                "gap_min_mean": float(np.mean([r["gap_min"] for r in rows])),
                "osr_mean": float(np.mean([r["operator_schmidt_rank"] for r in rows])),
            }
        )
    return out


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist(),
            }
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ndarray_to_list(v) for v in obj]
    if isinstance(obj, tuple):
        return [ndarray_to_list(v) for v in obj]
    return obj


