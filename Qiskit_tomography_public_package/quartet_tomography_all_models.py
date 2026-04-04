#!/usr/bin/env python3
"""Tomography suite for effective extracted-quartet models in BHZ, SSH, and BBH.

This module implements two-qubit state tomography and optional process tomography
for the representative effective loops used in the paper:

    - BHZ: CT, CB, C_plus, C_minus
    - SSH: CL, CR, Cdiag, Canti
    - BBH: Cx, Cy, Cdiag, Canti

Supported execution modes:

    - analytic_reference : NumPy reference evaluation
    - local_aer          : Qiskit Aer / Runtime local mode
    - ibm_* backend name : IBM Quantum hardware via SamplerV2

Running ``run_suite`` writes raw-count data, tomography summaries, overview tables,
and figure files to the selected output directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import expm, polar, sqrtm
from scipy.optimize import minimize


# --------------------------------------------------------------------------------------
# Type aliases and linear-algebra helpers
# --------------------------------------------------------------------------------------

ComplexMatrix = NDArray[np.complex128]
RealArray = NDArray[np.float64]

I2: ComplexMatrix = np.eye(2, dtype=complex)
X2: ComplexMatrix = np.array([[0, 1], [1, 0]], dtype=complex)
Y2: ComplexMatrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2: ComplexMatrix = np.array([[1, 0], [0, -1]], dtype=complex)
P0: ComplexMatrix = np.array([[1, 0], [0, 0]], dtype=complex)
P1: ComplexMatrix = np.array([[0, 0], [0, 1]], dtype=complex)

PAULI_1Q: Mapping[str, ComplexMatrix] = {
    "I": I2,
    "X": X2,
    "Y": Y2,
    "Z": Z2,
}

SINGLE_QUBIT_STATE_LABELS: Tuple[str, ...] = ("0", "1", "+", "+i")
TOMO_BASES: Tuple[str, ...] = ("X", "Y", "Z")
SELECTED_OBSERVABLES: Tuple[str, ...] = ("XI", "IX", "IY", "ZY", "YZ", "XX")
DEFAULT_MODEL_ORDER: Tuple[str, ...] = ("BHZ", "SSH", "BBH")


def normalize_real_vector(vec: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-15:
        raise ValueError("Cannot normalize a zero vector.")
    return arr / norm


def rz_matrix(theta: float) -> ComplexMatrix:
    return np.array(
        [[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]],
        dtype=complex,
    )


def ry_matrix(theta: float) -> ComplexMatrix:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def u2_from_euler(params: Sequence[float]) -> ComplexMatrix:
    """U(2) parameterization: e^{iα} Rz(β) Ry(γ) Rz(δ)."""
    alpha, beta, gamma, delta = params
    return np.exp(1j * alpha) * rz_matrix(beta) @ ry_matrix(gamma) @ rz_matrix(delta)


def vec(matrix: ComplexMatrix) -> ComplexMatrix:
    """Column-stacking vectorization."""
    return matrix.reshape((-1, 1), order="F")


def unvec(vector: ComplexMatrix, dim: int) -> ComplexMatrix:
    return vector.reshape((dim, dim), order="F")


def project_to_density_matrix(rho: ComplexMatrix) -> ComplexMatrix:
    rho_h = 0.5 * (rho + rho.conj().T)
    evals, evecs = np.linalg.eigh(rho_h)
    evals = np.clip(np.real(evals), 0.0, None)
    if float(evals.sum()) <= 1e-14:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    rho_proj = evecs @ np.diag(evals / evals.sum()) @ evecs.conj().T
    return 0.5 * (rho_proj + rho_proj.conj().T)


def state_fidelity(rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
    sqrt_rho = sqrtm(project_to_density_matrix(rho))
    inner = sqrt_rho @ project_to_density_matrix(sigma) @ sqrt_rho
    value = np.real(np.trace(sqrtm(0.5 * (inner + inner.conj().T))))
    return float(max(0.0, min(1.0, value**2)))


def concurrence(rho: ComplexMatrix) -> float:
    yy = np.kron(Y2, Y2)
    r = rho @ yy @ rho.conj() @ yy
    eigvals = np.linalg.eigvals(r)
    eigvals = np.sort(np.maximum(np.real(eigvals), 0.0))[::-1]
    lambdas = np.sqrt(eigvals)
    return max(0.0, float(lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))


def nearest_unitary(matrix: ComplexMatrix) -> ComplexMatrix:
    u, _ = polar(matrix)
    return u


def operator_schmidt_singular_values(unitary_4x4: ComplexMatrix) -> List[float]:
    reshaped = unitary_4x4.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    svals = np.linalg.svd(reshaped, compute_uv=False)
    return [float(np.real(x)) for x in svals]


def sorted_eigenphases(unitary_4x4: ComplexMatrix) -> List[float]:
    vals = np.linalg.eigvals(unitary_4x4)
    phases = np.sort(np.angle(vals))
    return [float(x) for x in phases]


def average_gate_overlap(u_ideal: ComplexMatrix, u_est: ComplexMatrix) -> float:
    dim = u_ideal.shape[0]
    overlap = abs(np.trace(u_ideal.conj().T @ u_est)) / dim
    return float(max(0.0, min(1.0, overlap)))


# --------------------------------------------------------------------------------------
# Effective model unitary builders
# --------------------------------------------------------------------------------------


def pauli_from_axis(axis: Sequence[float]) -> ComplexMatrix:
    nx, ny, nz = normalize_real_vector(axis)
    return (nx * X2 + ny * Y2 + nz * Z2).astype(complex)


def local_rotation(qubit: int, angle: float, axis: Sequence[float]) -> ComplexMatrix:
    h = pauli_from_axis(axis)
    u1 = expm(-1j * angle * h)
    if qubit == 0:
        return np.kron(u1, I2)
    if qubit == 1:
        return np.kron(I2, u1)
    raise ValueError("Qubit index must be 0 or 1.")


def controlled_rotation(angle: float, axis: Sequence[float], control_state: int) -> ComplexMatrix:
    h = pauli_from_axis(axis)
    u1 = expm(-1j * angle * h)
    if control_state == 0:
        return np.kron(P0, u1) + np.kron(P1, I2)
    if control_state == 1:
        return np.kron(P0, I2) + np.kron(P1, u1)
    raise ValueError("control_state must be 0 or 1.")


def zz_entangler(angle: float) -> ComplexMatrix:
    return expm(-1j * angle * np.kron(Z2, Z2))


def z_tensor_axis_entangler(angle: float, axis: Sequence[float]) -> ComplexMatrix:
    h = np.kron(Z2, pauli_from_axis(axis))
    return expm(-1j * angle * h)


def canonical_xy_entangler(xx_angle: float, yy_angle: float) -> ComplexMatrix:
    return expm(-1j * (xx_angle * np.kron(X2, X2) + yy_angle * np.kron(Y2, Y2)))


def bbh_axis_loop(qubit: int, local_angle: float, residual_angle: float, residual_pauli: str) -> ComplexMatrix:
    if residual_pauli == "XX":
        residual = expm(-1j * residual_angle * np.kron(X2, X2))
    elif residual_pauli == "YY":
        residual = expm(-1j * residual_angle * np.kron(Y2, Y2))
    else:
        raise ValueError("residual_pauli must be 'XX' or 'YY'.")
    return local_rotation(qubit=qubit, angle=local_angle, axis=(0.0, 0.0, 1.0)) @ residual


SSH_AXIS = normalize_real_vector((0.32, 0.16, -0.93))


@dataclass(frozen=True)
class LoopDefinition:
    model: str
    name: str
    description: str
    unitary_builder: Callable[[float], ComplexMatrix]
    paper_dloc: Optional[float] = None
    paper_note: str = ""


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    qubit_labels: Tuple[str, str]
    loops: Mapping[str, LoopDefinition]
    note: str = ""


# Representative effective gates.
# The parameters are chosen to respect the manuscript's effective descriptions and,
# for the BBH mixed cycle, to match the operator-Schmidt hierarchy of Table 2.


def build_model_catalog() -> Dict[str, ModelDefinition]:
    bhz_loops = {
        "CT": LoopDefinition(
            model="BHZ",
            name="CT",
            description="Single top-edge loop: controlled rotation on the helical qubit.",
            unitary_builder=lambda s: controlled_rotation(angle=0.18 * s, axis=(0.0, 0.0, 1.0), control_state=0),
            paper_dloc=0.18,
            paper_note="Eq. (2.4): K_T ≈ φ P_T ⊗ Z_h with φ = 0.18.",
        ),
        "CB": LoopDefinition(
            model="BHZ",
            name="CB",
            description="Single bottom-edge loop: controlled rotation on the helical qubit.",
            unitary_builder=lambda s: controlled_rotation(angle=0.18 * s, axis=(0.0, 0.0, 1.0), control_state=1),
            paper_dloc=0.18,
            paper_note="By symmetry with CT on the kx = 0 slice.",
        ),
        "C_plus": LoopDefinition(
            model="BHZ",
            name="C_plus",
            description="Co-rotating BHZ loop: almost local I ⊗ Z_h response.",
            unitary_builder=lambda s: local_rotation(qubit=1, angle=0.18 * s, axis=(0.0, 0.0, 1.0)),
            paper_dloc=0.01,
            paper_note="Eq. (2.4): K_+ ≈ φ I ⊗ Z_h.",
        ),
        "C_minus": LoopDefinition(
            model="BHZ",
            name="C_minus",
            description="Counter-rotating BHZ loop: Ising-like Z_edge ⊗ Z_h entangler.",
            unitary_builder=lambda s: zz_entangler(angle=0.18 * s),
            paper_dloc=0.37,
            paper_note="Eq. (2.4): K_- ≈ φ Z_edge ⊗ Z_h.",
        ),
    }

    ssh_loops = {
        "CL": LoopDefinition(
            model="SSH",
            name="CL",
            description="Single left-edge loop: controlled rotation on the spin qubit.",
            unitary_builder=lambda s: controlled_rotation(angle=0.20 * s, axis=SSH_AXIS, control_state=0),
            paper_dloc=0.20,
            paper_note="Eq. (3.6): K_L ≈ φ P_L ⊗ (n·σ), φ = 0.20.",
        ),
        "CR": LoopDefinition(
            model="SSH",
            name="CR",
            description="Single right-edge loop: controlled rotation on the spin qubit.",
            unitary_builder=lambda s: controlled_rotation(angle=0.20 * s, axis=SSH_AXIS, control_state=1),
            paper_dloc=0.20,
            paper_note="Symmetry partner of CL.",
        ),
        "Cdiag": LoopDefinition(
            model="SSH",
            name="Cdiag",
            description="Diagonal SSH loop: smaller nonlocal response on the same one-parameter edge.",
            unitary_builder=lambda s: local_rotation(qubit=1, angle=0.13 * s, axis=SSH_AXIS)
            @ z_tensor_axis_entangler(angle=0.07 * s, axis=SSH_AXIS),
            paper_dloc=0.14,
            paper_note="Representative smaller nonlocal loop in the SSH family.",
        ),
        "Canti": LoopDefinition(
            model="SSH",
            name="Canti",
            description="Anti-diagonal SSH loop: larger rank-two entangler on the same edge of the Weyl chamber.",
            unitary_builder=lambda s: z_tensor_axis_entangler(angle=0.19 * s, axis=SSH_AXIS),
            paper_dloc=0.38,
            paper_note="Table 3 / Table 2 hierarchy with D_loc ≈ 0.38.",
        ),
    }

    bbh_loops = {
        "Cx": LoopDefinition(
            model="BBH",
            name="Cx",
            description="BBH x-axis loop: almost local, dominated by a single coarse-side rotation.",
            unitary_builder=lambda s: bbh_axis_loop(qubit=0, local_angle=0.80 * s, residual_angle=0.005 * s, residual_pauli="YY"),
            paper_dloc=0.01,
            paper_note="Axis loop remains nearly local.",
        ),
        "Cy": LoopDefinition(
            model="BBH",
            name="Cy",
            description="BBH y-axis loop: almost local, with large local phase and tiny residual nonlocality.",
            unitary_builder=lambda s: bbh_axis_loop(qubit=1, local_angle=0.83 * s, residual_angle=0.004 * s, residual_pauli="XX"),
            paper_dloc=0.008,
            paper_note="Table 1 reports large eigenphases but D_loc ≈ 0.008.",
        ),
        "Cdiag": LoopDefinition(
            model="BBH",
            name="Cdiag",
            description="BBH diagonal mixed cycle: finite second nonlocal coordinate and four visible Schmidt channels.",
            unitary_builder=lambda s: canonical_xy_entangler(xx_angle=0.07 * s, yy_angle=0.03 * s),
            paper_dloc=0.15,
            paper_note="Matches the Table 2 operator-Schmidt hierarchy of the mixed cycle.",
        ),
        "Canti": LoopDefinition(
            model="BBH",
            name="Canti",
            description="BBH anti-diagonal mixed cycle: symmetry-related mixed entangler.",
            unitary_builder=lambda s: expm(-1j * (0.07 * s * np.kron(X2, X2) - 0.03 * s * np.kron(Y2, Y2))),
            paper_dloc=0.15,
            paper_note="Symmetry-related mixed family with the same qualitative D_loc.",
        ),
    }

    return {
        "BHZ": ModelDefinition(
            name="BHZ",
            qubit_labels=("edge", "helical pseudospin"),
            loops=bhz_loops,
            note="BHZ ribbon effective quartet on the extracted edge ⊗ helical basis.",
        ),
        "SSH": ModelDefinition(
            name="SSH",
            qubit_labels=("edge", "spin"),
            loops=ssh_loops,
            note="Spinful SSH edge quartet on the extracted edge ⊗ spin basis.",
        ),
        "BBH": ModelDefinition(
            name="BBH",
            qubit_labels=("x-side", "y-side"),
            loops=bbh_loops,
            note="BBH corner quartet on the extracted x-side ⊗ y-side basis.",
        ),
    }


MODEL_CATALOG: Dict[str, ModelDefinition] = build_model_catalog()


# --------------------------------------------------------------------------------------
# Qiskit availability and backend helpers
# --------------------------------------------------------------------------------------


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        import qiskit_ibm_runtime  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# State preparation, basis rotations, and reference backend
# --------------------------------------------------------------------------------------


def single_qubit_density(label: str) -> ComplexMatrix:
    if label == "0":
        return np.array([[1, 0], [0, 0]], dtype=complex)
    if label == "1":
        return np.array([[0, 0], [0, 1]], dtype=complex)
    if label == "+":
        return 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    if label == "+i":
        return 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
    raise ValueError(f"Unsupported state label: {label}")


def single_qubit_statevector(label: str) -> NDArray[np.complex128]:
    if label == "0":
        return np.array([1.0, 0.0], dtype=complex)
    if label == "1":
        return np.array([0.0, 1.0], dtype=complex)
    if label == "+":
        return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    if label == "+i":
        return np.array([1.0, 1.0j], dtype=complex) / math.sqrt(2.0)
    raise ValueError(f"Unsupported state label: {label}")


def two_qubit_input_density(label_a: str, label_b: str) -> ComplexMatrix:
    return np.kron(single_qubit_density(label_a), single_qubit_density(label_b))


def two_qubit_input_statevector(label_a: str, label_b: str) -> NDArray[np.complex128]:
    return np.kron(single_qubit_statevector(label_a), single_qubit_statevector(label_b))


def measurement_rotation_matrix(basis: str) -> ComplexMatrix:
    if basis == "Z":
        return I2.copy()
    if basis == "X":
        return (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
    if basis == "Y":
        s_dg = np.array([[1, 0], [0, -1j]], dtype=complex)
        h = (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
        return h @ s_dg
    raise ValueError(f"Unsupported measurement basis: {basis}")


def basis_index_to_qiskit_key(index: int) -> str:
    """Map |q0 q1> basis index to Qiskit count-string ordering b1 b0 = q1 q0."""
    q0 = (index >> 1) & 1
    q1 = index & 1
    return f"{q1}{q0}"


# --------------------------------------------------------------------------------------
# Circuit specs and execution config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitSpec:
    model: str
    loop: str
    input_a: str
    input_b: str
    meas_a: str
    meas_b: str

    @property
    def input_key(self) -> Tuple[str, str]:
        return (self.input_a, self.input_b)

    @property
    def measurement_key(self) -> Tuple[str, str]:
        return (self.meas_a, self.meas_b)

    @property
    def name(self) -> str:
        return (
            f"{self.model}__{self.loop}__prep_{self.input_a}{self.input_b}"
            f"__meas_{self.meas_a}{self.meas_b}"
        )


@dataclass
class ExecutionConfig:
    backend_name: str = "analytic_reference"
    models: Tuple[str, ...] = DEFAULT_MODEL_ORDER
    shots: int = 4096
    optimization_level: int = 1
    seed_transpiler: int = 1234
    seed_reference: int = 1234
    output_dir: str = "quartet_tomography_outputs"
    with_process_tomography: bool = True
    scan_points: int = 201
    token: Optional[str] = None
    instance: Optional[str] = None
    channel: str = "ibm_quantum_platform"
    save_raw_counts: bool = True
    selected_loops: Optional[Mapping[str, Sequence[str]]] = None


# --------------------------------------------------------------------------------------
# Model accessors
# --------------------------------------------------------------------------------------


def resolve_models(config: ExecutionConfig) -> Tuple[str, ...]:
    unknown = [m for m in config.models if m not in MODEL_CATALOG]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return tuple(config.models)


def resolve_loops(model: str, config: ExecutionConfig) -> Tuple[str, ...]:
    available = MODEL_CATALOG[model].loops
    if config.selected_loops is None or model not in config.selected_loops:
        return tuple(available.keys())
    chosen = tuple(config.selected_loops[model])
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise ValueError(f"Unknown loops for {model}: {unknown}")
    return chosen


def get_effective_unitary(model: str, loop: str, scale: float = 1.0) -> ComplexMatrix:
    return MODEL_CATALOG[model].loops[loop].unitary_builder(float(scale)).astype(complex)


def ideal_output_density(model: str, loop: str, input_a: str = "+", input_b: str = "+") -> ComplexMatrix:
    psi_in = two_qubit_input_statevector(input_a, input_b)
    psi_out = get_effective_unitary(model, loop) @ psi_in
    return np.outer(psi_out, psi_out.conj())


# --------------------------------------------------------------------------------------
# Qiskit circuit construction
# --------------------------------------------------------------------------------------


def apply_single_qubit_state_prep(qc: Any, qubit: int, label: str) -> None:
    if label == "0":
        return
    if label == "1":
        qc.x(qubit)
        return
    if label == "+":
        qc.h(qubit)
        return
    if label == "+i":
        qc.h(qubit)
        qc.s(qubit)
        return
    raise ValueError(f"Unsupported single-qubit prep label: {label}")


def apply_single_qubit_measurement_basis_rotation(qc: Any, qubit: int, basis: str) -> None:
    if basis == "Z":
        return
    if basis == "X":
        qc.h(qubit)
        return
    if basis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
        return
    raise ValueError(f"Unsupported measurement basis: {basis}")


def build_circuit(spec: CircuitSpec) -> Any:
    from qiskit import ClassicalRegister, QuantumCircuit

    creg = ClassicalRegister(2, "meas")
    qc = QuantumCircuit(2, creg, name=spec.name)
    apply_single_qubit_state_prep(qc, 0, spec.input_a)
    apply_single_qubit_state_prep(qc, 1, spec.input_b)

    unitary = get_effective_unitary(spec.model, spec.loop)
    qc.unitary(unitary, [0, 1], label=f"{spec.model}_{spec.loop}")

    apply_single_qubit_measurement_basis_rotation(qc, 0, spec.meas_a)
    apply_single_qubit_measurement_basis_rotation(qc, 1, spec.meas_b)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    return qc


# --------------------------------------------------------------------------------------
# Spec builders
# --------------------------------------------------------------------------------------


def build_witness_specs(config: ExecutionConfig) -> List[CircuitSpec]:
    specs: List[CircuitSpec] = []
    for model in resolve_models(config):
        for loop in resolve_loops(model, config):
            for b0 in TOMO_BASES:
                for b1 in TOMO_BASES:
                    specs.append(
                        CircuitSpec(
                            model=model,
                            loop=loop,
                            input_a="+",
                            input_b="+",
                            meas_a=b0,
                            meas_b=b1,
                        )
                    )
    return specs


def build_process_tomography_specs(config: ExecutionConfig) -> List[CircuitSpec]:
    specs: List[CircuitSpec] = []
    for model in resolve_models(config):
        for loop in resolve_loops(model, config):
            for in0 in SINGLE_QUBIT_STATE_LABELS:
                for in1 in SINGLE_QUBIT_STATE_LABELS:
                    for b0 in TOMO_BASES:
                        for b1 in TOMO_BASES:
                            specs.append(
                                CircuitSpec(
                                    model=model,
                                    loop=loop,
                                    input_a=in0,
                                    input_b=in1,
                                    meas_a=b0,
                                    meas_b=b1,
                                )
                            )
    return specs


# --------------------------------------------------------------------------------------
# Execution backends
# --------------------------------------------------------------------------------------


def build_backend_and_sampler(config: ExecutionConfig) -> Tuple[Any, Any]:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except Exception as exc:  # pragma: no cover - import-time environment issue
        raise ImportError(
            "qiskit-ibm-runtime is required for local_aer or IBM hardware backends. "
            "Install qiskit, qiskit-aer, and qiskit-ibm-runtime."
        ) from exc

    if config.backend_name == "local_aer":
        try:
            from qiskit_aer import AerSimulator
        except Exception as exc:  # pragma: no cover
            raise ImportError("qiskit-aer is required for local_aer mode.") from exc
        backend = AerSimulator(method="automatic")
        sampler = Sampler(mode=backend)
        return backend, sampler

    service_kwargs: Dict[str, Any] = {}
    if config.channel:
        service_kwargs["channel"] = config.channel
    if config.token:
        service_kwargs["token"] = config.token
    if config.instance:
        service_kwargs["instance"] = config.instance

    service = QiskitRuntimeService(**service_kwargs)
    backend = service.backend(config.backend_name)
    sampler = Sampler(mode=backend)
    return backend, sampler


def transpile_circuits(circuits: Sequence[Any], backend: Any, optimization_level: int, seed_transpiler: int) -> List[Any]:
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    return [pm.run(circuit) for circuit in circuits]


def run_sampler_counts(circuits: Sequence[Any], sampler: Any, shots: int) -> List[Dict[str, int]]:
    job = sampler.run(list(circuits), shots=shots)
    result = job.result()
    counts_list: List[Dict[str, int]] = []
    for idx, circ in enumerate(circuits):
        creg_name = circ.cregs[0].name
        pub_result = result[idx]
        bit_array = getattr(pub_result.data, creg_name)
        counts = dict(bit_array.get_counts())
        counts_list.append({str(k): int(v) for k, v in counts.items()})
    return counts_list


def reference_probabilities(spec: CircuitSpec) -> NDArray[np.float64]:
    psi = two_qubit_input_statevector(spec.input_a, spec.input_b)
    psi = get_effective_unitary(spec.model, spec.loop) @ psi
    rotation = np.kron(measurement_rotation_matrix(spec.meas_a), measurement_rotation_matrix(spec.meas_b))
    psi = rotation @ psi
    probs = np.abs(psi) ** 2
    probs = np.real_if_close(probs).astype(float)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return probs


def run_reference_counts(specs: Sequence[CircuitSpec], shots: int, seed: int) -> List[Dict[str, int]]:
    rng = np.random.default_rng(seed)
    counts_list: List[Dict[str, int]] = []
    for idx, spec in enumerate(specs):
        probs = reference_probabilities(spec)
        sampled = rng.multinomial(shots, probs)
        counts: Dict[str, int] = {}
        for basis_index, count in enumerate(sampled):
            if int(count) > 0:
                counts[basis_index_to_qiskit_key(basis_index)] = int(count)
        counts_list.append(counts)
    return counts_list


# --------------------------------------------------------------------------------------
# Tomography reconstruction
# --------------------------------------------------------------------------------------


def counts_to_z_expectations(counts: Mapping[str, int]) -> Tuple[float, float, float]:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        raise ValueError("Counts dictionary is empty.")

    exp_z0 = 0.0
    exp_z1 = 0.0
    exp_zz = 0.0
    for key, value in counts.items():
        bits = key.replace(" ", "")
        if len(bits) != 2:
            raise ValueError(f"Expected 2-bit outcome, got {key!r}.")
        q0 = int(bits[-1])
        q1 = int(bits[-2])
        z0 = 1.0 if q0 == 0 else -1.0
        z1 = 1.0 if q1 == 0 else -1.0
        p = float(value) / total
        exp_z0 += z0 * p
        exp_z1 += z1 * p
        exp_zz += z0 * z1 * p
    return exp_z0, exp_z1, exp_zz


def reconstruct_pauli_expectations(setting_counts: Mapping[Tuple[str, str], Mapping[str, int]]) -> Dict[str, float]:
    accum: Dict[str, List[float]] = {a + b: [] for a in PAULI_1Q for b in PAULI_1Q}
    accum["II"] = [1.0]

    for (basis_q0, basis_q1), counts in setting_counts.items():
        exp_q0, exp_q1, exp_q0q1 = counts_to_z_expectations(counts)
        accum[basis_q0 + "I"].append(exp_q0)
        accum["I" + basis_q1].append(exp_q1)
        accum[basis_q0 + basis_q1].append(exp_q0q1)

    reconstructed: Dict[str, float] = {}
    for obs, values in accum.items():
        reconstructed[obs] = float(np.mean(values)) if values else float("nan")
    return reconstructed


def density_from_pauli_expectations(pauli_expectations: Mapping[str, float]) -> ComplexMatrix:
    rho = np.zeros((4, 4), dtype=complex)
    for a, pa in PAULI_1Q.items():
        for b, pb in PAULI_1Q.items():
            rho = rho + pauli_expectations[a + b] * np.kron(pa, pb)
    return project_to_density_matrix(rho / 4.0)


def superoperator_from_input_output_states(
    input_states: Sequence[ComplexMatrix], output_states: Sequence[ComplexMatrix]
) -> ComplexMatrix:
    if len(input_states) != len(output_states):
        raise ValueError("Input and output state lists must have the same length.")
    r_in = np.concatenate([vec(rho) for rho in input_states], axis=1)
    r_out = np.concatenate([vec(rho) for rho in output_states], axis=1)
    return r_out @ np.linalg.pinv(r_in)


def superoperator_to_choi(superop: ComplexMatrix, dim: int = 4) -> ComplexMatrix:
    reshaped = superop.reshape((dim, dim, dim, dim), order="F")
    choi = np.transpose(reshaped, (0, 2, 1, 3)).reshape((dim * dim, dim * dim), order="F")
    return 0.5 * (choi + choi.conj().T)


def dominant_kraus_from_choi(choi: ComplexMatrix, dim: int = 4) -> ComplexMatrix:
    evals, evecs = np.linalg.eigh(choi)
    idx = int(np.argmax(np.real(evals)))
    lam = max(0.0, float(np.real(evals[idx])))
    kraus_vec = math.sqrt(lam) * evecs[:, idx]
    return unvec(kraus_vec, dim=dim)


# --------------------------------------------------------------------------------------
# D_loc minimization
# --------------------------------------------------------------------------------------


def dloc_minimization(unitary_4x4: ComplexMatrix, n_restarts: int = 8, seed: int = 1234) -> Tuple[float, ComplexMatrix, ComplexMatrix]:
    rng = np.random.default_rng(seed)
    bounds = [(-2.0 * math.pi, 2.0 * math.pi)] * 8

    def objective(params: RealArray) -> float:
        a = u2_from_euler(params[:4])
        b = u2_from_euler(params[4:])
        return float(np.linalg.norm(unitary_4x4 - np.kron(a, b), ord="fro"))

    best_value = float("inf")
    best_params: Optional[RealArray] = None
    initial_guesses: List[RealArray] = [np.zeros(8, dtype=float)]
    initial_guesses.extend(rng.uniform(-math.pi, math.pi, size=8) for _ in range(max(1, n_restarts - 1)))

    for x0 in initial_guesses:
        result = minimize(objective, x0=np.asarray(x0, dtype=float), method="L-BFGS-B", bounds=bounds)
        if result.fun < best_value:
            best_value = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)

    if best_params is None:
        raise RuntimeError("D_loc optimization failed.")
    best_a = u2_from_euler(best_params[:4])
    best_b = u2_from_euler(best_params[4:])
    return best_value, best_a, best_b


# --------------------------------------------------------------------------------------
# Grouped reconstruction containers
# --------------------------------------------------------------------------------------


@dataclass
class OutputStateSummary:
    model: str
    loop: str
    input_a: str
    input_b: str
    pauli_expectations: Dict[str, float]
    density_matrix: ComplexMatrix
    concurrence: float
    ideal_density_matrix: ComplexMatrix
    state_fidelity: float

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "loop": self.loop,
            "input_a": self.input_a,
            "input_b": self.input_b,
            "pauli_expectations": self.pauli_expectations,
            "density_matrix_real": np.real(self.density_matrix).tolist(),
            "density_matrix_imag": np.imag(self.density_matrix).tolist(),
            "ideal_density_matrix_real": np.real(self.ideal_density_matrix).tolist(),
            "ideal_density_matrix_imag": np.imag(self.ideal_density_matrix).tolist(),
            "concurrence": self.concurrence,
            "state_fidelity": self.state_fidelity,
        }


@dataclass
class ProcessTomographySummary:
    model: str
    loop: str
    superoperator: ComplexMatrix
    choi: ComplexMatrix
    dominant_kraus: ComplexMatrix
    nearest_unitary: ComplexMatrix
    ideal_unitary: ComplexMatrix
    dloc_estimated: float
    dloc_ideal: float
    best_local_a: ComplexMatrix
    best_local_b: ComplexMatrix
    operator_schmidt: List[float]
    ideal_operator_schmidt: List[float]
    eigenphases_estimated: List[float]
    eigenphases_ideal: List[float]
    unitary_overlap: float

    def as_serializable(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "loop": self.loop,
            "dloc_estimated": self.dloc_estimated,
            "dloc_ideal": self.dloc_ideal,
            "operator_schmidt": self.operator_schmidt,
            "ideal_operator_schmidt": self.ideal_operator_schmidt,
            "eigenphases_estimated": self.eigenphases_estimated,
            "eigenphases_ideal": self.eigenphases_ideal,
            "unitary_overlap": self.unitary_overlap,
            "superoperator_real": np.real(self.superoperator).tolist(),
            "superoperator_imag": np.imag(self.superoperator).tolist(),
            "choi_real": np.real(self.choi).tolist(),
            "choi_imag": np.imag(self.choi).tolist(),
            "dominant_kraus_real": np.real(self.dominant_kraus).tolist(),
            "dominant_kraus_imag": np.imag(self.dominant_kraus).tolist(),
            "nearest_unitary_real": np.real(self.nearest_unitary).tolist(),
            "nearest_unitary_imag": np.imag(self.nearest_unitary).tolist(),
            "ideal_unitary_real": np.real(self.ideal_unitary).tolist(),
            "ideal_unitary_imag": np.imag(self.ideal_unitary).tolist(),
            "best_local_a_real": np.real(self.best_local_a).tolist(),
            "best_local_a_imag": np.imag(self.best_local_a).tolist(),
            "best_local_b_real": np.real(self.best_local_b).tolist(),
            "best_local_b_imag": np.imag(self.best_local_b).tolist(),
        }


def group_counts_by_model_loop_and_input(
    specs: Sequence[CircuitSpec], counts_list: Sequence[Mapping[str, int]]
) -> Dict[str, Dict[str, Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, int]]]]]:
    grouped: Dict[str, Dict[str, Dict[Tuple[str, str], Dict[Tuple[str, str], Dict[str, int]]]]] = {}
    for spec, counts in zip(specs, counts_list):
        grouped.setdefault(spec.model, {}).setdefault(spec.loop, {}).setdefault(spec.input_key, {})[
            spec.measurement_key
        ] = dict(counts)
    return grouped


def reconstruct_output_states(
    grouped_counts: Mapping[str, Mapping[str, Mapping[Tuple[str, str], Mapping[Tuple[str, str], Mapping[str, int]]]]]
) -> Dict[str, Dict[str, Dict[Tuple[str, str], OutputStateSummary]]]:
    output: Dict[str, Dict[str, Dict[Tuple[str, str], OutputStateSummary]]] = {}
    for model, per_loop in grouped_counts.items():
        output[model] = {}
        for loop, per_input in per_loop.items():
            output[model][loop] = {}
            for input_key, setting_counts in per_input.items():
                pauli_exps = reconstruct_pauli_expectations(setting_counts)
                rho = density_from_pauli_expectations(pauli_exps)
                rho_ideal = ideal_output_density(model, loop, input_key[0], input_key[1])
                summary = OutputStateSummary(
                    model=model,
                    loop=loop,
                    input_a=input_key[0],
                    input_b=input_key[1],
                    pauli_expectations=pauli_exps,
                    density_matrix=rho,
                    concurrence=concurrence(rho),
                    ideal_density_matrix=rho_ideal,
                    state_fidelity=state_fidelity(rho, rho_ideal),
                )
                output[model][loop][input_key] = summary
    return output


def run_process_tomography_from_output_states(
    output_states: Mapping[str, Mapping[str, Mapping[Tuple[str, str], OutputStateSummary]]],
    seed: int = 1234,
    n_restarts: int = 8,
) -> Dict[str, Dict[str, ProcessTomographySummary]]:
    summaries: Dict[str, Dict[str, ProcessTomographySummary]] = {}
    ordered_input_keys = [(a, b) for a in SINGLE_QUBIT_STATE_LABELS for b in SINGLE_QUBIT_STATE_LABELS]
    for model, per_loop in output_states.items():
        summaries[model] = {}
        for loop, per_input in per_loop.items():
            input_density_list = [two_qubit_input_density(a, b) for (a, b) in ordered_input_keys]
            output_density_list = [per_input[(a, b)].density_matrix for (a, b) in ordered_input_keys]
            superop = superoperator_from_input_output_states(input_density_list, output_density_list)
            choi = superoperator_to_choi(superop, dim=4)

            choi_eval, choi_evec = np.linalg.eigh(0.5 * (choi + choi.conj().T))
            choi_eval = np.clip(np.real(choi_eval), 0.0, None)
            if float(choi_eval.sum()) > 1e-14:
                choi = choi_evec @ np.diag(choi_eval) @ choi_evec.conj().T

            dominant_k = dominant_kraus_from_choi(choi, dim=4)
            unitary_est = nearest_unitary(dominant_k)
            dloc_value, best_a, best_b = dloc_minimization(unitary_est, n_restarts=n_restarts, seed=seed)

            ideal_u = get_effective_unitary(model, loop)
            dloc_ideal, _, _ = dloc_minimization(ideal_u, n_restarts=n_restarts, seed=seed)
            summaries[model][loop] = ProcessTomographySummary(
                model=model,
                loop=loop,
                superoperator=superop,
                choi=choi,
                dominant_kraus=dominant_k,
                nearest_unitary=unitary_est,
                ideal_unitary=ideal_u,
                dloc_estimated=dloc_value,
                dloc_ideal=dloc_ideal,
                best_local_a=best_a,
                best_local_b=best_b,
                operator_schmidt=operator_schmidt_singular_values(unitary_est),
                ideal_operator_schmidt=operator_schmidt_singular_values(ideal_u),
                eigenphases_estimated=sorted_eigenphases(unitary_est),
                eigenphases_ideal=sorted_eigenphases(ideal_u),
                unitary_overlap=average_gate_overlap(ideal_u, unitary_est),
            )
    return summaries


# --------------------------------------------------------------------------------------
# Summary tables and scans
# --------------------------------------------------------------------------------------


def build_overview_dataframe(
    witness_states: Mapping[str, Mapping[str, OutputStateSummary]],
    process_summary: Optional[Mapping[str, Mapping[str, ProcessTomographySummary]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in DEFAULT_MODEL_ORDER:
        if model not in witness_states:
            continue
        for loop, witness in witness_states[model].items():
            row: Dict[str, Any] = {
                "model": model,
                "loop": loop,
                "paper_dloc": MODEL_CATALOG[model].loops[loop].paper_dloc,
                "witness_concurrence": witness.concurrence,
                "witness_state_fidelity": witness.state_fidelity,
            }
            if process_summary is not None and model in process_summary and loop in process_summary[model]:
                proc = process_summary[model][loop]
                row.update(
                    {
                        "estimated_dloc": proc.dloc_estimated,
                        "ideal_dloc": proc.dloc_ideal,
                        "unitary_overlap": proc.unitary_overlap,
                        "schmidt_s1": proc.operator_schmidt[0],
                        "schmidt_s2": proc.operator_schmidt[1],
                        "schmidt_s3": proc.operator_schmidt[2],
                        "schmidt_s4": proc.operator_schmidt[3],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_smooth_scan_dataframe(config: ExecutionConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scales = np.linspace(0.0, 1.0, max(2, int(config.scan_points)))
    psi_pp = two_qubit_input_statevector("+", "+")
    for model in resolve_models(config):
        for loop in resolve_loops(model, config):
            for scale in scales:
                unitary = get_effective_unitary(model, loop, scale=scale)
                # For dense scans we use the operator-Schmidt tail as a very accurate
                # proxy for D_loc. For the endpoint representative loops, the exact
                # D_loc values are still reconstructed from full process tomography.
                svals = operator_schmidt_singular_values(unitary)
                dloc_proxy = float(math.sqrt(sum(val * val for val in svals[1:])))
                psi_out = unitary @ psi_pp
                rho_out = np.outer(psi_out, psi_out.conj())
                rows.append(
                    {
                        "model": model,
                        "loop": loop,
                        "scale": float(scale),
                        "dloc": dloc_proxy,
                        "concurrence": concurrence(rho_out),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def public_config_summary(config: ExecutionConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "backend_name": config.backend_name,
        "models": list(config.models),
        "shots": config.shots,
        "optimization_level": config.optimization_level,
        "seed_transpiler": config.seed_transpiler,
        "seed_reference": config.seed_reference,
        "with_process_tomography": config.with_process_tomography,
        "scan_points": config.scan_points,
        "channel": config.channel,
        "save_raw_counts": config.save_raw_counts,
        "selected_loops": None if config.selected_loops is None else {k: list(v) for k, v in config.selected_loops.items()},
    }
    return payload


def save_overview_dloc_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    if "estimated_dloc" not in df.columns:
        raise ValueError("Overview dataframe does not contain estimated_dloc.")
    labels = [f"{m}:{l}" for m, l in zip(df["model"], df["loop"])]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(10.0, 0.9 * len(labels)), 5.8))
    plt.bar(x - width / 2, df["paper_dloc"].fillna(np.nan), width=width, label="paper")
    plt.bar(x + width / 2, df["estimated_dloc"], width=width, label="reconstructed")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(r"$D_{\mathrm{loc}}$")
    plt.title("Representative loops across BHZ / SSH / BBH")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "overview_dloc.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_overview_concurrence_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    labels = [f"{m}:{l}" for m, l in zip(df["model"], df["loop"])]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(10.0, 0.9 * len(labels)), 5.8))
    plt.bar(x, df["witness_concurrence"], width=0.6)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Concurrence from |++> output")
    plt.title("Output-state tomography witness across representative loops")
    plt.tight_layout()
    path = output_dir / "overview_concurrence.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_operator_schmidt_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    if not {"schmidt_s1", "schmidt_s2", "schmidt_s3", "schmidt_s4"}.issubset(df.columns):
        raise ValueError("Overview dataframe does not contain operator-Schmidt columns.")
    labels = [f"{m}:{l}" for m, l in zip(df["model"], df["loop"])]
    x = np.arange(len(labels))
    width = 0.18
    plt.figure(figsize=(max(10.0, 0.9 * len(labels)), 5.8))
    for idx, col in enumerate(["schmidt_s1", "schmidt_s2", "schmidt_s3", "schmidt_s4"]):
        plt.bar(x + (idx - 1.5) * width, df[col], width=width, label=col)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Operator-Schmidt singular value")
    plt.title("Reconstructed operator-Schmidt spectra")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "overview_operator_schmidt.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_selected_correlator_plot(
    witness_states: Mapping[str, Mapping[str, OutputStateSummary]], output_dir: Path, model: str
) -> Path:
    loops = list(witness_states[model].keys())
    x = np.arange(len(SELECTED_OBSERVABLES))
    width = 0.8 / max(1, len(loops))
    plt.figure(figsize=(11.0, 5.6))
    for idx, loop in enumerate(loops):
        vals = [witness_states[model][loop].pauli_expectations[obs] for obs in SELECTED_OBSERVABLES]
        plt.bar(x + (idx - 0.5 * (len(loops) - 1)) * width, vals, width=width, label=loop)
    plt.xticks(x, [obs[0] + "⊗" + obs[1] for obs in SELECTED_OBSERVABLES], fontsize=11)
    plt.ylim(-1.05, 1.05)
    plt.ylabel("Expectation value")
    plt.title(f"{model}: selected correlators from |++> output-state tomography")
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"{model.lower()}_selected_correlators.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_smooth_scan_plot(scan_df: pd.DataFrame, output_dir: Path, model: str, value_column: str, ylabel: str, title_suffix: str) -> Path:
    subset = scan_df[scan_df["model"] == model].copy()
    plt.figure(figsize=(8.8, 5.5))
    for loop, group in subset.groupby("loop", sort=False):
        group = group.sort_values("scale")
        plt.plot(group["scale"], group[value_column], linewidth=2.0, label=loop)
    plt.xlabel("effective loop scale")
    plt.ylabel(ylabel)
    plt.title(f"{model}: {title_suffix}")
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"{model.lower()}_{value_column}_scan.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# --------------------------------------------------------------------------------------
# Main execution flow
# --------------------------------------------------------------------------------------


def run_suite(config: ExecutionConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    witness_specs = build_witness_specs(config)
    if config.backend_name == "analytic_reference":
        witness_counts = run_reference_counts(witness_specs, shots=config.shots, seed=config.seed_reference)
    else:
        backend, sampler = build_backend_and_sampler(config)
        witness_circuits = [build_circuit(spec) for spec in witness_specs]
        witness_isa = transpile_circuits(
            witness_circuits,
            backend,
            config.optimization_level,
            config.seed_transpiler,
        )
        witness_counts = run_sampler_counts(witness_isa, sampler, config.shots)

    if config.save_raw_counts:
        save_json(
            output_dir / "witness_raw_counts.json",
            [{"spec": asdict(spec), "counts": counts} for spec, counts in zip(witness_specs, witness_counts)],
        )

    grouped_witness_counts = group_counts_by_model_loop_and_input(witness_specs, witness_counts)
    witness_state_dict_full = reconstruct_output_states(grouped_witness_counts)
    witness_pp = {
        model: {loop: witness_state_dict_full[model][loop][("+", "+")] for loop in witness_state_dict_full[model]}
        for model in witness_state_dict_full
    }

    save_json(
        output_dir / "witness_summary.json",
        {
            model: {loop: state.as_serializable() for loop, state in per_loop.items()}
            for model, per_loop in witness_pp.items()
        },
    )

    process_summary: Optional[Dict[str, Dict[str, ProcessTomographySummary]]] = None
    if config.with_process_tomography:
        proc_specs = build_process_tomography_specs(config)
        if config.backend_name == "analytic_reference":
            proc_counts = run_reference_counts(proc_specs, shots=config.shots, seed=config.seed_reference + 1)
        else:
            backend, sampler = build_backend_and_sampler(config)
            proc_circuits = [build_circuit(spec) for spec in proc_specs]
            proc_isa = transpile_circuits(
                proc_circuits,
                backend,
                config.optimization_level,
                config.seed_transpiler,
            )
            proc_counts = run_sampler_counts(proc_isa, sampler, config.shots)

        if config.save_raw_counts:
            save_json(
                output_dir / "process_raw_counts.json",
                [{"spec": asdict(spec), "counts": counts} for spec, counts in zip(proc_specs, proc_counts)],
            )

        grouped_proc_counts = group_counts_by_model_loop_and_input(proc_specs, proc_counts)
        proc_state_dict = reconstruct_output_states(grouped_proc_counts)
        process_summary = run_process_tomography_from_output_states(proc_state_dict, seed=config.seed_reference)
        save_json(
            output_dir / "process_summary.json",
            {
                model: {loop: summary.as_serializable() for loop, summary in per_loop.items()}
                for model, per_loop in process_summary.items()
            },
        )

    overview_df = build_overview_dataframe(witness_pp, process_summary)
    overview_csv = output_dir / "overview_summary.csv"
    overview_df.to_csv(overview_csv, index=False)
    save_json(output_dir / "overview_summary.json", overview_df.to_dict(orient="records"))

    scan_df = build_smooth_scan_dataframe(config)
    scan_csv = output_dir / "smooth_scans.csv"
    scan_df.to_csv(scan_csv, index=False)

    artifacts: Dict[str, str] = {
        "overview_summary_csv": overview_csv.name,
        "overview_summary_json": "overview_summary.json",
        "witness_summary_json": "witness_summary.json",
        "smooth_scans_csv": scan_csv.name,
    }
    if config.save_raw_counts:
        artifacts["witness_raw_counts_json"] = "witness_raw_counts.json"
        if config.with_process_tomography:
            artifacts["process_raw_counts_json"] = "process_raw_counts.json"
    if config.with_process_tomography:
        artifacts["process_summary_json"] = "process_summary.json"

    if config.with_process_tomography:
        artifacts["overview_dloc_plot"] = save_overview_dloc_plot(overview_df, output_dir).name
        artifacts["overview_operator_schmidt_plot"] = save_operator_schmidt_plot(overview_df, output_dir).name
    artifacts["overview_concurrence_plot"] = save_overview_concurrence_plot(overview_df, output_dir).name

    for model in resolve_models(config):
        artifacts[f"{model.lower()}_selected_correlators_plot"] = save_selected_correlator_plot(witness_pp, output_dir, model).name
        artifacts[f"{model.lower()}_dloc_scan_plot"] = (
            save_smooth_scan_plot(
                scan_df,
                output_dir,
                model=model,
                value_column="dloc",
                ylabel=r"$D_{\mathrm{loc}}$",
                title_suffix="smooth effective-generator scan",
            ).name
        )
        artifacts[f"{model.lower()}_concurrence_scan_plot"] = (
            save_smooth_scan_plot(
                scan_df,
                output_dir,
                model=model,
                value_column="concurrence",
                ylabel="Concurrence from |++> output",
                title_suffix="smooth |++> concurrence scan",
            ).name
        )

    result_payload: Dict[str, Any] = {
        "config": public_config_summary(config),
        "overview": overview_df.to_dict(orient="records"),
        "artifacts": artifacts,
    }
    save_json(output_dir / "run_summary.json", result_payload)
    result_payload["artifacts"]["run_summary_json"] = "run_summary.json"
    return result_payload


# --------------------------------------------------------------------------------------
# Convenience helpers for notebooks / CLI
# --------------------------------------------------------------------------------------




def catalog_dataframe() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in DEFAULT_MODEL_ORDER:
        mdef = MODEL_CATALOG[model]
        for loop_name, loop_def in mdef.loops.items():
            rows.append(
                {
                    "model": model,
                    "qubit_a": mdef.qubit_labels[0],
                    "qubit_b": mdef.qubit_labels[1],
                    "loop": loop_name,
                    "paper_dloc": loop_def.paper_dloc,
                    "description": loop_def.description,
                    "paper_note": loop_def.paper_note,
                }
            )
    return pd.DataFrame(rows)

def default_backend_name() -> str:
    return "local_aer" if qiskit_available() else "analytic_reference"


def load_overview_dataframe(output_dir: str | Path) -> pd.DataFrame:
    path = Path(output_dir) / "overview_summary.csv"
    return pd.read_csv(path)


def human_summary_text(summary: Mapping[str, Any]) -> str:
    lines = ["=== All-model tomography summary ==="]
    for row in summary.get("overview", []):
        model = row["model"]
        loop = row["loop"]
        bits = [f"{model}:{loop}"]
        if "estimated_dloc" in row:
            bits.append(f"D_loc={row['estimated_dloc']:.4f}")
        bits.append(f"C(|++>)={row['witness_concurrence']:.4f}")
        lines.append(" | ".join(bits))
    lines.append("\nArtifacts:")
    for key, value in summary.get("artifacts", {}).items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="All-model extracted quartet tomography suite.")
    parser.add_argument("--backend-name", default=default_backend_name(), help="analytic_reference, local_aer, or an IBM backend name")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--output-dir", default="quartet_tomography_outputs")
    parser.add_argument("--scan-points", type=int, default=201)
    parser.add_argument("--with-process-tomography", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_ORDER))
    parser.add_argument("--token", default=None)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--channel", default="ibm_quantum_platform")
    parser.add_argument("--no-save-raw-counts", action="store_true")
    args = parser.parse_args()

    config = ExecutionConfig(
        backend_name=args.backend_name,
        models=tuple(args.models),
        shots=args.shots,
        output_dir=args.output_dir,
        scan_points=args.scan_points,
        with_process_tomography=args.with_process_tomography,
        token=args.token,
        instance=args.instance,
        channel=args.channel,
        save_raw_counts=not args.no_save_raw_counts,
    )
    summary = run_suite(config)
    print(human_summary_text(summary))


if __name__ == "__main__":
    main()
