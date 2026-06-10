<div align="center">

# Entanglement Obstruction in Condensed Matter

### Loop-dependent entangling holonomies in localized topological quartets

[![arXiv:2604.11596](https://img.shields.io/badge/arXiv-2604.11596-b31b1b.svg)](https://arxiv.org/abs/2604.11596)
[![arXiv:2601.13764](https://img.shields.io/badge/Math%20Foundation-arXiv%3A2601.13764-blue.svg)](https://arxiv.org/abs/2601.13764)
[![Lecture Notes](https://img.shields.io/badge/Lecture%20Notes-arXiv%3A2601.19111-green.svg)](https://arxiv.org/abs/2601.19111)
[![Python](https://img.shields.io/badge/Python-Research%20Code-3776AB)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Tomography%20Workflow-6929C4)](https://www.ibm.com/quantum/qiskit)

<br>

**Entanglement geometry · Holonomy · Localized topological quartets · BHZ · SSH · BBH · Qiskit tomography**

<br>

<img src="Entanglement%20Geometry%20slides.gif" width="74%" alt="Entanglement Geometry slides">

</div>

---

## Overview

This repository contains research code, numerical datasets, reproducibility scripts, Qiskit tomography workflows, and presentation material for studying **entanglement obstruction in condensed-matter systems**.

The central idea is that an isolated multiplet may admit a local tensor-product structure at each point in parameter space, while the corresponding structure may fail to globalize around a loop. This failure appears as a **loop-dependent entangling holonomy**: the loop holonomy can leave the extracted local subgroup and become a genuinely entangling operation.

In condensed-matter terms, the repository studies this phenomenon in localized topological quartet systems, including:

- BHZ ribbon edge quartets,
- spinful SSH edge quartets,
- BBH higher-order corner quartets,
- contractible-loop controls,
- Qiskit-based effective two-qubit tomography.

---

## Scientific context

This project sits at the intersection of quantum information, condensed-matter physics, topology, and algebraic geometry.

### Mathematical foundation

The mathematical framework is based on:

> K. Ikeda,  
> **“Quantum Entanglement Geometry on Severi-Brauer Schemes: Subsystem Reductions of Azumaya Algebras,”**  
> arXiv:2601.13764.  
> [https://arxiv.org/abs/2601.13764](https://arxiv.org/abs/2601.13764)

This work formulates entanglement as a geometric obstruction to globalizing subsystem structure over a parameter space.

### Condensed-matter realization

The numerical and physical realization of the code is connected to:

> K. Ikeda and Y. Oz,  
> **“Loop-dependent entangling holonomies in localized topological quartets,”**  
> arXiv:2604.11596.  
> [https://arxiv.org/abs/2604.11596](https://arxiv.org/abs/2604.11596)

This work studies loop-dependent entangling holonomies in localized topological quartets and shows that conventional Berry data may not distinguish local and entangling holonomies.

### Lecture notes

For a broader pedagogical introduction, see:

> K. Ikeda,  
> **“Introduction to Quantum Entanglement Geometry,”**  
> arXiv:2601.19111.  
> [https://arxiv.org/abs/2601.19111](https://arxiv.org/abs/2601.19111)

These notes introduce quantum entanglement geometry, subsystem-reduction obstructions, Severi-Brauer geometry, and the role of entangling holonomies.

---

## Project map

| Component | Directory / File | Purpose |
|---|---|---|
| Main numerical release | [`entanglement_obstruction_release/`](entanglement_obstruction_release/) | Core models, quartet extraction, holonomy diagnostics, released data tables, and figure-generation scripts |
| Contractible-loop controls | [`contractible_external_release/`](contractible_external_release/) | Auxiliary contractible-circle benchmarks and orientation checks |
| Qiskit tomography workflow | [`Qiskit_tomography_public_package/`](Qiskit_tomography_public_package/) | Effective two-qubit tomography for BHZ, SSH, and BBH quartet models |
| Slides | [`Entanglement Geometry slides.gif`](Entanglement%20Geometry%20slides.gif) | Animated overview slides |
| Slides PDF | [`Entanglement Geometry slides.pdf`](Entanglement%20Geometry%20slides.pdf) | Static slide deck |

---

## Key features

- **Loop-dependent holonomy diagnostics** for isolated topological quartets.
- **Distance-to-local-subgroup analysis** for detecting entanglement obstruction.
- **BHZ, SSH, and BBH benchmark models**.
- **Contractible-loop control data** for comparison with nontrivial loop responses.
- **Operator-Schmidt and entangling-power diagnostics**.
- **Released CSV/JSON datasets** for reproducibility.
- **Figure-generation scripts** for reconstructing the numerical results.
- **Qiskit-based tomography workflow** for effective two-qubit quartet dynamics.
- **Analytic reference mode** for running selected workflows without Qiskit hardware access.

---

## Conceptual summary

A localized quartet may look like an effective two-qubit system at every point of a parameter space. The obstruction studied here asks whether this local two-qubit description can be chosen consistently around a loop.

In ordinary band topology, one often studies Berry phases, Chern numbers, determinant phases, and eigenphase spectra. In the entanglement-obstruction setting, these quantities may not fully capture whether the loop holonomy is local or entangling.

The relevant question is instead:

> Does the holonomy remain inside the extracted local subgroup, or does it become an entangling operation?

For a quartet with effective two-qubit structure, the local subgroup is morally the embedded two-qubit local unitary group

```text
U(2) ⊗ U(2)  ⊂  U(4).
```

A holonomy outside this subgroup signals an obstruction to globally trivializing the subsystem decomposition.

---

## Repository structure

```text
.
├── entanglement_obstruction_release/
│   ├── data/
│   ├── build_summary.py
│   ├── compute_auxiliary_datasets.py
│   ├── compute_grid_datasets.py
│   ├── compute_loop_scan.py
│   ├── core_models.py
│   ├── generate_datasets.py
│   ├── make_figures.py
│   ├── recompute_data_and_build.sh
│   ├── requirements.txt
│   └── README.md
│
├── contractible_external_release/
│   ├── code/
│   ├── data/
│   ├── figures/
│   ├── MANIFEST.sha256
│   ├── requirements.txt
│   └── README.md
│
├── Qiskit_tomography_public_package/
│   ├── data/
│   ├── quartet_tomography_all_models.ipynb
│   ├── quartet_tomography_all_models.py
│   ├── requirements.txt
│   └── README.md
│
├── Entanglement Geometry slides.gif
└── Entanglement Geometry slides.pdf
```

---

## Quick start

Clone the repository:

```bash
git clone https://github.com/IKEDAKAZUKI/Entanglement-Obstruction-in-Condensed-Matter.git
cd Entanglement-Obstruction-in-Condensed-Matter
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the requirements for the main numerical release:

```bash
python3 -m pip install -r entanglement_obstruction_release/requirements.txt
```

Optional: install requirements for the contractible-loop controls:

```bash
python3 -m pip install -r contractible_external_release/requirements.txt
```

Optional: install requirements for the Qiskit tomography workflow:

```bash
python3 -m pip install -r Qiskit_tomography_public_package/requirements.txt
```

---

## Main numerical release

The main numerical implementation is located in:

```text
entanglement_obstruction_release/
```

This directory contains the model definitions, quartet extraction tools, holonomy utilities, released data tables, and scripts used to regenerate figures.

Important files include:

| File | Description |
|---|---|
| `core_models.py` | Model Hamiltonians, quartet extraction, and holonomy utilities |
| `generate_datasets.py` | Full data generation from the lattice models |
| `compute_grid_datasets.py` | High-resolution grid calculations and loop diagnostics |
| `compute_loop_scan.py` | Continuous loop-scan data for mechanism figures |
| `compute_auxiliary_datasets.py` | Supporting BHZ and BBH data tables |
| `build_summary.py` | Builds `data/manuscript_summary.json` from released CSV tables |
| `make_figures.py` | Regenerates figure files from the released data |
| `recompute_data_and_build.sh` | Full data-regeneration and rebuild script |

To rebuild summary data and figures from released tables:

```bash
cd entanglement_obstruction_release
python3 build_summary.py
python3 make_figures.py
```

For full numerical regeneration:

```bash
bash recompute_data_and_build.sh
```

---

## Contractible-loop controls

The contractible-loop benchmark package is located in:

```text
contractible_external_release/
```

It contains auxiliary data and scripts for contractible-circle controls centered at

```text
(theta_1, theta_2) / pi = (1, 1).
```

The loop family is

```text
C_0(rho):  (theta_1, theta_2) / pi = (1, 1) + rho (cos t, sin t),
            t in [0, 2pi].
```

To recompute the contractible-loop tables:

```bash
cd contractible_external_release
python3 code/compute_contractible_center11.py \
  --release ../entanglement_obstruction_release \
  --out data
```

To recreate the figures:

```bash
python3 code/plot_contractible_center11.py \
  --data data \
  --out figures
```

The released data include radius scans, orientation-reversal checks, refined local-connection maps, and rendered figures.

---

## Qiskit tomography workflow

The Qiskit proof-of-concept package is located in:

```text
Qiskit_tomography_public_package/
```

It provides an effective two-qubit tomography workflow for the BHZ, SSH, and BBH quartet models.

Included representative loops:

| Model | Representative loops |
|---|---|
| BHZ | `CT`, `CB`, `C_plus`, `C_minus` |
| SSH | `CL`, `CR`, `Cdiag`, `Canti` |
| BBH | `Cx`, `Cy`, `Cdiag`, `Canti` |

To run the notebook:

```bash
cd Qiskit_tomography_public_package
jupyter notebook quartet_tomography_all_models.ipynb
```

Inside the notebook, set `BACKEND_NAME` to one of:

```python
"local_aer"            # local Qiskit Aer execution
"analytic_reference"   # reference evaluation without Qiskit
"ibm_brisbane"         # example IBM Quantum backend
```

The notebook builds circuits, runs the selected backend, reconstructs output density matrices, optionally reconstructs process matrices, and saves plots and summary files.

---

## Diagnostics

The released data and scripts use several diagnostics for detecting entanglement obstruction:

| Diagnostic | Meaning |
|---|---|
| `D_loc_strict` | Distance to the extracted local subgroup |
| `D_loc_swap` | Distance to the swap-extended local subgroup |
| `ep_exact` | Exact two-qubit entangling power |
| `arg_det` | Argument of the determinant of the holonomy |
| `max_abs_phase` | Largest absolute eigenphase |
| `eigphase_1`–`eigphase_4` | Sorted holonomy eigenphases |
| `schmidt_1`–`schmidt_4` | Operator-Schmidt singular values |
| `inverse_consistency_norm` | Orientation-reversal consistency check |

These diagnostics are designed to distinguish genuinely entangling holonomies from holonomies that remain local up to the extracted subgroup.

---

## Slides

The animated overview slides are included in the repository:

- [Quantum Entanglement as a Geometric Obstruction: Mathematical and Physical Insights — SlideShare](https://www.slideshare.net/slideshow/quantum-entanglement-as-a-geometric-obstruction-mathematical-and-physical-insights/287998123)

```text
Entanglement Geometry slides.gif
```

A static PDF version is also available:

```text
Entanglement Geometry slides.pdf
```

---

## How to cite

Please cite the relevant work depending on how this repository is used.

### Mathematical foundation

```bibtex
@misc{Ikeda2026EntanglementGeometrySeveriBrauer,
  author        = {Ikeda, Kazuki},
  title         = {Quantum Entanglement Geometry on Severi-Brauer Schemes:
                   Subsystem Reductions of Azumaya Algebras},
  year          = {2026},
  eprint        = {2601.13764},
  archivePrefix = {arXiv},
  primaryClass  = {math.AG}
}
```

### Condensed-matter implementation and numerical code

```bibtex
@misc{IkedaOz2026LoopDependentEntanglingHolonomies,
  author        = {Ikeda, Kazuki and Oz, Yaron},
  title         = {Loop-dependent entangling holonomies in localized topological quartets},
  year          = {2026},
  eprint        = {2604.11596},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mes-hall}
}
```

### Lecture notes

```bibtex
@misc{Ikeda2026IntroductionEntanglementGeometry,
  author        = {Ikeda, Kazuki},
  title         = {Introduction to Quantum Entanglement Geometry},
  year          = {2026},
  eprint        = {2601.19111},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

---

## Related links

- Mathematical foundation: [arXiv:2601.13764](https://arxiv.org/abs/2601.13764)
- Condensed-matter implementation: [arXiv:2604.11596](https://arxiv.org/abs/2604.11596)
- Lecture notes: [arXiv:2601.19111](https://arxiv.org/abs/2601.19111)
- Repository: [Entanglement-Obstruction-in-Condensed-Matter](https://github.com/IKEDAKAZUKI/Entanglement-Obstruction-in-Condensed-Matter)

---

## License

A repository-wide license file is not currently included.

Before redistribution or reuse, please add an explicit `LICENSE` file to the root directory. For academic use, please also cite the relevant papers listed above.

---

<div align="center">

**Entanglement Obstruction · Condensed Matter · Quantum Geometry · Holonomy · Topological Quartets**

</div>
