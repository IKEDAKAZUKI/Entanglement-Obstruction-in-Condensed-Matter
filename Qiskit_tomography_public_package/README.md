# Qiskit proof-of-concept for extracted-quartet tomography

This package contains the code, notebook, and example output files for the effective two-qubit tomography workflow used for the BHZ, SSH, and BBH quartet models.

## Included models and representative loops

- **BHZ**: `CT`, `CB`, `C_plus`, `C_minus`
- **SSH**: `CL`, `CR`, `Cdiag`, `Canti`
- **BBH**: `Cx`, `Cy`, `Cdiag`, `Canti`

## Package contents

- `quartet_tomography_all_models.py` — main implementation
- `quartet_tomography_all_models.ipynb` — Jupyter notebook for running the workflow
- `data/` — example JSON/CSV outputs and figure files
- `latex/` — short LaTeX text blocks for describing the implementation in a paper
- `requirements.txt` — Python package requirements

## Requirements

Install the required packages with

```bash
pip install -r requirements.txt
```

The Qiskit execution path uses

- `qiskit`
- `qiskit-aer`
- `qiskit-ibm-runtime`

The package also includes a reference evaluation mode (`analytic_reference`) that can be used without Qiskit.

## Running the notebook

Open `quartet_tomography_all_models.ipynb` and set `BACKEND_NAME` to one of the following:

- `"local_aer"` for local Qiskit execution
- an IBM Quantum backend name such as `"ibm_brisbane"`
- `"analytic_reference"` for reference evaluation without Qiskit

The notebook builds the circuits, runs the selected backend, reconstructs the output density matrices, optionally reconstructs the process matrices, and saves plots and summary files.

## Output files

The example files in `data/` include

- raw counts (`witness_raw_counts.json`, `process_raw_counts.json`)
- tomography summaries (`witness_summary.json`, `process_summary.json`)
- overview tables (`overview_summary.csv`, `overview_summary.json`)
- smooth scan data (`smooth_scans.csv`)
- figure files (`*.png`)

All paths in the distributed files are relative. No credentials or machine-specific paths are included.
