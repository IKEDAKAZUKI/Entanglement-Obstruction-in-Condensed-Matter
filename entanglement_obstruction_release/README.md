# Source release

This release contains the manuscript source, the numerical data tables used for the figures, and the scripts needed to rebuild the figures and the manuscript PDF.

## Contents

- `core_models.py`: model Hamiltonians, quartet extraction, and holonomy utilities
- `generate_datasets.py`: full data generation from the lattice models
- `compute_grid_datasets.py`: additional high-resolution grid calculations and loop diagnostics
- `compute_loop_scan.py`: continuous loop-scan data used in the mechanism figure
- `compute_auxiliary_datasets.py`: supporting BHZ and BBH data tables used in the figure set
- `build_summary.py`: rebuilds `data/manuscript_summary.json` from the released CSV tables
- `make_figures.py`: regenerates the figure files from the released data tables
- `recompute_data_and_build.sh`: regenerates the numerical data tables and then rebuilds the figures and manuscript PDF
- `data/`: released CSV and JSON data tables

## Requirements

- Python 3.10 or newer
- `latexmk` and a standard LaTeX installation
- Python packages listed in `requirements.txt`

Install the Python packages with

```bash
python3 -m pip install -r requirements.txt
```

## Rebuild from the released data

```bash
bash build_manuscript.sh
```

This command rebuilds the summary file, regenerates the figures from the released data tables, and recompiles the manuscript PDF.

## Full data regeneration

```bash
bash recompute_data_and_build.sh
```

This command reruns the numerical calculations, rebuilds the summary file, regenerates the figures, and recompiles the manuscript PDF.

## Notes

- All paths are relative to the extracted release directory.
- The files under `data/` are the inputs used by the released manuscript PDF.
- The shell scripts set BLAS and OpenMP thread counts to 1 for reproducible runtimes.
