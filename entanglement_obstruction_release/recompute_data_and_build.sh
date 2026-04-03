#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$(dirname "$0")"
python3 generate_datasets.py
python3 compute_grid_datasets.py
python3 compute_loop_scan.py
python3 build_summary.py
python3 compute_auxiliary_datasets.py
python3 make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error entangling_gluing_manuscript.tex
