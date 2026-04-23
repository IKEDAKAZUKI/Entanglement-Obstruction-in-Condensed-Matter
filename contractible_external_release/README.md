# Contractible-loop data for the entanglement-obstruction benchmarks

This directory contains the auxiliary data and scripts for the contractible-circle controls centered at

\[
(\theta_1,\theta_2)/\pi=(1,1).
\]

The loop family is

\[
C_0(\rho):\quad (\theta_1,\theta_2)/\pi=(1,1)+\rho(\cos t,\sin t),\qquad t\in[0,2\pi].
\]

The data are intended to be used together with the released benchmark code in
`entanglement_obstruction_release`.

## Contents

- `code/compute_contractible_center11.py` computes the radius scan and the orientation-reversal check using the released model definitions and diagnostics.
- `code/plot_contractible_center11.py` recreates the two contractible-loop figures from the CSV tables.
- `data/contractible_center11_radius_scan.csv` contains the radius scan for BHZ, SSH, and BBH.
- `data/contractible_center11_orientation_check.csv` contains the positive- and reverse-orientation check at the radius indicated in the table.
- `data/contractible_center11_summary.json` records the loop family, radius range, and largest BBH response in the scan window.
- `data/heatmap_Nloc_*_n81.csv` contains the refined local-connection maps used for the overlay figure.
- `figures/fig_contractible_loops.*` and `figures/fig_contractible_radius_scan.*` are the rendered figures.

## Recomputing the tables

From the parent directory of this package, run

```bash
python code/compute_contractible_center11.py \
  --release /path/to/entanglement_obstruction_release \
  --out data
```

The default loop discretization uses 121 points. It can be changed with `--n-loop`.

## Recreating the figures

```bash
python code/plot_contractible_center11.py \
  --data data \
  --out figures
```

## Table columns

- `model`: benchmark model.
- `orientation`: loop orientation, `+` or `-`.
- `center_theta1_over_pi`, `center_theta2_over_pi`: loop center.
- `radius_over_pi`: circle radius in units of \(\pi\).
- `n_points`: number of loop discretization points.
- `D_loc_strict`: distance to the extracted local subgroup.
- `D_loc_swap`: distance to the swap-extended local subgroup.
- `ep_exact`: exact two-qubit entangling power.
- `arg_det`: argument of the determinant of the holonomy.
- `max_abs_phase`: largest absolute eigenphase.
- `eigphase_1`--`eigphase_4`: sorted eigenphases.
- `schmidt_1`--`schmidt_4`: operator-Schmidt singular values.
- `inverse_consistency_norm`: Frobenius norm of \(U_- - U_+^\dagger\), included only in the orientation-check table.
