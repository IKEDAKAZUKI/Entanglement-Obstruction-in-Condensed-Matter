#!/usr/bin/env python3
"""Assemble the compact summary JSON file used by the figure builder."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SUMMARY_PATH = DATA / 'manuscript_summary.json'

with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
    summary = json.load(f)


ssh_grid = pd.read_csv(DATA / 'ssh_torus_grid.csv')
ssh_quality = pd.read_csv(DATA / 'ssh_left_loop_quality.csv')
ssh_sizes = pd.read_csv(DATA / 'ssh_size_scaling.csv')
summary['ssh']['gap_min'] = float(ssh_grid['gap'].min())
summary['ssh']['joint_max'] = float(ssh_grid['joint'].max())
summary['ssh']['deltaA_min'] = float(ssh_grid['deltaA'].min())
summary['ssh']['deltaB_min'] = float(ssh_grid['deltaB'].min())
summary['ssh']['edge_weight_min'] = float(ssh_grid['edge_weight'].min())
summary['ssh']['left_quality'] = {
    'gap_min': float(ssh_quality['gap'].min()),
    'gap_max': float(ssh_quality['gap'].max()),
    'joint_max': float(ssh_quality['joint'].max()),
    'deltaA_min': float(ssh_quality['deltaA'].min()),
    'deltaB_min': float(ssh_quality['deltaB'].min()),
    'overlap_min': float(ssh_quality['frame_overlap'].min()),
}
summary['ssh']['size_scaling'] = ssh_sizes.to_dict(orient='records')

bbh_grid = pd.read_csv(DATA / 'bbh_torus_grid.csv')
bbh_quality = pd.read_csv(DATA / 'bbh_diagonal_loop_quality.csv')
summary['bbh']['gap_min'] = float(bbh_grid['gap'].min())
summary['bbh']['joint_max'] = float(bbh_grid['joint'].max())
summary['bbh']['deltaA_min'] = float(bbh_grid['deltaA'].min())
summary['bbh']['deltaB_min'] = float(bbh_grid['deltaB'].min())
summary['bbh']['corner_weight_min'] = float(bbh_grid['corner_weight'].min())
summary['bbh']['diag_quality'] = {
    'gap_min': float(bbh_quality['gap'].min()),
    'gap_max': float(bbh_quality['gap'].max()),
    'joint_max': float(bbh_quality['joint'].max()),
    'deltaA_min': float(bbh_quality['deltaA'].min()),
    'deltaB_min': float(bbh_quality['deltaB'].min()),
    'corner_weight_min': float(bbh_quality['corner_weight'].min()),
    'overlap_min': float(bbh_quality['frame_overlap'].min()),
}

scan = pd.read_csv(DATA / 'continuous_loop_scan.csv')
loop_summary = {
    'eta_window': [-2.0, 2.0],
    'models': {},
}
for model in ['SSH', 'BBH', 'BHZ']:
    sub = scan[(scan['model'] == model) & (scan['eta_numeric'].notna())].sort_values('eta_numeric')
    axis2 = scan[(scan['model'] == model) & (scan['eta_axis2'] == True)].iloc[0]
    row_max = sub.loc[sub['D_loc'].idxmax()]
    row_min = sub.loc[sub['D_loc'].idxmin()]
    loop_summary['models'][model.lower()] = {
        'max_D_loc': {
            'eta': float(row_max['eta_numeric']),
            'D_loc': float(row_max['D_loc']),
            'ep_mean': float(row_max['ep_mean']),
        },
        'min_D_loc': {
            'eta': float(row_min['eta_numeric']),
            'D_loc': float(row_min['D_loc']),
            'ep_mean': float(row_min['ep_mean']),
        },
        'second_axis_control': {
            'D_loc': float(axis2['D_loc']),
            'ep_mean': float(axis2['ep_mean']),
        },
    }
summary['loop_family_scan'] = loop_summary

with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f'Wrote {SUMMARY_PATH}.')
