#!/usr/bin/env python3
"""Compute the continuous loop scan used in the mechanism figure."""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

import core_models as core

I2 = np.eye(2, dtype=complex)
SX = np.array([[0,1],[1,0]], dtype=complex)
SY = np.array([[0,-1j],[1j,0]], dtype=complex)
SZ = np.array([[1,0],[0,-1]], dtype=complex)
TAU0 = I2
TAU1 = SX
TAU2 = SY
TAU3 = SZ

def kron(a, b):
    return np.kron(a, b)

def delta_metrics_from_compressed(oa_tilde, ob_tilde):
    a = 0.5*(oa_tilde+oa_tilde.conj().T)
    b = 0.5*(ob_tilde+ob_tilde.conj().T)
    evals_a, vecs_a = np.linalg.eigh(a)
    order = np.argsort(evals_a)
    evals_a = evals_a[order]
    vecs_a = vecs_a[:, order]
    delta_a = float(evals_a[2]-evals_a[1])
    deltas_b=[]
    for idxs in [range(2), range(2,4)]:
        sub = vecs_a[:,idxs]
        b_sub = sub.conj().T @ b @ sub
        evals_b = np.linalg.eigvalsh(0.5*(b_sub+b_sub.conj().T))
        deltas_b.append(float(evals_b[1]-evals_b[0]))
    return delta_a, min(deltas_b)

def frame_overlap_min(f0,f1):
    return float(np.min(np.abs(np.diag(f0.conj().T @ f1))))

def loop_frames(loop_points, data_fn):
    frames = []
    records = []
    overlaps = []
    prev = None
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

def entangling_power_stats(u, n_samples=256, seed=0):
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(n_samples):
        psi=np.kron(core.random_qubit_state(rng), core.random_qubit_state(rng))
        vals.append(core.linear_entropy_from_state(u@psi))
    arr=np.maximum(np.array(vals,dtype=float),0.0)
    return {'mean': float(np.mean(arr)), 'stderr': float(np.std(arr,ddof=1)/np.sqrt(len(arr))), 'n_samples': int(n_samples)}

def holonomy_metrics(u, seed=0, ep_samples=256):
    d_strict, payload = core.best_local_procrustes(u, include_swap=False, n_restart=18, seed=seed)
    ep = entangling_power_stats(u, n_samples=ep_samples, seed=seed)
    return {'D_loc_strict': float(d_strict), 'ep_mean': float(ep['mean']), 'ep_stderr': float(ep['stderr'])}

# SSH
ssh_cfg = core.SSHConfig()
ssh_oa = core.ssh_edge_observable(ssh_cfg)
ssh_ob = core.ssh_spin_observable(ssh_cfg)
ssh_refs = core.ssh_refs(ssh_cfg)

def ssh_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(theta_l, theta_r):
        h=core.build_spinful_ssh(cfg, theta_l, theta_r)
        evals,evecs=np.linalg.eigh(h)
        idx = np.argsort(np.abs(evals))[:4]
        idx = idx[np.argsort(evals[idx])]
        v=evecs[:,idx]
        abs_sorted=np.sort(np.abs(evals))
        gap=float(abs_sorted[4]-abs_sorted[3])
        oa_t,ob_t=core.compress_observables(v,oa,ob)
        frame,_=core.joint_diagonalize(v,oa_t,ob_t)
        frame=core.fix_column_phases(frame,refs)
        delta_a,delta_b=delta_metrics_from_compressed(oa_t,ob_t)
        return {'frame': frame,'gap':gap,'joint': core.jointness_metric(oa_t,ob_t),'deltaA':delta_a,'deltaB':delta_b,'oa_tilde':oa_t,'ob_tilde':ob_t}
    return _data
ssh_data=ssh_data_factory(ssh_cfg, ssh_oa, ssh_ob, ssh_refs)

def ssh_loop(kind, data_fn=ssh_data, n=81):
    ts=np.linspace(0,2*math.pi,n)
    if kind=='right': pts=[(0.0,float(t)) for t in ts]
    elif kind=='left': pts=[(float(t),0.0) for t in ts]
    elif kind=='diag': pts=[(float(t),float(t)) for t in ts]
    elif kind=='anti': pts=[(float(t),float(-t)) for t in ts]
    else: raise ValueError
    return loop_frames(pts, data_fn)

# BBH
bbh_cfg = core.BBHConfig()
bbh_oa = core.bbh_pos_observable(bbh_cfg,'x')
bbh_ob = core.bbh_pos_observable(bbh_cfg,'y')
bbh_refs = core.bbh_refs(bbh_cfg)

def bbh_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(theta_x, theta_y):
        h=core.build_bbh(cfg, theta_x, theta_y)
        evals,evecs=np.linalg.eigh(h)
        order = np.argsort(np.abs(evals))
        evals = evals[order]
        evecs = evecs[:, order]
        v=evecs[:,:4]
        abs_sorted = np.sort(np.abs(evals))
        gap = float(abs_sorted[4] - abs_sorted[3])
        oa_t,ob_t=core.compress_observables(v,oa,ob)
        frame,_=core.joint_diagonalize(v,oa_t,ob_t)
        frame=core.fix_column_phases(frame,refs)
        delta_a,delta_b=delta_metrics_from_compressed(oa_t,ob_t)
        return {'frame':frame,'gap':gap,'joint':core.jointness_metric(oa_t,ob_t),'deltaA':delta_a,'deltaB':delta_b,'oa_tilde':oa_t,'ob_tilde':ob_t}
    return _data
bbh_data=bbh_data_factory(bbh_cfg, bbh_oa, bbh_ob, bbh_refs)

def bbh_loop(kind, data_fn=bbh_data, n=81):
    ts=np.linspace(0,2*math.pi,n)
    if kind=='y': pts=[(0.0,float(t)) for t in ts]
    elif kind=='x': pts=[(float(t),0.0) for t in ts]
    elif kind=='diag': pts=[(float(t),float(t)) for t in ts]
    elif kind=='anti': pts=[(float(t),float(-t)) for t in ts]
    else: raise ValueError
    return loop_frames(pts, data_fn)

# BHZ
@dataclass(frozen=True)
class BHZConfig:
    Ly: int = 10
    M: float = 1.0
    B: float = 1.0
    A: float = 1.0
    lamR: float = 0.2
    h_e: float = 0.6
bhz_cfg=BHZConfig()

def bhz_h(kx, th_t, th_b, cfg=bhz_cfg):
    dim=4*cfg.Ly
    h=np.zeros((dim,dim),dtype=complex)
    h0=(cfg.M-4.0*cfg.B+2.0*cfg.B*math.cos(kx))*kron(TAU3,TAU0)
    h0+=cfg.A*math.sin(kx)*kron(TAU1,SZ)
    h0+=cfg.lamR*math.sin(kx)*kron(TAU1,SY)
    ty=cfg.B*kron(TAU3,TAU0)-0.5j*cfg.A*kron(TAU2,TAU0)+0.5j*cfg.lamR*kron(TAU1,SX)
    def sl(y): return slice(4*y,4*y+4)
    for y in range(cfg.Ly): h[sl(y),sl(y)] += h0
    for y in range(cfg.Ly-1):
        h[sl(y),sl(y+1)] += ty
        h[sl(y+1),sl(y)] += ty.conj().T
    edge_top = cfg.h_e*(math.cos(th_t)*kron(TAU0,SX)+math.sin(th_t)*kron(TAU0,SY))
    edge_bottom = cfg.h_e*(math.cos(th_b)*kron(TAU0,SX)+math.sin(th_b)*kron(TAU0,SY))
    h[sl(0), sl(0)] += edge_top
    h[sl(cfg.Ly - 1), sl(cfg.Ly - 1)] += edge_bottom
    return 0.5*(h+h.conj().T)

def bhz_side_observable(cfg=bhz_cfg):
    out=np.zeros((4*cfg.Ly,4*cfg.Ly),dtype=float)
    ys=np.linspace(-1.0,1.0,cfg.Ly)
    for y,val in enumerate(ys): out[4*y:4*y+4,4*y:4*y+4] = val*np.eye(4)
    return out

def bhz_spin_observable(cfg=bhz_cfg):
    out = np.zeros((4 * cfg.Ly, 4 * cfg.Ly), dtype=complex)
    blk = kron(TAU0, SZ)
    for y in range(cfg.Ly): out[4*y:4*y+4,4*y:4*y+4] = blk
    return out

def bhz_refs(cfg=bhz_cfg): return [1,0,4*(cfg.Ly-1)+1,4*(cfg.Ly-1)+0]

def bhz_data_factory(cfg, oa, ob, refs):
    @lru_cache(maxsize=None)
    def _data(kx, theta_t, theta_b):
        h=bhz_h(kx,theta_t,theta_b,cfg)
        evals,evecs=np.linalg.eigh(h)
        idx = np.argsort(np.abs(evals))[:4]
        idx = idx[np.argsort(evals[idx])]
        v=evecs[:,idx]
        abs_sorted = np.sort(np.abs(evals))
        gap = float(abs_sorted[4] - abs_sorted[3])
        oa_t,ob_t=core.compress_observables(v,oa,ob)
        frame,_=core.joint_diagonalize(v,oa_t,ob_t)
        frame=core.fix_column_phases(frame,refs)
        delta_a,delta_b=delta_metrics_from_compressed(oa_t,ob_t)
        return {'frame':frame,'gap':gap,'joint':core.jointness_metric(oa_t,ob_t),'deltaA':delta_a,'deltaB':delta_b,'oa_tilde':oa_t,'ob_tilde':ob_t}
    return _data
bhz_data=bhz_data_factory(bhz_cfg, bhz_side_observable(), bhz_spin_observable(), bhz_refs())

def bhz_loop(kind, kx=0.0, data_fn=bhz_data, n=81):
    ts=np.linspace(0,2*math.pi,n)
    pts=[]
    for t in ts:
        if kind=='bottom': pts.append((kx,0.0,float(t)))
        elif kind=='top': pts.append((kx,float(t),0.0))
        elif kind=='diag': pts.append((kx,float(t),float(t)))
        elif kind=='anti': pts.append((kx,float(t),float(-t)))
        else: raise ValueError
    return loop_frames(pts, data_fn)


def single_winding_eta_points(eta: float, n_main: int = 41, n_close: int = 13):
    pts=[]
    for t in np.linspace(0.0, 2.0*math.pi, n_main):
        pts.append((float(t%(2.0*math.pi)), float((eta*t)%(2.0*math.pi))))
    y_end = (eta*2.0*math.pi)%(2.0*math.pi)
    if abs(y_end)>1e-12 and abs(y_end-2.0*math.pi)>1e-12:
        if y_end <= math.pi:
            yvals=np.linspace(y_end,0.0,n_close)
            for y in yvals[1:]: pts.append((0.0,float(y)))
        else:
            yvals=np.linspace(y_end,2.0*math.pi,n_close//2+1)
            for y in yvals[1:]: pts.append((0.0,float(y%(2.0*math.pi))))
            for _ in range(max(0,n_close//2-1)): pts.append((0.0,0.0))
    return pts


def continuous_loop_scan_2d(model, data_fn, axis_loop, n_main, n_close, eta_grid):
    rows=[]
    for eta in eta_grid:
        frames, records, overlaps = loop_frames(single_winding_eta_points(float(eta), n_main=n_main, n_close=n_close), data_fn)
        u = core.berry_holonomy(frames)
        hm = holonomy_metrics(u, seed=17, ep_samples=256)
        rows.append({'model': model, 'eta_numeric': float(eta), 'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']), 'ep_stderr': float(hm['ep_stderr']), 'gap_min': float(min(float(r['gap']) for r in records)), 'joint_max': float(max(float(r['joint']) for r in records)), 'overlap_min': float(min(overlaps)), 'eta_axis2': False})
    frames, records, overlaps = axis_loop()
    u = core.berry_holonomy(frames)
    hm = holonomy_metrics(u, seed=17, ep_samples=256)
    rows.append({'model': model, 'eta_numeric': float('nan'), 'D_loc': float(hm['D_loc_strict']), 'ep_mean': float(hm['ep_mean']), 'ep_stderr': float(hm['ep_stderr']), 'gap_min': float(min(float(r['gap']) for r in records)), 'joint_max': float(max(float(r['joint']) for r in records)), 'overlap_min': float(min(overlaps)), 'eta_axis2': True})
    return pd.DataFrame(rows)

eta_grid = np.linspace(-2.0, 2.0, 61)
loop_family = pd.concat([
    continuous_loop_scan_2d('SSH', ssh_data, lambda: ssh_loop('right', n=81), n_main=41, n_close=13, eta_grid=eta_grid),
    continuous_loop_scan_2d('BBH', bbh_data, lambda: bbh_loop('y', n=81), n_main=31, n_close=11, eta_grid=eta_grid),
    continuous_loop_scan_2d('BHZ', lambda a,b: bhz_data(0.0, a, b), lambda: bhz_loop('bottom', 0.0, n=81), n_main=41, n_close=13, eta_grid=eta_grid),
], ignore_index=True)
loop_family.to_csv(DATA / 'continuous_loop_scan.csv', index=False)
print('Wrote continuous_loop_scan.csv.')
