#!/usr/bin/env python3
"""Independent re-check of the wider audit: recount per-window violations on three recordings
(5% saturation) and confirm a sample of them by direct summation.
Usage: python3 -I audit/verify_wide.py DATA_DIR"""
import os, sys, csv, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'audit'))
import mdms_ref as M
from wide_audit import recordings, masks_for, SEG0
rows={(r['recording'],r['mask'],r['mode']):r for r in csv.DictReader(open(os.path.join(ROOT,'results/wide/wide_audit_n3000_m100.csv')))}
recs=recordings(sys.argv[1])
rng=np.random.default_rng(1)
for name in ['1730_OR@6_21_DE12','1772_IR_14_DE12','1750_B_7_DE12']:
    Tc=recs[name][SEG0:SEG0+3000]; m=100
    mask=masks_for(Tc,m,0)['clip_q95']
    T_in=np.where(mask,Tc,0.0)
    ws=M.window_stats(T_in,mask,m,'window'); os_=M.oracle_stats(Tc,m)
    L=ws['L']; kmin=M.excl_zone(m)+1
    viol=[]
    for k in range(kmin,L):
        d2=M.diagonal_lb_d2(ws,k); d2o=M.oracle_diagonal_d2(os_,k)
        v=np.flatnonzero(np.isfinite(d2)&np.isfinite(d2o)&(d2>d2o*(1+1e-9)+1e-9))
        viol += [(int(i),int(i+k)) for i in v]
    rep=int(rows[(name,'clip_q95','window')]['pair_viol'])
    samp=[viol[i] for i in rng.choice(len(viol),min(300,len(viol)),replace=False)] if viol else []
    conf=sum(M.pair_lb_d2(ws,i,j) > M.oracle_pair_d2(Tc,m,i,j)*(1+1e-9)+1e-9 for i,j in samp)
    print(name,'recount',len(viol),'reported',rep,'direct-sum confirmed',conf,'/',len(samp))
