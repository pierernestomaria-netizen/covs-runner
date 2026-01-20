#!/usr/bin/env python3
# RUN B - LNPR Test runner (ASCII-only)

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-6
V_MIN = 0.05
PREREG_SENSORS = ["s2","s3","s4","s7","s11","s12","s15"]
LOG_PATH = "logs/run_b.log"
OUTDIR = "results"

def setup_logging(log_path: str):
    log_dir = os.path.dirname(log_path) if os.path.dirname(log_path) else "."
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("run_b")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def load_fd001(path: str) -> pd.DataFrame:
    cols = ["unit","cycle","setting1","setting2","setting3"] + ["s"+str(i) for i in range(1,22)]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)

def minmax_unit_column(x: np.ndarray) -> Tuple[np.ndarray,bool]:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx-mn < EPS:
        return np.full_like(x, np.nan, dtype=float), False
    return (x-mn)/max(mx-mn, EPS), True

def per_unit_minmax(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray,bool]:
    arr = np.empty((len(df), len(cols)), dtype=float)
    for j,c in enumerate(cols):
        norm, ok = minmax_unit_column(df[c].values.astype(float))
        if not ok:
            return np.full_like(arr, np.nan), False
        arr[:,j] = norm
    return arr, True

def compute_V_t(arr: np.ndarray) -> np.ndarray:
    V = 1.0 - np.nanmean(arr, axis=1)
    V0 = V[0] if len(V)>0 else np.nan
    V = V / max(V0 if V0>0 else EPS, EPS)
    return np.clip(V, EPS, 1.0)

def fit_log_decay(V: np.ndarray, t: np.ndarray):
    if np.any(V<=0) or np.any(np.isnan(V)):
        return None, None
    y = np.log(V)
    a,b = np.polyfit(t.astype(float), y, 1)
    yhat = a*t + b
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 0.0
    return float(-a), float(r2)

def finite_collapse(V: np.ndarray):
    idx = np.where(V<=V_MIN)[0]
    return (idx.size>0, int(idx[0]) if idx.size>0 else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    args = ap.parse_args()

    logger = setup_logging(LOG_PATH)

    if not os.path.isfile(args.train):
        logger.error("Missing train file")
        sys.exit(2)

    df = load_fd001(args.train)

    for s in PREREG_SENSORS:
        if s not in df.columns:
            logger.error("Missing preregistered sensor: %s" % s)
            sys.exit(2)

    records = []
    for u in sorted(df["unit"].unique()):
        df_u = df[df["unit"]==u].sort_values("cycle")
        arr, ok = per_unit_minmax(df_u, PREREG_SENSORS)
        if not ok:
            records.append({"unit":int(u),"valid":False})
            continue
        V = compute_V_t(arr)
        gamma, r2 = fit_log_decay(V, np.arange(len(V)))
        collapse, tstar = finite_collapse(V)
        records.append({"unit":int(u),"valid":True,"gamma":gamma,"r2":r2,"t_star":tstar})

    valid = [r for r in records if r.get("valid")]
    gammas = np.array([r["gamma"] for r in valid if r["gamma"] is not None])
    r2s = np.array([r["r2"] for r in valid if r["r2"] is not None])
    collapses = np.array([1 if r.get("t_star") is not None else 0 for r in valid])

    prop_gamma = float(np.sum(gammas>0))/len(valid) if len(valid)>0 else 0.0
    med_r2 = float(np.median(r2s)) if len(r2s)>0 else float("nan")
    prop_collapse = float(np.sum(collapses))/len(valid) if len(valid)>0 else 0.0

    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(OUTDIR,"run_b_units.csv"), index=False)
    summary = {
        "prop_gamma_positive": prop_gamma,
        "median_r2": med_r2,
        "prop_collapse": prop_collapse,
        "result": "PASS" if (prop_gamma>=0.70 and med_r2>=0.50 and prop_collapse>=0.90) else "FAIL"
    }
    with open(os.path.join(OUTDIR,"run_b_summary.json"),"w") as fh:
        json.dump(summary, fh, indent=2)

    sys.exit(0 if summary["result"]=="PASS" else 3)

if __name__ == "__main__":
    main()
