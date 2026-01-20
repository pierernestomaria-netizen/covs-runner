#!/usr/bin/env python3
# RUN A - NASA C-MAPSS FD001 benchmark runner (ASCII-only)

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

DEFAULT_SEED = 42
VAR_THRESH = 1e-10
DDOF = 0

def setup_logging(log_path: str):
    log_dir = os.path.dirname(log_path) if os.path.dirname(log_path) else "."
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("run_a")
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

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def load_fd001(path: str) -> pd.DataFrame:
    cols = ["unit","cycle","setting1","setting2","setting3"] + ["s"+str(i) for i in range(1,22)]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)

def compute_rul_train(df: pd.DataFrame) -> pd.Series:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).astype(float)

def read_rul_file(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:,0].astype(float).values

def remove_low_variance_features(df_train: pd.DataFrame, feature_cols: List[str], logger) -> Tuple[List[str], List[str]]:
    variances = df_train[feature_cols].var(axis=0, ddof=DDOF)
    keep, removed = [], []
    for c in feature_cols:
        v = variances[c]
        if np.isnan(v) or v < VAR_THRESH:
            removed.append(c)
        else:
            keep.append(c)
    logger.info("Final feature set (%d cols): %s" % (len(keep), str(keep)))
    return keep, removed

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d/13.0)-1.0, np.exp(d/10.0)-1.0)
    return float(np.sum(score))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--rul", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--log", default="logs/run_a.log")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    logger = setup_logging(args.log)
    np.random.seed(int(args.seed))

    for p in (args.train, args.test, args.rul):
        if not os.path.isfile(p):
            logger.error("Missing file: %s" % p)
            sys.exit(2)

    df_train = load_fd001(args.train)
    df_test = load_fd001(args.test)
    rul = read_rul_file(args.rul)

    df_train["RUL"] = compute_rul_train(df_train)

    features = ["setting1","setting2","setting3"] + ["s"+str(i) for i in range(1,22)]
    keep, removed = remove_low_variance_features(df_train, features, logger)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[keep].values)
    X_test_all = scaler.transform(df_test[keep].values)

    model = Ridge()
    model.fit(X_train, df_train["RUL"].values)

    last_idx = df_test.groupby("unit")["cycle"].idxmax().sort_index().values
    df_last = df_test.loc[last_idx].sort_values("unit").reset_index(drop=True)

    if len(rul) != len(df_last):
        logger.error("RUL length mismatch")
        sys.exit(2)

    y_pred = model.predict(X_test_all[last_idx,:])
    rmse = float(np.sqrt(mean_squared_error(rul, y_pred)))
    score = nasa_score(rul, y_pred)

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame({"unit":df_last["unit"],"y_true":rul,"y_pred":y_pred}).to_csv(os.path.join(args.outdir,"run_a_predictions.csv"), index=False)

    summary = {
        "rmse": rmse,
        "nasa_score": score,
        "features_used": keep,
        "removed_features": removed,
        "sha256": {
            "train": sha256_of_file(args.train),
            "test": sha256_of_file(args.test),
            "rul": sha256_of_file(args.rul)
        }
    }
    with open(os.path.join(args.outdir,"run_a_summary.json"),"w") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(args.outdir,"run_a_model_coeffs.json"),"w") as fh:
        json.dump({"intercept": float(model.intercept_), "coeffs": model.coef_.tolist(), "features": keep}, fh, indent=2)

    sys.exit(0)

if __name__ == "__main__":
    main()
