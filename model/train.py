"""Train the reference model on UCI Adult, persist artifacts + baseline profile.

Outputs (all regeneratable, gitignored except results/):
  model/artifacts/model.pkl           healthy LogisticRegression pipeline
  model/artifacts/degraded_model.pkl  label-flipped retrain (must fail gates)
  model/artifacts/baseline.json       training-window baseline profile
  model/artifacts/meta.json           feature schema + split info
  data/splits/{train,golden,traffic}.csv
  results/model_eval.json             measured metrics (committed)

Run: .venv/bin/python model/train.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modelwatch.monitor import BaselineProfile, MonitorConfig  # noqa: E402
from modelwatch.stats import expected_calibration_error  # noqa: E402

SEED = 42
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
           "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
FEATURES = NUMERIC + CATEGORICAL
DEGRADED_FLIP_FRACTION = 0.40


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant",
                                         fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             CATEGORICAL),
        ])),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def evaluate(model, x: pd.DataFrame, y: np.ndarray) -> dict:
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "ece": float(expected_calibration_error(y, probs)),
        "slices": {},
    }
    for feature in ("sex", "race"):
        for value in sorted(x[feature].astype(str).unique()):
            mask = (x[feature].astype(str) == value).to_numpy()
            out["slices"][f"{feature}={value}"] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y[mask], preds[mask])),
            }
    return out


def main() -> None:
    os.makedirs(os.path.join(ROOT, "model", "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "splits"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    print("Fetching UCI Adult (OpenML adult v2, cached under data/)...")
    x, y_raw = fetch_openml("adult", version=2, as_frame=True, return_X_y=True,
                            data_home=os.path.join(ROOT, "data", "openml_cache"),
                            parser="auto")
    x = x[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype(object)
    y = (y_raw == ">50K").astype(int).to_numpy()

    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.4, random_state=SEED, stratify=y)
    x_golden, x_traffic, y_golden, y_traffic = train_test_split(
        x_rest, y_rest, test_size=0.5, random_state=SEED, stratify=y_rest)
    print(f"splits: train={len(x_train)} golden={len(x_golden)} "
          f"traffic={len(x_traffic)}")

    model = build_pipeline()
    model.fit(x_train, y_train)

    rng = np.random.default_rng(SEED)
    flip = rng.random(len(y_train)) < DEGRADED_FLIP_FRACTION
    y_flipped = np.where(flip, 1 - y_train, y_train)
    degraded = build_pipeline()
    degraded.fit(x_train, y_flipped)

    train_scores = model.predict_proba(x_train)[:, 1]
    baseline = BaselineProfile.from_frame(
        x_train, NUMERIC, CATEGORICAL, train_scores,
        config=MonitorConfig(), seed=SEED)

    art = os.path.join(ROOT, "model", "artifacts")
    with open(os.path.join(art, "model.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    with open(os.path.join(art, "degraded_model.pkl"), "wb") as fh:
        pickle.dump(degraded, fh)
    baseline.save(os.path.join(art, "baseline.json"))
    with open(os.path.join(art, "meta.json"), "w") as fh:
        json.dump({"numeric_features": NUMERIC,
                   "categorical_features": CATEGORICAL,
                   "seed": SEED,
                   "degraded_flip_fraction": DEGRADED_FLIP_FRACTION,
                   "splits": {"train": len(x_train), "golden": len(x_golden),
                              "traffic": len(x_traffic)}}, fh, indent=2)

    for name, frame, target in (("train", x_train, y_train),
                                ("golden", x_golden, y_golden),
                                ("traffic", x_traffic, y_traffic)):
        out = frame.copy()
        out["__label__"] = target
        out.to_csv(os.path.join(ROOT, "data", "splits", f"{name}.csv"),
                   index=False)

    results = {
        "dataset": "UCI Adult (OpenML 'adult' version 2, 48842 rows)",
        "model": "LogisticRegression (median/constant impute, standardize, "
                 "one-hot, max_iter=2000)",
        "degraded_model": f"same pipeline retrained with "
                          f"{DEGRADED_FLIP_FRACTION:.0%} labels flipped "
                          f"(seed {SEED})",
        "evaluated_on": "golden split (held out from training)",
        "healthy": evaluate(model, x_golden, y_golden),
        "degraded": evaluate(degraded, x_golden, y_golden),
    }
    with open(os.path.join(ROOT, "results", "model_eval.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: results[k] for k in ("healthy", "degraded")}, indent=2))
    print("artifacts written to model/artifacts/")


if __name__ == "__main__":
    main()
