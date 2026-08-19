import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modelwatch.monitor import BaselineProfile  # noqa: E402

ART = os.path.join(ROOT, "model", "artifacts")


def _ensure_artifacts():
    needed = [os.path.join(ART, f) for f in
              ("model.pkl", "degraded_model.pkl", "baseline.json",
               "meta.json")]
    needed.append(os.path.join(ROOT, "data", "splits", "golden.csv"))
    if all(os.path.exists(p) for p in needed):
        return
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "train.py")],
                   check=True)


@pytest.fixture(scope="session")
def artifacts():
    _ensure_artifacts()
    with open(os.path.join(ART, "model.pkl"), "rb") as fh:
        model = pickle.load(fh)
    with open(os.path.join(ART, "degraded_model.pkl"), "rb") as fh:
        degraded = pickle.load(fh)
    with open(os.path.join(ART, "meta.json")) as fh:
        meta = json.load(fh)
    baseline = BaselineProfile.load(os.path.join(ART, "baseline.json"))
    return {"model": model, "degraded": degraded, "meta": meta,
            "baseline": baseline}


def _load_split(name, meta):
    df = pd.read_csv(os.path.join(ROOT, "data", "splits", f"{name}.csv"))
    y = df.pop("__label__").to_numpy()
    for f in meta["categorical_features"]:
        df[f] = df[f].astype(object)
    return df, y


@pytest.fixture(scope="session")
def golden(artifacts):
    return _load_split("golden", artifacts["meta"])


@pytest.fixture(scope="session")
def traffic(artifacts):
    return _load_split("traffic", artifacts["meta"])


@pytest.fixture(scope="session")
def traffic_scores(artifacts, traffic):
    return artifacts["model"].predict_proba(traffic[0])[:, 1]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(99)
