"""Gate pass/fail matrix for the healthy vs degraded model.

Run: .venv/bin/python model/gates_report.py  ->  results/gates_matrix.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modelwatch.gates import GateThresholds, run_gates  # noqa: E402


def main() -> None:
    art = os.path.join(ROOT, "model", "artifacts")
    with open(os.path.join(art, "meta.json")) as fh:
        meta = json.load(fh)
    golden = pd.read_csv(os.path.join(ROOT, "data", "splits", "golden.csv"))
    y = golden.pop("__label__").to_numpy()
    for f in meta["categorical_features"]:
        golden[f] = golden[f].astype(object)

    out = {"thresholds": GateThresholds().to_dict(), "models": {}}
    for name, fname in (("healthy", "model.pkl"),
                        ("degraded", "degraded_model.pkl")):
        with open(os.path.join(art, fname), "rb") as fh:
            model = pickle.load(fh)
        report = run_gates(model, golden, y)
        out["models"][name] = report.to_dict()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "gates_matrix.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for name, rep in out["models"].items():
        failed = [g["name"] for g in rep["gates"] if not g["passed"]]
        print(f"{name}: overall_passed={rep['overall_passed']} "
              f"failed_gates={failed}")


if __name__ == "__main__":
    main()
