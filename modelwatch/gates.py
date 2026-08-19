"""Golden-dataset regression gates: accuracy, ROC-AUC, calibration (ECE),
and per-slice accuracy (responsible-AI style slice monitoring)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .stats import expected_calibration_error


@dataclass
class GateThresholds:
    min_accuracy: float = 0.84
    min_auc: float = 0.88
    max_ece: float = 0.06
    min_slice_accuracy: float = 0.75
    slice_features: Tuple[str, ...] = ("sex", "race")
    min_slice_size: int = 50

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slice_features"] = list(d["slice_features"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GateThresholds":
        d = dict(d)
        d["slice_features"] = tuple(d.get("slice_features", ("sex", "race")))
        return cls(**d)


@dataclass
class GateReport:
    gates: List[dict]
    slices: Dict[str, dict] = field(default_factory=dict)
    overall_passed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def run_gates(model, golden_x: pd.DataFrame, golden_y: np.ndarray,
              thresholds: GateThresholds = None) -> GateReport:
    """Evaluate a model against the golden dataset and threshold every gate.

    Slices smaller than `min_slice_size` are reported but not gated
    (small-sample accuracy is too noisy to fail a release on).
    """
    thresholds = thresholds or GateThresholds()
    y = np.asarray(golden_y, dtype=int)
    probs = model.predict_proba(golden_x)[:, 1]
    preds = (probs >= 0.5).astype(int)

    accuracy = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))
    ece = float(expected_calibration_error(y, probs))

    gates = [
        {"name": "accuracy", "value": accuracy,
         "threshold": thresholds.min_accuracy, "direction": ">=",
         "passed": accuracy >= thresholds.min_accuracy},
        {"name": "roc_auc", "value": auc,
         "threshold": thresholds.min_auc, "direction": ">=",
         "passed": auc >= thresholds.min_auc},
        {"name": "ece", "value": ece,
         "threshold": thresholds.max_ece, "direction": "<=",
         "passed": ece <= thresholds.max_ece},
    ]

    slices: Dict[str, dict] = {}
    for feature in thresholds.slice_features:
        if feature not in golden_x.columns:
            continue
        for value, idx in golden_x.groupby(golden_x[feature].astype(str)).groups.items():
            loc = golden_x.index.get_indexer(idx)
            slice_acc = float(accuracy_score(y[loc], preds[loc]))
            gated = len(loc) >= thresholds.min_slice_size
            passed = (not gated) or slice_acc >= thresholds.min_slice_accuracy
            slices[f"{feature}={value}"] = {
                "n": int(len(loc)), "accuracy": slice_acc,
                "gated": gated, "passed": passed}
            if gated:
                gates.append({
                    "name": f"slice_accuracy[{feature}={value}]",
                    "value": slice_acc,
                    "threshold": thresholds.min_slice_accuracy,
                    "direction": ">=", "passed": passed})

    return GateReport(gates=gates, slices=slices,
                      overall_passed=all(g["passed"] for g in gates))
