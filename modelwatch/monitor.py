"""Baseline profiling and windowed drift monitoring."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import stats


@dataclass
class MonitorConfig:
    window_size: int = 500
    n_bins: int = 10
    baseline_sample_size: int = 2000
    psi_threshold: float = 0.2
    prediction_psi_threshold: float = 0.2
    ks_stat_threshold: float = 0.1
    ks_pvalue_threshold: float = 1e-3
    missing_rate_delta: float = 0.10
    novelty_rate_threshold: float = 0.01
    out_of_range_rate_threshold: float = 0.01
    rare_category_threshold: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorConfig":
        return cls(**d)


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


class BaselineProfile:
    """Per-feature training-window profile: quantile-bin histograms, ranges,
    category proportions, missing rates, a seeded raw sample for KS tests,
    and the prediction-score distribution."""

    def __init__(self, numeric: dict, categorical: dict, prediction: dict,
                 n_rows: int):
        self.numeric = numeric
        self.categorical = categorical
        self.prediction = prediction
        self.n_rows = n_rows

    @classmethod
    def from_frame(cls, df: pd.DataFrame, numeric_features: List[str],
                   categorical_features: List[str], scores: np.ndarray,
                   config: Optional[MonitorConfig] = None,
                   seed: int = 0) -> "BaselineProfile":
        config = config or MonitorConfig()
        rng = np.random.default_rng(seed)
        numeric = {}
        for f in numeric_features:
            values = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
            clean = values[~np.isnan(values)]
            edges = stats.quantile_bin_edges(clean, n_bins=config.n_bins)
            sample = clean if clean.size <= config.baseline_sample_size else \
                rng.choice(clean, size=config.baseline_sample_size, replace=False)
            numeric[f] = {
                "edges": edges.tolist(),
                "props": stats.bin_proportions(clean, edges).tolist(),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "missing_rate": _missing_rate(df[f]),
                "sample": np.sort(sample).tolist(),
            }
        categorical = {}
        for f in categorical_features:
            series = df[f].astype(object)
            non_missing = series.dropna().astype(str)
            full = {str(k): float(v) for k, v in
                    non_missing.value_counts(normalize=True).items()}
            rare = sorted(k for k, v in full.items()
                          if v < config.rare_category_threshold)
            props = {k: v for k, v in full.items() if k not in rare}
            props["__rare__"] = float(sum(full[k] for k in rare))
            categorical[f] = {
                "props": props,
                "rare_categories": rare,
                "categories": sorted(full),  # all train categories (novelty)
                "missing_rate": _missing_rate(series),
            }
        scores = np.asarray(scores, dtype=float)
        pred_edges = stats.quantile_bin_edges(scores, n_bins=config.n_bins)
        prediction = {
            "edges": pred_edges.tolist(),
            "props": stats.bin_proportions(scores, pred_edges).tolist(),
        }
        return cls(numeric, categorical, prediction, n_rows=len(df))

    def to_dict(self) -> dict:
        return {"numeric": self.numeric, "categorical": self.categorical,
                "prediction": self.prediction, "n_rows": self.n_rows}

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "BaselineProfile":
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["numeric"], d["categorical"], d["prediction"], d["n_rows"])


@dataclass
class WindowReport:
    window_index: int
    n_rows: int
    features: Dict[str, dict]
    prediction_psi: float
    verdict: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class WindowMonitor:
    """Sliding (tumbling) window monitor: buffers scored rows and evaluates
    each full window of `window_size` rows against the baseline profile."""

    def __init__(self, baseline: BaselineProfile,
                 config: Optional[MonitorConfig] = None):
        self.baseline = baseline
        self.config = config or MonitorConfig()
        self._rows: List[dict] = []
        self._scores: List[float] = []
        self.reports: List[WindowReport] = []
        self._window_count = 0

    @property
    def buffered_rows(self) -> int:
        return len(self._rows)

    def add_row(self, features: dict, score: float) -> Optional[WindowReport]:
        """Buffer one scored row; returns a WindowReport when a window closes."""
        self._rows.append(features)
        self._scores.append(float(score))
        if len(self._rows) >= self.config.window_size:
            df = pd.DataFrame(self._rows)
            scores = np.array(self._scores)
            self._rows, self._scores = [], []
            report = self.evaluate(df, scores)
            return report
        return None

    def latest_report(self) -> Optional[WindowReport]:
        return self.reports[-1] if self.reports else None

    def evaluate(self, df: pd.DataFrame, scores: np.ndarray) -> WindowReport:
        cfg = self.config
        features: Dict[str, dict] = {}
        reasons: List[str] = []

        for f, base in self.baseline.numeric.items():
            entry: dict = {"type": "numeric"}
            flags: List[str] = []
            if f in df.columns:
                col = pd.to_numeric(df[f], errors="coerce")
                values = col.to_numpy(dtype=float)
            else:
                values = np.full(len(df), np.nan)
            missing_rate = float(np.isnan(values).mean())
            clean = values[~np.isnan(values)]
            entry["missing_rate"] = missing_rate
            if missing_rate > base["missing_rate"] + cfg.missing_rate_delta:
                flags.append("missing_spike")
            if clean.size:
                edges = np.array(base["edges"])
                entry["psi"] = stats.psi_from_proportions(
                    np.array(base["props"]), stats.bin_proportions(clean, edges))
                ks_stat, ks_p = stats.ks_2samp(np.array(base["sample"]), clean)
                entry["ks_stat"], entry["ks_pvalue"] = ks_stat, ks_p
                entry["out_of_range_rate"] = float(
                    ((clean < base["min"]) | (clean > base["max"])).mean())
                if entry["psi"] > cfg.psi_threshold:
                    flags.append("psi")
                if ks_p < cfg.ks_pvalue_threshold and ks_stat > cfg.ks_stat_threshold:
                    flags.append("ks")
                if entry["out_of_range_rate"] > cfg.out_of_range_rate_threshold:
                    flags.append("out_of_range")
            entry["flags"] = flags
            features[f] = entry
            reasons.extend(f"{f}:{flag}" for flag in flags)

        for f, base in self.baseline.categorical.items():
            entry = {"type": "categorical"}
            flags = []
            raw = df[f].astype(object).tolist() if f in df.columns \
                else [None] * len(df)
            n_missing = sum(1 for v in raw if v is None or
                            (isinstance(v, float) and math.isnan(v)))
            missing_rate = n_missing / max(len(raw), 1)
            entry["missing_rate"] = missing_rate
            if missing_rate > base["missing_rate"] + cfg.missing_rate_delta:
                flags.append("missing_spike")
            known = set(base["categories"])
            present = [str(v) for v in raw if v is not None and
                       not (isinstance(v, float) and math.isnan(v))]
            novelty = (sum(1 for v in present if v not in known) /
                       max(len(present), 1))
            entry["novelty_rate"] = novelty
            entry["psi"] = stats.categorical_psi(
                base["props"], raw, base["rare_categories"])
            if novelty > cfg.novelty_rate_threshold:
                flags.append("category_novelty")
            if entry["psi"] > cfg.psi_threshold:
                flags.append("psi")
            entry["flags"] = flags
            features[f] = entry
            reasons.extend(f"{f}:{flag}" for flag in flags)

        pred_edges = np.array(self.baseline.prediction["edges"])
        prediction_psi = stats.psi_from_proportions(
            np.array(self.baseline.prediction["props"]),
            stats.bin_proportions(np.asarray(scores, dtype=float), pred_edges))
        if prediction_psi > cfg.prediction_psi_threshold:
            reasons.append("prediction:psi")

        report = WindowReport(
            window_index=self._window_count, n_rows=len(df), features=features,
            prediction_psi=float(prediction_psi), verdict=bool(reasons),
            reasons=reasons)
        self._window_count += 1
        self.reports.append(report)
        return report
