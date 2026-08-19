"""Seeded drift-scenario generator.

Replays held-out UCI Adult traffic rows as windows and injects LABELED drift
episodes. Ground truth is therefore INJECTED/SYNTHETIC drift applied to real
data, not real-world drift.

Scenario types:
  covariate_shift    demographic resampling (rows with education-num >= 13
                     weighted 4x)
  feature_scaling    hours-per-week scaled by 1.5x
  missing_spike      30% of 'occupation' values nulled
  category_novelty   10% of 'workclass' replaced with a novel category
  prediction_shift   resampling weighted by model score^2 (shifts the score
                     distribution toward high-probability rows)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

SCENARIOS = ("covariate_shift", "feature_scaling", "missing_spike",
             "category_novelty", "prediction_shift")


@dataclass
class Episode:
    start: int  # first drifted window index (inclusive)
    end: int    # last drifted window index (inclusive)
    scenario: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WindowSpec:
    index: int
    is_drift: bool
    scenario: Optional[str]


class ScenarioGenerator:
    """Deterministic (seeded) stream of monitoring windows with ground truth."""

    def __init__(self, traffic: pd.DataFrame, scores: np.ndarray,
                 window_size: int = 500, n_windows: int = 240, seed: int = 7):
        self.traffic = traffic.reset_index(drop=True)
        self.scores = np.asarray(scores, dtype=float)
        self.window_size = window_size
        self.n_windows = n_windows
        self.seed = seed
        self.specs, self.episodes = self._build_schedule()

    def _build_schedule(self) -> Tuple[List[WindowSpec], List[Episode]]:
        rng = np.random.default_rng(self.seed)
        specs: List[WindowSpec] = []
        episodes: List[Episode] = []
        scenario_queue: List[str] = []
        i = 0
        while i < self.n_windows:
            for _ in range(int(rng.integers(3, 7))):  # normal stretch
                if i >= self.n_windows:
                    break
                specs.append(WindowSpec(i, False, None))
                i += 1
            if i >= self.n_windows:
                break
            if not scenario_queue:
                scenario_queue = list(rng.permutation(SCENARIOS))
            scenario = scenario_queue.pop(0)
            length = int(rng.integers(2, 5))
            start = i
            for _ in range(length):
                if i >= self.n_windows:
                    break
                specs.append(WindowSpec(i, True, scenario))
                i += 1
            episodes.append(Episode(start, i - 1, scenario))
        return specs, episodes

    def _sample(self, rng: np.random.Generator,
                weights: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(self.traffic)
        if weights is None:
            return rng.choice(n, size=self.window_size, replace=True)
        p = weights / weights.sum()
        return rng.choice(n, size=self.window_size, replace=True, p=p)

    def windows(self) -> Iterator[Tuple[WindowSpec, pd.DataFrame]]:
        """Yield (spec, window_frame). Same seed -> identical stream."""
        rng = np.random.default_rng(self.seed + 1)
        edu = pd.to_numeric(self.traffic["education-num"],
                            errors="coerce").to_numpy(dtype=float)
        cov_weights = np.where(edu >= 13, 4.0, 1.0)
        pred_weights = (self.scores + 0.02) ** 2
        for spec in self.specs:
            if spec.scenario == "covariate_shift":
                idx = self._sample(rng, cov_weights)
            elif spec.scenario == "prediction_shift":
                idx = self._sample(rng, pred_weights)
            else:
                idx = self._sample(rng)
            frame = self.traffic.iloc[idx].reset_index(drop=True)
            if spec.scenario == "feature_scaling":
                frame["hours-per-week"] = (
                    pd.to_numeric(frame["hours-per-week"]) * 1.5)
            elif spec.scenario == "missing_spike":
                mask = rng.random(len(frame)) < 0.30
                frame.loc[mask, "occupation"] = np.nan
            elif spec.scenario == "category_novelty":
                mask = rng.random(len(frame)) < 0.10
                frame.loc[mask, "workclass"] = "GigEconomyPlatform"
            yield spec, frame
