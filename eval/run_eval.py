"""Drift-detection evaluation against ground-truth injected drift episodes.

HONESTY TAG: precision/recall are measured over LABELED, INJECTED (synthetic)
drift episodes replayed over real held-out UCI Adult rows — not real-world
drift.

Run: .venv/bin/python eval/run_eval.py   ->  results/drift_eval.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.scenarios import SCENARIOS, ScenarioGenerator  # noqa: E402
from modelwatch.monitor import BaselineProfile, MonitorConfig, WindowMonitor  # noqa: E402

SEED = 7
N_WINDOWS = 240
WINDOW_SIZE = 500


def alarm_segments(verdicts):
    """Group consecutive flagged windows into alarm segments [(start, end)]."""
    segments, start = [], None
    for i, v in enumerate(verdicts):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(verdicts) - 1))
    return segments


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    art = os.path.join(ROOT, "model", "artifacts")
    with open(os.path.join(art, "model.pkl"), "rb") as fh:
        model = pickle.load(fh)
    with open(os.path.join(art, "meta.json")) as fh:
        meta = json.load(fh)
    baseline = BaselineProfile.load(os.path.join(art, "baseline.json"))
    traffic = pd.read_csv(os.path.join(ROOT, "data", "splits", "traffic.csv"))
    traffic = traffic.drop(columns=["__label__"])
    for f in meta["categorical_features"]:
        traffic[f] = traffic[f].astype(object)

    pool_scores = model.predict_proba(traffic)[:, 1]
    gen = ScenarioGenerator(traffic, pool_scores, window_size=WINDOW_SIZE,
                            n_windows=N_WINDOWS, seed=SEED)
    config = MonitorConfig(window_size=WINDOW_SIZE)
    monitor = WindowMonitor(baseline, config)

    truths, verdicts, scenarios, reasons_log = [], [], [], []
    for spec, frame in gen.windows():
        scores = model.predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        truths.append(spec.is_drift)
        verdicts.append(report.verdict)
        scenarios.append(spec.scenario)
        reasons_log.append(report.reasons)

    truths_a = np.array(truths)
    verdicts_a = np.array(verdicts)

    tp = int((truths_a & verdicts_a).sum())
    fp = int((~truths_a & verdicts_a).sum())
    fn = int((truths_a & ~verdicts_a).sum())
    tn = int((~truths_a & ~verdicts_a).sum())
    window_overall = prf(tp, fp, fn)
    window_overall["tn"] = tn

    per_scenario_window = {}
    for s in SCENARIOS:
        mask = np.array([sc == s for sc in scenarios])
        s_tp = int((verdicts_a & mask).sum())
        s_fn = int((~verdicts_a & mask).sum())
        per_scenario_window[s] = {
            "windows": int(mask.sum()),
            "detected": s_tp,
            "recall": round(s_tp / mask.sum(), 4) if mask.sum() else None,
        }

    segments = alarm_segments(verdicts)
    ep_ranges = [(e.start, e.end, e.scenario) for e in gen.episodes]

    def overlaps(seg, ep):
        return seg[0] <= ep[1] and ep[0] <= seg[1]

    ep_tp = sum(1 for e in ep_ranges if any(overlaps(s, e) for s in segments))
    ep_fn = len(ep_ranges) - ep_tp
    seg_fp = sum(1 for s in segments
                 if not any(overlaps(s, e) for e in ep_ranges))
    episode_overall = prf(len(segments) - seg_fp, seg_fp, ep_fn)
    episode_overall["note"] = ("tp = alarm segments overlapping a true "
                               "episode; an episode counts detected if any "
                               "alarm overlaps it")

    per_scenario_episode = defaultdict(lambda: {"episodes": 0, "detected": 0})
    for e in ep_ranges:
        d = per_scenario_episode[e[2]]
        d["episodes"] += 1
        if any(overlaps(s, e) for s in segments):
            d["detected"] += 1
    for s, d in per_scenario_episode.items():
        d["recall"] = round(d["detected"] / d["episodes"], 4)

    out = {
        "honesty_tag": "Ground truth is INJECTED/synthetic drift applied to "
                       "real held-out UCI Adult rows; this is not real-world "
                       "drift. False positives are windows sampled i.i.d. "
                       "from the same traffic pool that were still flagged.",
        "seed": SEED,
        "n_windows": N_WINDOWS,
        "window_size": WINDOW_SIZE,
        "n_drift_windows": int(truths_a.sum()),
        "n_normal_windows": int((~truths_a).sum()),
        "n_episodes": len(ep_ranges),
        "monitor_config": config.to_dict(),
        "window_level": {"overall": window_overall,
                         "per_scenario_recall": per_scenario_window},
        "episode_level": {"overall": episode_overall,
                          "per_scenario": dict(per_scenario_episode)},
        "false_positive_window_reasons": [
            {"window": int(i), "reasons": reasons_log[i]}
            for i in np.where(~truths_a & verdicts_a)[0]],
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "drift_eval.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["window_level"], indent=2))
    print(json.dumps(out["episode_level"], indent=2))


if __name__ == "__main__":
    main()
