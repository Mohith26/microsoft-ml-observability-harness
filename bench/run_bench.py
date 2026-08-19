"""Benchmarks: drift-engine throughput and API latency (best of 5).

HONESTY TAG: API latencies are measured with FastAPI's in-process TestClient
(no network socket, no uvicorn); they exclude real network/serialization-over-
wire cost. Drift-engine throughput is wall-clock rows/sec through
WindowMonitor.evaluate.

Run: .venv/bin/python bench/run_bench.py  ->  results/bench.json
"""
from __future__ import annotations

import json
import os
import pickle
import statistics
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from modelwatch.api import create_app  # noqa: E402
from modelwatch.monitor import BaselineProfile, MonitorConfig, WindowMonitor  # noqa: E402

WINDOW_SIZE = 500
ENGINE_WINDOWS = 20   # rows per engine run = 20 * 500 = 10,000
SCORE_REQUESTS = 1000
DRIFT_REQUESTS = 200
RUNS = 5


def pctl(values, q):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))
    return values[idx]


def bench_engine(baseline, traffic, scores):
    monitor = WindowMonitor(baseline, MonitorConfig(window_size=WINDOW_SIZE))
    rng = np.random.default_rng(0)
    windows = []
    for _ in range(ENGINE_WINDOWS):
        idx = rng.choice(len(traffic), WINDOW_SIZE, replace=True)
        windows.append((traffic.iloc[idx].reset_index(drop=True), scores[idx]))
    runs = []
    for _ in range(RUNS):
        start = time.perf_counter()
        for frame, s in windows:
            monitor.evaluate(frame, s)
        elapsed = time.perf_counter() - start
        runs.append(ENGINE_WINDOWS * WINDOW_SIZE / elapsed)
    return {"rows_per_run": ENGINE_WINDOWS * WINDOW_SIZE,
            "runs_rows_per_sec": [round(r, 1) for r in runs],
            "best_rows_per_sec": round(max(runs), 1),
            "median_rows_per_sec": round(statistics.median(runs), 1)}


def bench_endpoint(client, method, url, payloads, n, runs):
    all_runs = []
    for _ in range(runs):
        latencies = []
        for i in range(n):
            body = payloads[i % len(payloads)] if payloads else None
            start = time.perf_counter()
            if method == "POST":
                resp = client.post(url, json=body)
            else:
                resp = client.get(url)
            latencies.append((time.perf_counter() - start) * 1000.0)
            assert resp.status_code == 200, resp.text
        all_runs.append({"p50_ms": round(pctl(latencies, 50), 3),
                         "p95_ms": round(pctl(latencies, 95), 3)})
    best = min(all_runs, key=lambda r: r["p50_ms"])
    return {"requests_per_run": n, "runs": all_runs, "best": best}


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
    scores = model.predict_proba(traffic)[:, 1]

    engine = bench_engine(baseline, traffic, scores)

    app = create_app()
    client = TestClient(app)
    rows = traffic.head(600).where(pd.notna(traffic.head(600)), None)
    payloads = [{"features": r} for r in rows.to_dict(orient="records")]
    client.post("/score", json=payloads[0])  # warm-up
    client.get("/drift")
    score_bench = bench_endpoint(client, "POST", "/score", payloads,
                                 SCORE_REQUESTS, RUNS)
    # fill at least one full window so /drift returns a real report
    for p in payloads[:WINDOW_SIZE]:
        client.post("/score", json=p)
    drift_bench = bench_endpoint(client, "GET", "/drift", None,
                                 DRIFT_REQUESTS, RUNS)

    out = {
        "honesty_tag": "In-process FastAPI TestClient latencies (no network "
                       "socket); drift-engine throughput is wall-clock over "
                       "pre-materialized windows. Best of 5 runs reported.",
        "machine": "local dev machine, CPU only",
        "drift_engine": engine,
        "score_endpoint": score_bench,
        "drift_endpoint": drift_bench,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "bench.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
