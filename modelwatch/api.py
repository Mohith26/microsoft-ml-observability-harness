"""FastAPI service: /score, /drift, /gates, /health, /metrics.

Every JSON endpoint uses the envelope {"ok": bool, "data": ..., "error": ...}.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest, CONTENT_TYPE_LATEST)
from pydantic import BaseModel

from .gates import GateThresholds, run_gates
from .monitor import BaselineProfile, MonitorConfig, WindowMonitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "model", "artifacts")
GOLDEN_CSV = os.path.join(ROOT, "data", "splits", "golden.csv")


class ScoreRequest(BaseModel):
    features: Dict[str, Any]


def envelope(data: Any = None, error: Optional[str] = None,
             status: int = 200) -> JSONResponse:
    return JSONResponse({"ok": error is None, "data": data, "error": error},
                        status_code=status)


def create_app(model_path: str = os.path.join(ARTIFACTS, "model.pkl"),
               baseline_path: str = os.path.join(ARTIFACTS, "baseline.json"),
               meta_path: str = os.path.join(ARTIFACTS, "meta.json"),
               golden_path: str = GOLDEN_CSV,
               config: Optional[MonitorConfig] = None,
               thresholds: Optional[GateThresholds] = None) -> FastAPI:
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(meta_path) as fh:
        meta = json.load(fh)
    baseline = BaselineProfile.load(baseline_path)
    monitor = WindowMonitor(baseline, config or MonitorConfig())
    thresholds = thresholds or GateThresholds()
    feature_names = meta["numeric_features"] + meta["categorical_features"]

    app = FastAPI(title="ModelWatch", version="1.0.0")
    registry = CollectorRegistry()
    state = {"gates_cache": None, "requests": 0}

    m_requests = Counter("modelwatch_requests_total", "Requests by endpoint",
                         ["endpoint"], registry=registry)
    m_latency = Histogram("modelwatch_score_latency_seconds",
                          "POST /score latency", registry=registry)
    m_verdict = Gauge("modelwatch_drift_verdict",
                      "Latest window drift verdict (1=drift)",
                      registry=registry)
    m_windows = Counter("modelwatch_windows_total",
                        "Evaluated monitoring windows", registry=registry)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request,
                                 exc: RequestValidationError):
        return envelope(error=f"invalid request body: {exc.errors()[0]['msg']}",
                        status=422)

    @app.get("/health")
    def health():
        m_requests.labels("health").inc()
        return envelope({"status": "ok", "model_loaded": True,
                         "buffered_rows": monitor.buffered_rows,
                         "windows_evaluated": len(monitor.reports)})

    @app.post("/score")
    def score(req: ScoreRequest):
        m_requests.labels("score").inc()
        start = time.perf_counter()
        unknown = set(req.features) - set(feature_names)
        if unknown:
            return envelope(error=f"unknown features: {sorted(unknown)}",
                            status=400)
        missing = set(feature_names) - set(req.features)
        if missing:
            return envelope(
                error=f"missing features: {sorted(missing)}", status=400)
        row = {f: req.features[f] for f in feature_names}
        frame = pd.DataFrame([row])
        for f in meta["numeric_features"]:
            frame[f] = pd.to_numeric(frame[f], errors="coerce")
        try:
            prob = float(model.predict_proba(frame)[0, 1])
        except Exception as exc:  # defensive: malformed values
            return envelope(error=f"scoring failed: {exc}", status=400)
        report = monitor.add_row(row, prob)
        if report is not None:
            m_windows.inc()
            m_verdict.set(1.0 if report.verdict else 0.0)
        m_latency.observe(time.perf_counter() - start)
        return envelope({"prediction": int(prob >= 0.5), "probability": prob,
                         "buffered_rows": monitor.buffered_rows})

    @app.get("/drift")
    def drift():
        m_requests.labels("drift").inc()
        report = monitor.latest_report()
        if report is None:
            return envelope({"status": "pending",
                             "buffered_rows": monitor.buffered_rows,
                             "window_size": monitor.config.window_size})
        return envelope({"status": "ready", "report": report.to_dict(),
                         "buffered_rows": monitor.buffered_rows})

    @app.get("/gates")
    def gates():
        m_requests.labels("gates").inc()
        if state["gates_cache"] is None:
            if not os.path.exists(golden_path):
                return envelope(error="golden dataset not found; "
                                      "run model/train.py", status=503)
            golden = pd.read_csv(golden_path)
            y = golden.pop("__label__").to_numpy()
            for f in meta["categorical_features"]:
                golden[f] = golden[f].astype(object)
            state["gates_cache"] = run_gates(model, golden, y,
                                             thresholds).to_dict()
        return envelope(state["gates_cache"])

    @app.get("/metrics")
    def metrics():
        m_requests.labels("metrics").inc()
        return PlainTextResponse(generate_latest(registry),
                                 media_type=CONTENT_TYPE_LATEST)

    app.state.monitor = monitor
    app.state.model = model
    return app
