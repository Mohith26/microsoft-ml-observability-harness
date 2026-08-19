"""API endpoints, envelope shape, error paths, metrics."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modelwatch.api import create_app
from modelwatch.monitor import MonitorConfig


@pytest.fixture(scope="module")
def client(artifacts):
    app = create_app(config=MonitorConfig(window_size=20))
    return TestClient(app)


@pytest.fixture(scope="module")
def payload(traffic):
    row = traffic[0].head(1).where(pd.notna(traffic[0].head(1)), None)
    return {"features": row.to_dict(orient="records")[0]}


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True and body["error"] is None
        assert body["data"]["status"] == "ok"
        assert body["data"]["model_loaded"] is True


class TestScore:
    def test_score_returns_prediction(self, client, payload):
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0

    def test_score_buffers_row(self, client, payload):
        before = client.get("/health").json()["data"]["buffered_rows"]
        client.post("/score", json=payload)
        after = client.get("/health").json()["data"]["buffered_rows"]
        assert after == (before + 1) % 20

    def test_score_null_feature_ok(self, client, payload):
        p = {"features": dict(payload["features"], occupation=None)}
        resp = client.post("/score", json=p)
        assert resp.status_code == 200

    def test_score_missing_feature_400(self, client, payload):
        feats = dict(payload["features"])
        feats.pop("age")
        resp = client.post("/score", json={"features": feats})
        assert resp.status_code == 400
        body = resp.json()
        assert body["ok"] is False and "missing features" in body["error"]

    def test_score_unknown_feature_400(self, client, payload):
        p = {"features": dict(payload["features"], bogus=1)}
        resp = client.post("/score", json=p)
        assert resp.status_code == 400
        assert "unknown features" in resp.json()["error"]

    def test_score_unscorable_value_400(self, client, payload):
        p = {"features": dict(payload["features"], sex=["unhashable"])}
        resp = client.post("/score", json=p)
        assert resp.status_code == 400
        assert "scoring failed" in resp.json()["error"]

    def test_score_invalid_body_422_envelope(self, client):
        resp = client.post("/score", json={"nope": 1})
        assert resp.status_code == 422
        body = resp.json()
        assert body["ok"] is False and body["error"] is not None

    def test_window_closes_after_window_size_rows(self, artifacts, traffic):
        app = create_app(config=MonitorConfig(window_size=10))
        c = TestClient(app)
        rows = traffic[0].head(10).where(pd.notna(traffic[0].head(10)), None)
        for r in rows.to_dict(orient="records"):
            c.post("/score", json={"features": r})
        drift = c.get("/drift").json()["data"]
        assert drift["status"] == "ready"
        assert drift["report"]["n_rows"] == 10


class TestDrift:
    def test_drift_pending_before_first_window(self, artifacts):
        app = create_app(config=MonitorConfig(window_size=50))
        c = TestClient(app)
        body = c.get("/drift").json()
        assert body["ok"] is True
        assert body["data"]["status"] == "pending"
        assert body["data"]["window_size"] == 50

    def test_drift_report_shape(self, client, payload, traffic):
        rows = traffic[0].head(20).where(pd.notna(traffic[0].head(20)), None)
        for r in rows.to_dict(orient="records"):
            client.post("/score", json={"features": r})
        data = client.get("/drift").json()["data"]
        assert data["status"] == "ready"
        report = data["report"]
        assert set(report) >= {"features", "prediction_psi", "verdict",
                               "reasons"}


class TestGatesEndpoint:
    def test_gates_healthy_passes(self, client):
        body = client.get("/gates").json()
        assert body["ok"] is True
        assert body["data"]["overall_passed"] is True

    def test_gates_cached_stable(self, client):
        assert client.get("/gates").json() == client.get("/gates").json()

    def test_gates_missing_golden_503(self, artifacts, tmp_path):
        app = create_app(golden_path=str(tmp_path / "nope.csv"))
        c = TestClient(app)
        resp = c.get("/gates")
        assert resp.status_code == 503
        assert resp.json()["ok"] is False


class TestMetrics:
    def test_metrics_prometheus_format(self, client, payload):
        client.post("/score", json=payload)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "modelwatch_requests_total" in text
        assert "modelwatch_score_latency_seconds" in text
        assert "modelwatch_windows_total" in text

    def test_metrics_counter_increments(self, client, payload):
        def count():
            for line in client.get("/metrics").text.splitlines():
                if line.startswith(
                        'modelwatch_requests_total{endpoint="score"}'):
                    return float(line.split()[-1])
            return 0.0
        before = count()
        client.post("/score", json=payload)
        assert count() == before + 1

    def test_drift_verdict_gauge_present_after_window(self, artifacts,
                                                      traffic):
        app = create_app(config=MonitorConfig(window_size=5))
        c = TestClient(app)
        rows = traffic[0].head(5).where(pd.notna(traffic[0].head(5)), None)
        for r in rows.to_dict(orient="records"):
            c.post("/score", json={"features": r})
        assert "modelwatch_drift_verdict" in c.get("/metrics").text
