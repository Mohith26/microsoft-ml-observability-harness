"""Baseline profiling and windowed monitoring."""
import numpy as np
import pandas as pd
import pytest

from modelwatch.monitor import BaselineProfile, MonitorConfig, WindowMonitor


def _window(traffic, rng, n=500):
    idx = rng.choice(len(traffic), n, replace=True)
    return traffic.iloc[idx].reset_index(drop=True)


@pytest.fixture()
def monitor(artifacts):
    return WindowMonitor(artifacts["baseline"], MonitorConfig())


class TestBaselineProfile:
    def test_roundtrip_save_load(self, artifacts, tmp_path):
        path = str(tmp_path / "baseline.json")
        artifacts["baseline"].save(path)
        loaded = BaselineProfile.load(path)
        assert loaded.to_dict() == artifacts["baseline"].to_dict()

    def test_numeric_profile_fields(self, artifacts):
        prof = artifacts["baseline"].numeric["age"]
        assert prof["min"] <= prof["max"]
        assert np.isclose(sum(prof["props"]), 1.0)
        assert len(prof["sample"]) <= MonitorConfig().baseline_sample_size

    def test_rare_categories_pooled(self, artifacts):
        prof = artifacts["baseline"].categorical["native-country"]
        assert "__rare__" in prof["props"]
        assert len(prof["rare_categories"]) > 0
        assert set(prof["rare_categories"]) <= set(prof["categories"])

    def test_from_frame_builds_full_profile(self):
        rng = np.random.default_rng(21)
        df = pd.DataFrame({
            "num": rng.normal(50, 10, 1000),
            "const": np.zeros(1000),
            "cat": rng.choice(["a", "b", "c"], 1000,
                              p=[0.6, 0.395, 0.005]).astype(object),
        })
        df.loc[:49, "cat"] = np.nan
        scores = rng.uniform(0, 1, 1000)
        prof = BaselineProfile.from_frame(df, ["num", "const"], ["cat"],
                                          scores, MonitorConfig(), seed=1)
        assert prof.n_rows == 1000
        assert np.isclose(sum(prof.numeric["num"]["props"]), 1.0)
        assert prof.numeric["const"]["min"] == prof.numeric["const"]["max"]
        assert prof.categorical["cat"]["missing_rate"] == pytest.approx(0.05)
        assert "c" in prof.categorical["cat"]["rare_categories"]
        assert np.isclose(sum(prof.prediction["props"]), 1.0)

    def test_from_frame_deterministic_sample(self, artifacts, traffic):
        df = traffic[0].head(3000)
        scores = np.linspace(0, 1, 3000)
        p1 = BaselineProfile.from_frame(df, ["age"], ["sex"], scores, seed=5)
        p2 = BaselineProfile.from_frame(df, ["age"], ["sex"], scores, seed=5)
        assert p1.to_dict() == p2.to_dict()

    def test_config_roundtrip(self):
        cfg = MonitorConfig(window_size=123, psi_threshold=0.5)
        assert MonitorConfig.from_dict(cfg.to_dict()) == cfg


class TestWindowVerdicts:
    def test_normal_window_no_drift(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(11)
        frame = _window(traffic[0], rng)
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert report.verdict is False
        assert report.reasons == []

    def test_scaled_feature_detected(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(12)
        frame = _window(traffic[0], rng)
        frame["hours-per-week"] = frame["hours-per-week"] * 1.5
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert report.verdict is True
        assert any(r.startswith("hours-per-week:") for r in report.reasons)

    def test_missing_spike_detected(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(13)
        frame = _window(traffic[0], rng)
        frame.loc[rng.random(len(frame)) < 0.3, "occupation"] = np.nan
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert "occupation:missing_spike" in report.reasons

    def test_category_novelty_detected(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(14)
        frame = _window(traffic[0], rng)
        frame.loc[rng.random(len(frame)) < 0.1, "workclass"] = "Metaverse"
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert "workclass:category_novelty" in report.reasons

    def test_out_of_range_detected(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(15)
        frame = _window(traffic[0], rng)
        frame.loc[rng.random(len(frame)) < 0.05, "age"] = 250
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert "age:out_of_range" in report.reasons

    def test_prediction_shift_detected(self, monitor, artifacts, traffic,
                                       traffic_scores):
        rng = np.random.default_rng(16)
        weights = (traffic_scores + 0.02) ** 2
        idx = rng.choice(len(traffic[0]), 500, p=weights / weights.sum())
        frame = traffic[0].iloc[idx].reset_index(drop=True)
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        report = monitor.evaluate(frame, scores)
        assert "prediction:psi" in report.reasons

    def test_missing_column_counts_as_missing(self, monitor, artifacts,
                                              traffic):
        rng = np.random.default_rng(17)
        frame = _window(traffic[0], rng).drop(columns=["age", "workclass"])
        scores = np.full(len(frame), 0.2)
        report = monitor.evaluate(frame, scores)
        assert report.features["age"]["missing_rate"] == 1.0
        assert "age:missing_spike" in report.reasons
        assert "workclass:missing_spike" in report.reasons


class TestBuffering:
    def test_add_row_triggers_window(self, artifacts, traffic):
        cfg = MonitorConfig(window_size=50)
        mon = WindowMonitor(artifacts["baseline"], cfg)
        rows = traffic[0].head(50).to_dict(orient="records")
        report = None
        for i, row in enumerate(rows):
            out = mon.add_row(row, 0.2)
            if i < 49:
                assert out is None
            else:
                report = out
        assert report is not None
        assert report.n_rows == 50
        assert mon.buffered_rows == 0
        assert mon.latest_report() is report

    def test_latest_report_none_initially(self, artifacts):
        mon = WindowMonitor(artifacts["baseline"])
        assert mon.latest_report() is None
        assert mon.buffered_rows == 0

    def test_report_to_dict_keys(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(18)
        frame = _window(traffic[0], rng)
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        d = monitor.evaluate(frame, scores).to_dict()
        assert {"window_index", "n_rows", "features", "prediction_psi",
                "verdict", "reasons"} <= set(d)
        assert "age" in d["features"] and "sex" in d["features"]

    def test_window_indices_increment(self, monitor, artifacts, traffic):
        rng = np.random.default_rng(19)
        frames = [_window(traffic[0], rng, 100) for _ in range(3)]
        idxs = [monitor.evaluate(f, np.full(len(f), 0.2)).window_index
                for f in frames]
        assert idxs == sorted(idxs)
        assert len(set(idxs)) == 3
