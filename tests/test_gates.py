"""Regression gates: healthy model must pass, degraded model must fail."""
import pytest

from modelwatch.gates import GateReport, GateThresholds, run_gates


class TestGates:
    def test_healthy_model_passes_all_gates(self, artifacts, golden):
        report = run_gates(artifacts["model"], golden[0], golden[1])
        assert report.overall_passed is True
        assert all(g["passed"] for g in report.gates)

    def test_degraded_model_fails_gates(self, artifacts, golden):
        report = run_gates(artifacts["degraded"], golden[0], golden[1])
        assert report.overall_passed is False
        failed = {g["name"] for g in report.gates if not g["passed"]}
        assert "accuracy" in failed
        assert "roc_auc" in failed
        assert "ece" in failed

    def test_degraded_fails_a_slice_gate(self, artifacts, golden):
        report = run_gates(artifacts["degraded"], golden[0], golden[1])
        assert any(g["name"].startswith("slice_accuracy") and not g["passed"]
                   for g in report.gates)

    def test_core_gates_present(self, artifacts, golden):
        report = run_gates(artifacts["model"], golden[0], golden[1])
        names = {g["name"] for g in report.gates}
        assert {"accuracy", "roc_auc", "ece"} <= names
        assert any(n.startswith("slice_accuracy[sex=") for n in names)
        assert any(n.startswith("slice_accuracy[race=") for n in names)

    def test_small_slices_reported_not_gated(self, artifacts, golden):
        thresholds = GateThresholds(min_slice_size=5000)
        report = run_gates(artifacts["model"], golden[0], golden[1],
                           thresholds)
        small = [s for s in report.slices.values() if s["n"] < 5000]
        assert small and all(not s["gated"] for s in small)

    def test_impossible_threshold_fails_healthy(self, artifacts, golden):
        report = run_gates(artifacts["model"], golden[0], golden[1],
                           GateThresholds(min_accuracy=0.999))
        assert report.overall_passed is False

    def test_thresholds_roundtrip(self):
        t = GateThresholds(min_accuracy=0.9, slice_features=("sex",))
        assert GateThresholds.from_dict(t.to_dict()) == t

    def test_report_to_dict(self, artifacts, golden):
        d = run_gates(artifacts["model"], golden[0], golden[1]).to_dict()
        assert {"gates", "slices", "overall_passed"} <= set(d)

    def test_missing_slice_feature_skipped(self, artifacts, golden):
        thresholds = GateThresholds(slice_features=("sex", "not-a-column"))
        report = run_gates(artifacts["model"], golden[0], golden[1],
                           thresholds)
        assert not any("not-a-column" in k for k in report.slices)

    def test_gate_values_match_metric_ranges(self, artifacts, golden):
        report = run_gates(artifacts["model"], golden[0], golden[1])
        by_name = {g["name"]: g["value"] for g in report.gates}
        assert 0.8 < by_name["accuracy"] < 1.0
        assert 0.85 < by_name["roc_auc"] < 1.0
        assert 0.0 <= by_name["ece"] < 0.05
