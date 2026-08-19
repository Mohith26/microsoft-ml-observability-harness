"""Scenario generator: determinism, schedule shape, injected effects."""
import numpy as np
import pandas as pd
import pytest

from eval.scenarios import SCENARIOS, ScenarioGenerator


@pytest.fixture(scope="module")
def gen(traffic, traffic_scores):
    return ScenarioGenerator(traffic[0], traffic_scores, window_size=200,
                             n_windows=60, seed=7)


class TestSchedule:
    def test_min_window_count_default(self, traffic, traffic_scores):
        g = ScenarioGenerator(traffic[0], traffic_scores)
        assert g.n_windows >= 200
        assert len(g.specs) == g.n_windows

    def test_all_scenario_types_scheduled(self, traffic, traffic_scores):
        g = ScenarioGenerator(traffic[0], traffic_scores)
        assert {e.scenario for e in g.episodes} == set(SCENARIOS)

    def test_episode_labels_align_with_specs(self, gen):
        drift_idx = {s.index for s in gen.specs if s.is_drift}
        episode_idx = set()
        for e in gen.episodes:
            episode_idx.update(range(e.start, e.end + 1))
        assert drift_idx == episode_idx

    def test_same_seed_same_stream(self, traffic, traffic_scores):
        kw = dict(window_size=100, n_windows=25, seed=42)
        g1 = ScenarioGenerator(traffic[0], traffic_scores, **kw)
        g2 = ScenarioGenerator(traffic[0], traffic_scores, **kw)
        assert [(s.index, s.is_drift, s.scenario) for s in g1.specs] == \
               [(s.index, s.is_drift, s.scenario) for s in g2.specs]
        for (s1, f1), (s2, f2) in zip(g1.windows(), g2.windows()):
            pd.testing.assert_frame_equal(f1, f2)

    def test_different_seed_different_stream(self, traffic, traffic_scores):
        g1 = ScenarioGenerator(traffic[0], traffic_scores, window_size=100,
                               n_windows=25, seed=1)
        g2 = ScenarioGenerator(traffic[0], traffic_scores, window_size=100,
                               n_windows=25, seed=2)
        sched1 = [(s.is_drift, s.scenario) for s in g1.specs]
        sched2 = [(s.is_drift, s.scenario) for s in g2.specs]
        frames_differ = any(
            not f1.equals(f2)
            for (_, f1), (_, f2) in zip(g1.windows(), g2.windows()))
        assert sched1 != sched2 or frames_differ


class TestInjectedEffects:
    def _first_window(self, gen, scenario):
        for spec, frame in gen.windows():
            if spec.scenario == scenario:
                return frame
        raise AssertionError(f"no {scenario} window generated")

    def test_missing_spike_raises_nan_rate(self, gen, traffic):
        frame = self._first_window(gen, "missing_spike")
        base_rate = traffic[0]["occupation"].isna().mean()
        assert frame["occupation"].isna().mean() > base_rate + 0.15

    def test_category_novelty_injects_novel_value(self, gen, traffic):
        frame = self._first_window(gen, "category_novelty")
        assert (frame["workclass"] == "GigEconomyPlatform").mean() > 0.03
        assert "GigEconomyPlatform" not in set(
            traffic[0]["workclass"].dropna())

    def test_feature_scaling_shifts_mean(self, gen, traffic):
        frame = self._first_window(gen, "feature_scaling")
        assert frame["hours-per-week"].mean() > \
            traffic[0]["hours-per-week"].mean() * 1.3

    def test_covariate_shift_overrepresents_high_education(self, gen,
                                                           traffic):
        frame = self._first_window(gen, "covariate_shift")
        pool_rate = (traffic[0]["education-num"] >= 13).mean()
        assert (frame["education-num"] >= 13).mean() > pool_rate + 0.15

    def test_prediction_shift_raises_scores(self, gen, artifacts, traffic,
                                            traffic_scores):
        frame = self._first_window(gen, "prediction_shift")
        scores = artifacts["model"].predict_proba(frame)[:, 1]
        assert scores.mean() > traffic_scores.mean() + 0.1

    def test_normal_windows_untouched_columns(self, gen, traffic):
        for spec, frame in gen.windows():
            if not spec.is_drift:
                assert set(frame.columns) == set(traffic[0].columns)
                assert frame["hours-per-week"].max() <= \
                    traffic[0]["hours-per-week"].max()
                break
