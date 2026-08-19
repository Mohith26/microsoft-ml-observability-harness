"""Statistic implementations vs scipy / naive oracles."""
import math

import numpy as np
import pytest
from scipy import stats as sps

from modelwatch import stats


def _pairs(seed=1, n_cases=50):
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        n1 = int(rng.integers(100, 2000))
        n2 = int(rng.integers(100, 2000))
        kind = rng.integers(0, 3)
        if kind == 0:
            yield rng.normal(0, 1, n1), rng.normal(0, 1, n2)
        elif kind == 1:
            yield rng.normal(0, 1, n1), rng.normal(0.4, 1.3, n2)
        else:  # heavy ties
            yield (np.where(rng.random(n1) < 0.9, 0.0,
                            rng.exponential(1000, n1)),
                   np.where(rng.random(n2) < 0.8, 0.0,
                            rng.exponential(1500, n2)))


class TestKS:
    def test_statistic_matches_scipy_exactly(self):
        for a, b in _pairs(seed=1):
            d, _ = stats.ks_2samp(a, b)
            assert d == pytest.approx(sps.ks_2samp(a, b).statistic, abs=1e-12)

    def test_pvalue_close_to_scipy_asymp(self):
        max_dev = 0.0
        for a, b in _pairs(seed=2):
            _, p = stats.ks_2samp(a, b)
            ref = sps.ks_2samp(a, b, mode="asymp").pvalue
            max_dev = max(max_dev, abs(p - ref))
        assert max_dev < 0.03  # limit distribution vs scipy finite-n kstwo

    def test_identical_samples_high_pvalue(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 1500)
        d, p = stats.ks_2samp(a, a.copy())
        assert d == 0.0
        assert p == 1.0

    def test_disjoint_samples_stat_one(self):
        d, p = stats.ks_2samp(np.arange(100), np.arange(200, 300))
        assert d == 1.0
        assert p < 1e-10

    def test_nan_values_dropped(self):
        a = np.array([1.0, 2.0, 3.0, np.nan])
        b = np.array([1.0, 2.0, 3.0])
        d, _ = stats.ks_2samp(a, b)
        assert d == pytest.approx(sps.ks_2samp(a[:3], b).statistic)

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError):
            stats.ks_2samp(np.array([]), np.array([1.0]))

    def test_kolmogorov_sf_bounds_and_monotonic(self):
        xs = np.linspace(0.01, 3.0, 50)
        vals = [stats.kolmogorov_sf(x) for x in xs]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert all(v1 >= v2 - 1e-12 for v1, v2 in zip(vals, vals[1:]))
        assert stats.kolmogorov_sf(0.0) == 1.0

    def test_kolmogorov_sf_matches_scipy_kstwobign(self):
        for x in (0.3, 0.5, 0.8, 1.0, 1.36, 2.0, 2.5):
            assert stats.kolmogorov_sf(x) == pytest.approx(
                sps.kstwobign.sf(x), abs=1e-6)


class TestPSI:
    def test_matches_naive_oracle(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "oracle_check", os.path.join(os.path.dirname(__file__), "..",
                                         "bench", "oracle_check.py"))
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        for a, b in _pairs(seed=4, n_cases=25):
            assert stats.psi(a, b) == pytest.approx(
                oracle.naive_psi(a, b), abs=1e-9)

    def test_same_distribution_low_psi(self):
        rng = np.random.default_rng(5)
        pool = rng.normal(0, 1, 20000)
        assert stats.psi(pool[:10000], pool[10000:]) < 0.02

    def test_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(6)
        assert stats.psi(rng.normal(0, 1, 5000),
                         rng.normal(1.0, 1, 5000)) > 0.2

    def test_psi_nonnegative(self):
        for a, b in _pairs(seed=7, n_cases=20):
            assert stats.psi(a, b) >= 0.0

    def test_proportion_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            stats.psi_from_proportions(np.array([0.5, 0.5]),
                                       np.array([0.3, 0.3, 0.4]))

    def test_quantile_edges_dedupe_constant_heavy(self):
        values = np.array([0.0] * 950 + list(range(50)), dtype=float)
        edges = stats.quantile_bin_edges(values, n_bins=10)
        assert edges[0] == -math.inf and edges[-1] == math.inf
        assert len(np.unique(edges)) == len(edges)

    def test_quantile_edges_empty_raises(self):
        with pytest.raises(ValueError):
            stats.quantile_bin_edges(np.array([np.nan]))

    def test_bin_proportions_sum_to_one(self):
        rng = np.random.default_rng(8)
        v = rng.normal(0, 1, 1000)
        edges = stats.quantile_bin_edges(v)
        assert stats.bin_proportions(v, edges).sum() == pytest.approx(1.0)

    def test_bin_proportions_all_nan(self):
        edges = np.array([-math.inf, 0.0, math.inf])
        assert stats.bin_proportions(np.array([np.nan]), edges).sum() == 0.0


class TestCategoricalPSI:
    def test_same_mix_low(self):
        props = {"a": 0.6, "b": 0.4}
        values = ["a"] * 600 + ["b"] * 400
        assert stats.categorical_psi(props, values) < 0.01

    def test_shifted_mix_high(self):
        props = {"a": 0.6, "b": 0.4}
        values = ["a"] * 100 + ["b"] * 900
        assert stats.categorical_psi(props, values) > 0.2

    def test_novel_category_increases_psi(self):
        props = {"a": 0.6, "b": 0.4}
        base = stats.categorical_psi(props, ["a"] * 60 + ["b"] * 40)
        novel = stats.categorical_psi(props, ["a"] * 40 + ["b"] * 30 +
                                      ["zzz"] * 30)
        assert novel > base + 0.1

    def test_rare_bucket_pooling(self):
        props = {"a": 0.5, "b": 0.4, "__rare__": 0.1}
        values = ["a"] * 50 + ["b"] * 40 + ["r1"] * 5 + ["r2"] * 5
        psi = stats.categorical_psi(props, values,
                                    rare_categories=("r1", "r2"))
        assert psi < 0.01

    def test_missing_values_excluded(self):
        props = {"a": 0.5, "b": 0.5}
        values = ["a", "b", None, float("nan")] * 25
        assert stats.categorical_psi(props, values) < 0.01

    def test_all_missing_zero_proportions(self):
        assert stats.categorical_proportions([None, float("nan")],
                                             ["a"]).sum() == 0.0


class TestECE:
    def test_perfectly_calibrated_low(self):
        rng = np.random.default_rng(9)
        probs = rng.uniform(0, 1, 200000)
        y = (rng.random(200000) < probs).astype(float)
        assert stats.expected_calibration_error(y, probs) < 0.01

    def test_overconfident_high(self):
        y = np.array([0.0, 1.0] * 500)
        probs = np.where(y == 1, 0.99, 0.01) * 0 + 0.99
        assert stats.expected_calibration_error(y, probs) > 0.4

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            stats.expected_calibration_error(np.array([1.0]),
                                             np.array([0.5, 0.5]))
