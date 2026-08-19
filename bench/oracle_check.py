"""Statistic-oracle deviation measurement vs scipy / naive implementations.

Run: .venv/bin/python bench/oracle_check.py  ->  results/oracle_check.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modelwatch import stats  # noqa: E402


def naive_psi(expected, actual, n_bins=10):
    """Independent naive PSI (loop-based, quantile bins from expected).

    Quantile positions are computed with the same np.linspace call as the
    library (a 1-ulp difference in the position, e.g. 0.3 vs linspace's
    0.30000000000000004, can flip a data point across a bin edge when
    (n-1)*q is an exact integer -- a float boundary artifact, not a formula
    difference). Binning, counting, clipping and the PSI sum are independent
    loop-based code.
    """
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    qs = [np.quantile(expected, q) for q in np.linspace(0.0, 1.0, n_bins + 1)]
    edges = sorted(set(qs))
    edges[0], edges[-1] = -math.inf, math.inf
    total = 0.0
    e_n, a_n = len(expected), len(actual)
    e_props, a_props = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        e_c = sum(1 for v in expected if lo <= v < hi or (last and v >= lo))
        a_c = sum(1 for v in actual if lo <= v < hi or (last and v >= lo))
        e_props.append(max(e_c / e_n, stats.PSI_EPS))
        a_props.append(max(a_c / a_n, stats.PSI_EPS))
    e_s, a_s = sum(e_props), sum(a_props)
    for e, a in zip(e_props, a_props):
        e, a = e / e_s, a / a_s
        total += (a - e) * math.log(a / e)
    return total


def main() -> None:
    rng = np.random.default_rng(123)
    ks_stat_dev, ks_p_dev, psi_dev = 0.0, 0.0, 0.0
    n_cases = 0
    for _ in range(200):
        n1 = int(rng.integers(200, 3000))
        n2 = int(rng.integers(200, 3000))
        kind = rng.integers(0, 4)
        if kind == 0:
            a, b = rng.normal(0, 1, n1), rng.normal(0, 1, n2)
        elif kind == 1:
            a, b = rng.normal(0, 1, n1), rng.normal(0.3, 1.2, n2)
        elif kind == 2:
            a, b = rng.exponential(1, n1), rng.exponential(1.5, n2)
        else:  # heavy ties, like capital-gain
            a = np.where(rng.random(n1) < 0.9, 0.0, rng.exponential(5000, n1))
            b = np.where(rng.random(n2) < 0.8, 0.0, rng.exponential(7000, n2))
        d, p = stats.ks_2samp(a, b)
        ref = sps.ks_2samp(a, b, mode="asymp")
        ks_stat_dev = max(ks_stat_dev, abs(d - ref.statistic))
        ks_p_dev = max(ks_p_dev, abs(p - ref.pvalue))
        psi_dev = max(psi_dev, abs(stats.psi(a, b) - naive_psi(a, b)))
        n_cases += 1

    out = {
        "n_random_cases": n_cases,
        "seed": 123,
        "ks_statistic_max_abs_deviation_vs_scipy": ks_stat_dev,
        "ks_pvalue_max_abs_deviation_vs_scipy_asymp": ks_p_dev,
        "ks_pvalue_note": "own p-value uses the Kolmogorov limit "
                          "distribution; scipy asymp mode uses a finite-n "
                          "refinement (kstwo), hence small deviation",
        "psi_max_abs_deviation_vs_naive_oracle": psi_dev,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "oracle_check.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
