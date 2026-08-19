"""Statistical primitives: PSI, two-sample KS, ECE.

Own implementations; verified against scipy.stats / naive oracles in tests.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

PSI_EPS = 1e-6


def quantile_bin_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from a reference sample, outer edges opened to +/-inf.

    Duplicate quantiles (heavy-tailed / mostly-constant features such as
    capital-gain) are collapsed, so the effective number of bins may be lower.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("cannot build bin edges from an empty sample")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if edges.size < 2:  # constant feature -> single degenerate bin
        edges = np.array([edges[0], edges[0]])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Proportion of non-NaN values falling in each bin defined by edges."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges)
    return counts / counts.sum()


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray,
                         eps: float = PSI_EPS) -> float:
    """PSI between two aligned proportion vectors (eps-clipped, renormalized)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError("proportion vectors must have the same shape")
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected_values: np.ndarray, actual_values: np.ndarray,
        n_bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the expected sample."""
    edges = quantile_bin_edges(expected_values, n_bins=n_bins)
    return psi_from_proportions(bin_proportions(expected_values, edges),
                                bin_proportions(actual_values, edges))


def categorical_proportions(values: Sequence, categories: Sequence[str],
                            rare_categories: Sequence[str] = ()) -> np.ndarray:
    """Proportions over known categories + a __rare__ bucket + __other__ bucket.

    Values in `rare_categories` are pooled into the __rare__ bucket (rare
    categories are individually too noisy in a small window and inflate PSI);
    values in neither list land in __other__ (novel). NaN/None are excluded
    (they are tracked by the missing-rate signal).
    """
    known = list(categories)
    index = {c: i for i, c in enumerate(known)}
    rare = set(rare_categories)
    counts = np.zeros(len(known) + 2)  # [known..., __rare__, __other__]
    total = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        total += 1
        s = str(v)
        if s in index:
            counts[index[s]] += 1
        elif s in rare:
            counts[len(known)] += 1
        else:
            counts[len(known) + 1] += 1
    if total == 0:
        return counts
    return counts / total


def categorical_psi(expected_props: Dict[str, float], actual_values: Sequence,
                    rare_categories: Sequence[str] = ()) -> float:
    """PSI over category proportions (baseline dict vs raw window values).

    `expected_props` may contain a '__rare__' key holding the pooled baseline
    proportion of `rare_categories`; the expected __other__ proportion is 0.
    """
    categories = [c for c in expected_props if c != "__rare__"]
    expected = np.array([expected_props[c] for c in categories] +
                        [expected_props.get("__rare__", 0.0), 0.0])
    actual = categorical_proportions(actual_values, categories,
                                     rare_categories)
    return psi_from_proportions(expected, actual)


def kolmogorov_sf(x: float) -> float:
    """Survival function of the Kolmogorov limit distribution Q_KS(x).

    Two-series form (Numerical Recipes): the theta-function series for small x,
    the alternating series otherwise.
    """
    if x < 1e-8:
        return 1.0
    if x < 1.18:
        y = math.exp(-math.pi ** 2 / (8.0 * x ** 2))
        cdf = (math.sqrt(2.0 * math.pi) / x) * (y + y ** 9 + y ** 25 + y ** 49)
        return max(0.0, min(1.0, 1.0 - cdf))
    y = math.exp(-2.0 * x ** 2)
    sf = 2.0 * (y - y ** 4 + y ** 9 - y ** 16 + y ** 25)
    return max(0.0, min(1.0, sf))


def ks_2samp(sample_a: np.ndarray, sample_b: np.ndarray):
    """Two-sample two-sided KS statistic and asymptotic p-value.

    Statistic matches scipy.stats.ks_2samp exactly; the p-value uses the
    Kolmogorov limit distribution (scipy's asymp mode uses a finite-n
    refinement, so p-values agree to ~1e-2; measured in tests).
    """
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        raise ValueError("ks_2samp requires non-empty samples")
    data_all = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data_all, side="right") / n1
    cdf_b = np.searchsorted(b, data_all, side="right") / n2
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = n1 * n2 / (n1 + n2)
    p = kolmogorov_sf(math.sqrt(en) * d)
    return d, p


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if y_true.shape != probs.shape:
        raise ValueError("y_true and probs must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = y_true.size
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
