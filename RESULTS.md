# Measured results

These are my benchmark notes from an actual run on 2026-08-19 (local dev
machine, macOS, CPU only, Python 3.9.6). Raw outputs are committed under
`results/`. Reproduce commands assume the venv from the README and that
`model/train.py` has been run first.

## 1. Drift detection vs ground-truth injected episodes

Setup: 240 tumbling windows x 500 rows, seed 7; 32 labeled drift episodes
(95 drift windows, 145 normal windows) across 5 scenario types. Detector =
`WindowMonitor` with default `MonitorConfig` thresholds (PSI > 0.2, KS p <
1e-3 & stat > 0.1, missing-rate delta > 0.10, novelty > 0.01, out-of-range >
0.01, prediction PSI > 0.2). Keep in mind the ground truth here is injected
drift replayed over real held-out UCI Adult traffic rows, not real-world
drift, so these numbers measure detector correctness, not field performance.

Reproduce: `.venv/bin/python eval/run_eval.py` -> `results/drift_eval.json`

### Window level (overall)

| precision | recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| 0.9896 | 1.0000 | 0.9948 | 95 | 1 | 0 | 144 |

The single false positive was a normal window flagged by the KS test on
`fnlwgt` (reasons logged in `results/drift_eval.json`).

### Episode level (overall)

| precision | recall | F1 | detected episodes | false alarm segments |
|---|---|---|---|---|
| 0.9697 | 1.0000 | 0.9846 | 32/32 | 1 |

An episode counts as detected if at least one alarm segment overlaps it; an
alarm segment is a false positive if it overlaps no true episode.

### Per scenario type

| scenario | windows | window recall | episodes | episode recall |
|---|---|---|---|---|
| covariate_shift (education resampling) | 23 | 1.00 | 7 | 1.00 |
| feature_scaling (hours-per-week x1.5) | 21 | 1.00 | 7 | 1.00 |
| missing_spike (30% occupation nulled) | 15 | 1.00 | 6 | 1.00 |
| category_novelty (10% novel workclass) | 18 | 1.00 | 6 | 1.00 |
| prediction_shift (score^2 resampling) | 18 | 1.00 | 6 | 1.00 |

Window-level precision is global, since a false positive isn't attributable
to any one scenario type. Worth recording: before I added rare-category
pooling to the categorical PSI, `native-country` (41 categories, many with
under 1% share) caused 77 false-positive windows and precision was 0.55.
Pooling rare categories into a `__rare__` bucket removed all of them. I'm
keeping that in the notes because it changed the headline number a lot.

## 2. Statistic-oracle deviations

Reproduce: `.venv/bin/python bench/oracle_check.py` ->
`results/oracle_check.json` (200 random sample pairs, mixed distributions
including heavy ties, seed 123).

| check | max abs deviation |
|---|---|
| KS statistic vs `scipy.stats.ks_2samp` | 0.0 (exact) |
| KS p-value vs scipy `mode="asymp"` | 0.02072 |
| PSI vs independent naive loop implementation | 1.11e-16 |

On the KS p-value: my implementation uses the Kolmogorov limit distribution
while scipy's asymp mode uses a finite-n refinement (`kstwo`). The worst
deviation occurs at p around 0.65 with n1,n2 around 380, which is irrelevant
to the 1e-3 alert threshold. On PSI: the naive oracle deliberately uses the
same `np.linspace` quantile positions, because a 1-ulp difference in the
position (0.3 vs 0.30000000000000004) can flip a data point across a bin
edge when (n-1)*q is an exact integer; the binning, counting and formula are
independent code.

## 3. Regression gates: healthy vs degraded model

Reproduce: `.venv/bin/python model/gates_report.py` ->
`results/gates_matrix.json`. Golden split n=9,768. Thresholds: accuracy >=
0.84, ROC-AUC >= 0.88, ECE <= 0.06, slice accuracy >= 0.75 (slices with n >=
50). The degraded model is the same pipeline retrained with 40% flipped
labels.

| gate | healthy | degraded | threshold | healthy | degraded |
|---|---|---|---|---|---|
| accuracy | 0.8545 | 0.8315 | >=0.84 | PASS | **FAIL** |
| roc_auc | 0.9089 | 0.8749 | >=0.88 | PASS | **FAIL** |
| ece | 0.0101 | 0.2621 | <=0.06 | PASS | **FAIL** |
| slice sex=Female (n=3238) | 0.9287 | 0.9132 | >=0.75 | PASS | PASS |
| slice sex=Male (n=6530) | 0.8178 | 0.7910 | >=0.75 | PASS | PASS |
| slice race=Amer-Indian-Eskimo (n=89) | 0.8764 | 0.8764 | >=0.75 | PASS | PASS |
| slice race=Asian-Pac-Islander (n=314) | 0.8121 | 0.7484 | >=0.75 | PASS | **FAIL** |
| slice race=Black (n=954) | 0.9004 | 0.8931 | >=0.75 | PASS | PASS |
| slice race=Other (n=89) | 0.8989 | 0.8989 | >=0.75 | PASS | PASS |
| slice race=White (n=8322) | 0.8502 | 0.8264 | >=0.75 | PASS | PASS |
| **overall** | | | | **PASS** | **FAIL** |

Both directions are also proven in tests (`tests/test_gates.py`).

## 4. Model quality (golden split)

Reproduce: `.venv/bin/python model/train.py` -> `results/model_eval.json`.

| metric | healthy | degraded (40% labels flipped) |
|---|---|---|
| accuracy | 0.8545 | 0.8315 |
| ROC-AUC | 0.9089 | 0.8749 |
| ECE (10-bin) | 0.0101 | 0.2621 |

Per-slice accuracy (healthy): sex=Female 0.9287, sex=Male 0.8178,
race=Amer-Indian-Eskimo 0.8764, race=Asian-Pac-Islander 0.8121,
race=Black 0.9004, race=Other 0.8989, race=White 0.8502.

## 5. Throughput and latency (best of 5 runs)

Reproduce: `.venv/bin/python bench/run_bench.py` -> `results/bench.json`

| benchmark | result |
|---|---|
| drift engine throughput | best 148,828 rows/sec, median 147,833 rows/sec |
| POST /score latency (1,000 req/run) | p50 2.613 ms / p95 3.108 ms |
| GET /drift latency (200 req/run) | p50 1.937 ms / p95 2.418 ms |

The API latencies are in-process via FastAPI's TestClient, so there is no
network socket, no uvicorn worker, and no wire serialization in these
numbers. Drift-engine throughput is wall clock over pre-materialized 500-row
windows (10,000 rows per run).

## 6. Tests and coverage

Reproduce:
`.venv/bin/python -m pytest tests/ --cov=modelwatch --cov-report=term --color=no -rN`

- **81 passed** (0 failed), runtime ~1.8 s (artifacts pre-built).
- **Coverage on `modelwatch/`: 100%** (414/414 statements: api.py 95,
  gates.py 53, monitor.py 161, stats.py 104, `__init__.py` 1).

## What I did not measure

- Real-network API latency: that would need uvicorn plus a client over an
  actual socket. The in-process numbers above stand in for it.
- Real-world (non-injected) drift precision and recall: no labeled real
  drift exists for this dataset, so all drift ground truth is synthetic by
  design.
