# ModelWatch — ML Observability & Drift-Detection Harness

A Python harness that monitors a trained model in "production": ingests scoring
traffic, detects covariate/prediction drift (PSI, two-sample KS) against a
training baseline, tracks label-free health signals (missing-value rate,
out-of-range rate, category novelty), runs golden-dataset regression gates
(accuracy / ROC-AUC / calibration / per-slice accuracy), and exposes everything
over FastAPI with Prometheus-style metrics.

Drift detection quality is **measured against ground truth**: a seeded scenario
generator replays held-out UCI Adult rows and injects labeled drift episodes;
precision/recall/F1 are computed at window and episode level. Statistic
implementations are verified against scipy / naive oracles. All measured
numbers live in [RESULTS.md](RESULTS.md) and `results/*.json`.

## Architecture

```
                        +--------------------+
   POST /score  ------> |  FastAPI (api.py)  | ----> prediction (JSON envelope)
                        +---------+----------+
                                  | buffers scored rows
                                  v
   +-----------------+   +------------------+    windows of 500 rows
   | BaselineProfile |-->|  WindowMonitor   | --> WindowReport (per-feature
   | (train window:  |   |  (monitor.py)    |     PSI, KS stat/p, missing/
   |  histograms,    |   +------------------+     novelty/out-of-range rates,
   |  quantile bins, |                            prediction PSI, verdict)
   |  cat props,     |   +------------------+
   |  score dist)    |   |  Gates (gates.py)| --> accuracy / AUC / ECE /
   +-----------------+   |  golden dataset  |     per-slice gate matrix
                         +------------------+
   GET /drift -> latest window report      GET /gates -> gate results
   GET /health -> liveness + buffer state  GET /metrics -> Prometheus text
```

- `modelwatch/stats.py` — own implementations of PSI (10-bin quantile,
  rare-category pooling for categoricals), two-sample KS (statistic exact vs
  scipy; asymptotic Kolmogorov p-value), and ECE.
- `modelwatch/monitor.py` — baseline profiling + tumbling-window monitoring
  with configurable thresholds.
- `modelwatch/gates.py` — golden-dataset regression gates incl. per-slice
  accuracy (sex, race) as a responsible-AI release check.
- `modelwatch/api.py` — FastAPI service, consistent
  `{"ok", "data", "error"}` envelope, Prometheus counters/histogram/gauge.
- `model/train.py` — trains the reference + deliberately degraded model,
  persists baseline profile and data splits.
- `eval/` — seeded scenario generator + ground-truth drift evaluation.
- `bench/` — throughput/latency benchmarks and statistic-oracle checks.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install "scikit-learn==1.3.2" "pandas==2.0.3" "numpy==1.24.4" \
  "scipy==1.10.1" "fastapi==0.103.2" "uvicorn==0.23.2" "httpx==0.24.1" \
  "prometheus-client==0.19.0" "pytest==7.4.4" "pytest-cov==4.1.0"

.venv/bin/python model/train.py        # fetch UCI Adult, train, profile
.venv/bin/python eval/run_eval.py      # drift precision/recall vs ground truth
.venv/bin/python model/gates_report.py # gate matrix healthy vs degraded
.venv/bin/python bench/oracle_check.py # statistic deviations vs scipy/naive
.venv/bin/python bench/run_bench.py    # throughput + API latency
.venv/bin/python -m pytest tests/ --cov=modelwatch   # 81 tests, 100% cov

.venv/bin/uvicorn modelwatch.api:create_app --factory  # serve (optional)
```

## Model card

- **Model**: scikit-learn `LogisticRegression` (max_iter=2000) on
  median-imputed + standardized numerics and constant-imputed + one-hot
  categoricals (`handle_unknown="ignore"`). A deliberately **degraded** twin is
  retrained with 40% of training labels flipped (seed 42) to prove the gates
  fail in the bad direction.
- **Task**: binary income classification (>50K vs <=50K), used here purely as
  a realistic monitoring subject — not as a deployable income predictor.
- **Data**: UCI Adult / "Census Income" (OpenML `adult` version 2, 48,842
  rows, 1994 US Census extract). Splits: 60% train / 20% golden / 20% traffic,
  stratified, seed 42. Downloaded via `sklearn.datasets.fetch_openml` with
  scikit-learn's ARFF checksum-validated cache under `data/` (gitignored).
- **Measured performance** (golden split): accuracy 0.8545, ROC-AUC 0.9089,
  ECE 0.0101. Per-slice accuracy ranges from 0.8121 (race=Asian-Pac-Islander,
  n=314) to 0.9287 (sex=Female, n=3,238) — full table in RESULTS.md.
- **Caveats / responsible-AI notes**: the dataset encodes 1994 census
  demographics and historical income disparities; sex/race slice accuracy
  differs by up to ~11 points. Labels reflect a $50K threshold that is not
  inflation-adjusted. This repo uses the dataset as a standard monitoring
  benchmark; per-slice gates exist precisely to surface such disparities.
  Do not use this model for real decisions about people.

## Honest limits

- **Injected, not real-world, drift**: eval ground truth is synthetic drift
  (demographic resampling, feature scaling, missing spikes, novel categories,
  score-weighted resampling) applied to real held-out rows. Real-world drift
  is messier; these numbers upper-bound what the same detector would do there.
- **In-process latency**: API p50/p95 use FastAPI's TestClient (no network
  socket, no uvicorn worker, no serialization over a wire).
- **KS p-value**: statistic matches scipy exactly; the p-value uses the
  Kolmogorov limit distribution, deviating ≤0.0207 from scipy's finite-n
  `asymp` mode (worst case at p≈0.65 — far from the 1e-3 alert threshold).
- **Single tumbling window** per monitor (500 rows); no multi-resolution
  windows, no streaming infra, no retraining loop, no auth, no dashboard
  (out of scope by design).
