# ModelWatch: an ML observability and drift-detection harness

I built ModelWatch to answer a question I kept running into: when people say
a drift detector "works", how would you actually know? Most monitoring demos
show PSI and KS numbers going up and call it a day. Here I score the detector
itself: a seeded scenario generator replays held-out UCI Adult rows, injects
labeled drift episodes, and the harness computes precision, recall and F1 for
the detector at both the window level and the episode level.

The harness monitors a trained model in a mock production setting: it ingests
scoring traffic, detects covariate and prediction drift (PSI, two-sample KS)
against a training baseline, tracks label-free health signals (missing-value
rate, out-of-range rate, category novelty), runs golden-dataset regression
gates (accuracy, ROC-AUC, calibration, per-slice accuracy), and exposes all
of it over FastAPI with Prometheus-style metrics. The statistic
implementations are my own and are verified against scipy and naive oracle
loops. All measured numbers live in [RESULTS.md](RESULTS.md) and
`results/*.json`.

Headline results: window-level drift detection scored 0.99 precision at 1.00
recall against ground truth (all 32 injected episodes caught, 1 false alarm),
my KS statistic matches scipy exactly, and the release gates correctly pass
the healthy model and fail a deliberately degraded one on 4 of 10 gates.

## How the pieces fit together

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

- `modelwatch/stats.py`: my implementations of PSI (10-bin quantile binning,
  rare-category pooling for categoricals), two-sample KS (statistic exact vs
  scipy; asymptotic Kolmogorov p-value), and ECE.
- `modelwatch/monitor.py`: baseline profiling plus tumbling-window monitoring
  with configurable thresholds.
- `modelwatch/gates.py`: golden-dataset regression gates, including per-slice
  accuracy on sex and race so a release that quietly degrades one group gets
  caught.
- `modelwatch/api.py`: FastAPI service with a consistent
  `{"ok", "data", "error"}` envelope and Prometheus counters, histogram and
  gauge.
- `model/train.py`: trains the reference model and a deliberately degraded
  twin, persists the baseline profile and data splits.
- `eval/`: seeded scenario generator plus ground-truth drift evaluation.
- `bench/`: throughput and latency benchmarks, statistic-oracle checks.

## The model under observation

The monitoring subject is a scikit-learn `LogisticRegression` (max_iter=2000)
on median-imputed and standardized numerics plus constant-imputed one-hot
categoricals (`handle_unknown="ignore"`), trained on UCI Adult / "Census
Income" (OpenML `adult` version 2, 48,842 rows, a 1994 US Census extract).
Splits are 60% train / 20% golden / 20% traffic, stratified, seed 42, fetched
via `sklearn.datasets.fetch_openml` into a gitignored `data/` cache.

On the golden split it scores accuracy 0.8545, ROC-AUC 0.9089, ECE 0.0101,
with per-slice accuracy from 0.8121 (race=Asian-Pac-Islander, n=314) to
0.9287 (sex=Female, n=3,238). A degraded twin, retrained with 40% of training
labels flipped (seed 42), exists to prove the gates fail in the bad
direction.

A note on the data: UCI Adult encodes 1994 census demographics and historical
income disparities, and slice accuracy differs by up to ~11 points across
sex/race groups. The task (income >50K) is used here purely as a realistic
monitoring subject; the per-slice gates exist precisely to surface those
disparities. Nobody should use this model for real decisions about people.

## Running it

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

## Limitations

- Drift ground truth is injected synthetically (demographic resampling,
  feature scaling, missing-value spikes, novel categories, score-weighted
  resampling) on real held-out UCI Adult rows. So the eval measures detector
  correctness against known drift, not real-world drift, which is messier;
  treat the precision/recall numbers as an upper bound.
- API latency is measured in-process with FastAPI's TestClient, so it
  excludes the network hop, uvicorn workers, and wire serialization.
- My KS statistic matches scipy exactly, but the p-value uses the Kolmogorov
  limit distribution and deviates up to 0.0207 from scipy's finite-n `asymp`
  mode. The worst case sits near p=0.65, far from the 1e-3 alert threshold,
  so it never changes a verdict here.
- One tumbling window per monitor (500 rows). No multi-resolution windows,
  no streaming infrastructure, no retraining loop, no auth, no dashboard. I
  kept the scope on detection quality, not platform plumbing.
