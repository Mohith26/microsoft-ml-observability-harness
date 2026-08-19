# Resume Bullets

All numbers measured from actual runs (see RESULTS.md for reproduce commands).

- Built an ML observability harness (Python/FastAPI) for a census-income model: PSI/KS drift detection at 0.99 precision, 1.00 recall
  over 240 windows / 32 labeled injected-drift episodes (5 scenario types) on real UCI Adult data (synthetic drift, not real-world)

- Shipped golden-dataset regression gates (accuracy/AUC/ECE/per-slice) passing the healthy model and failing a label-flipped retrain
  (acc 0.854->0.831, AUC 0.909->0.875, ECE 0.010->0.262, worst slice 0.812->0.748), both failure directions proven in tests

- Processed 148K rows/sec in the drift engine; /score p95 3.1 ms (in-process FastAPI TestClient, not a network socket); verified by
  81 pytest tests at 100% coverage on the monitoring package, PSI/KS statistics validated against scipy and naive oracles

## Honesty tags

- Drift precision/recall are over INJECTED (synthetic) drift episodes replayed
  on real held-out UCI Adult rows — not real-world drift.
- API latencies are in-process FastAPI TestClient measurements (no network
  socket, no uvicorn), best of 5 runs.
- The KS statistic matches scipy exactly (0.0 deviation); the KS p-value uses
  the Kolmogorov limit distribution and deviates from scipy's finite-n asymp
  mode by at most 0.0207 (occurring at p≈0.65, far from the 1e-3 alert
  threshold). PSI matches a naive independent oracle to 1.1e-16.
