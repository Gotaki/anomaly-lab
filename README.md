# Anomaly Detection Lab

Build and compare unsupervised anomaly detectors on server-style time-series
metrics, scored with leakage-free, event-based evaluation.

## Goal
Compare a static-threshold baseline against seasonality-aware and ML detectors
(Isolation Forest) and explain the trade-offs.

## Status
- [x] Synthetic data generator
- [x] Labelled anomaly injection
- [ ] Static threshold baseline
- [ ] Event-based evaluation
- [ ] Seasonal residual detector
- [ ] Isolation Forest
- [ ] Real data (Numenta Anomaly Benchmark)
- [ ] Results and write-up

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout
- `src/` reusable code (data generation, detectors, evaluation)
- `notebooks/` exploration and plots
- `results/` output tables

## Design decisions
- The first 7 days of data are kept clean and used as the training period.
- All randomness uses fixed seeds so results are reproducible.
- Anomalies are placed at times where they look normal against the overall range but abnormal for the time of day, so a static threshold cannot trivially detect them.

## Results
To be added.