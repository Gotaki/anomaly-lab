import numpy as np
import pandas as pd


def make_series(
    n_days=28,
    freq_min=5,
    seed=0,
    baseline=50.0,
    daily_amplitude=20.0,
    drift_per_step=0.002,
    noise_sd=2.5,
    start="2026-01-01",
):
    """Synthetic server metric: daily cycle + slow drift + Gaussian noise.

    Returns a pd.Series named "value" with a DatetimeIndex sampled every
    `freq_min` minutes for `n_days` days. The same seed gives the same series.
    """
    rng = np.random.default_rng(seed)  # seeded so results are reproducible

    # How many samples: per day, then in total
    points_per_day = 24 * 60 // freq_min
    n_points = points_per_day * n_days

    # Time axis: one timestamp every `freq_min` minutes
    idx = pd.date_range(start=start, periods=n_points, freq=f"{freq_min}min")

    # Step counter 0, 1, 2, ... used for the time-dependent parts
    t = np.arange(n_points)

    # Daily cycle: one full sine wave per day, trough at midnight, peak at noon
    daily = daily_amplitude * np.sin(2 * np.pi * t / points_per_day - np.pi / 2)

    # Slow upward drift across the whole period
    drift = drift_per_step * t

    # Random jitter around the signal
    noise = rng.normal(0, noise_sd, n_points)

    values = baseline + daily + drift + noise
    return pd.Series(values, index=idx, name="value")
    
def inject_anomalies(s, anomalies):
    """Add labelled anomalies to a copy of the series.

    anomalies: list of (start, length, size), where start is a position index.
    Returns (series_with_anomalies, labels); labels is a boolean Series.
    """
    out = s.copy()
    labels = pd.Series(False, index=s.index, name="is_anomaly")
    for start, length, size in anomalies:
        out.iloc[start : start + length] += size
        labels.iloc[start : start + length] = True
    return out, labels