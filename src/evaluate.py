import numpy as np


def runs(mask, merge_gap=0):
    """Contiguous True runs as inclusive (start, end) positions.

    Example: mask = [0, 1, 1, 0, 0, 1] gives [(1, 2), (5, 5)].
    With merge_gap=2 those runs are only 2 False points apart, so they merge: [(1, 5)].
    """
    idx = np.flatnonzero(np.asarray(mask))  # positions of every True value
    if len(idx) == 0:
        return []  # nothing flagged

    out = []
    start = prev = int(idx[0])  # the current run begins at the first True
    for i in idx[1:]:
        i = int(i)
        # i - prev - 1 = number of False points between the two True points.
        # If that gap is larger than merge_gap, close the current run and start a new one.
        if i - prev > merge_gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))  # close the final run
    return out


def event_metrics(flags, labels, train_end, per_day, tol=6):
    """Score detector output per alert event rather than per point.

    flags:     boolean array, True where the detector says "anomaly"
    labels:    boolean array, True where there is a real anomaly (ground truth)
    train_end: points before this index are training data and are never scored
    per_day:   points per day (288 at 5-minute sampling), used for alerts per day
    tol:       slack in points (6 = 30 minutes) for timing and for merging alerts

    Definitions:
    - DETECTED: a true anomaly window is detected if any flag lands in it (+/- tol).
    - EVENT: flags within `tol` points of each other merge into one alert.
    - TRUE POSITIVE event: an alert that overlaps a real window.
    - FALSE ALARM: an alert that overlaps no real window.
    """
    flags = np.asarray(flags, dtype=bool).copy()  # copy so we don't change the caller's array
    labels = np.asarray(labels, dtype=bool)
    flags[:train_end] = False  # never score the training period

    windows = runs(labels)               # real anomaly windows
    events = runs(flags, merge_gap=tol)  # detector alerts

    # How many real windows had at least one flag inside them (with slack)?
    detected = sum(flags[max(0, a - tol): b + tol + 1].any() for a, b in windows)

    # How many alerts overlap a real window (with slack)? The rest are false alarms.
    tp_events = sum(
        any(es <= b + tol and ee >= a - tol for a, b in windows) for es, ee in events
    )
    fp_events = len(events) - tp_events

    test_days = (len(flags) - train_end) / per_day  # length of the scored period in days

    # Recall: what fraction of real windows did we catch?
    recall = detected / len(windows) if windows else float("nan")
    # Precision: what fraction of alerts were real? (0 if the detector never alerted)
    precision = tp_events / len(events) if events else 0.0
    # F1: harmonic mean of precision and recall (0 if both are 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "events": len(events),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_day": fp_events / test_days,
    }
