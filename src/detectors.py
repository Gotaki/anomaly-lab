import pandas as pd


def static_threshold_scores(s, train_end):
    """Score each point by its distance from the training mean, in training standard deviations.

    This is the baseline detector: one fixed "normal" level for all times of day.
    Mean and standard deviation are learned only from the first `train_end`
    points (the clean training period), so nothing from the test period leaks in.

    Returns a Series of scores. A score of 3 means "3 standard deviations from
    normal". Flag points with `scores > k` for whatever k you choose.
    """
    train = s.iloc[:train_end]               # clean data only
    mu, sd = train.mean(), train.std()       # what "normal" looks like
    return (s - mu).abs() / sd               # absolute distance, in standard deviations
