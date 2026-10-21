import pandas as pd


def static_threshold_scores(s, train_end):
    """Score each point by its distance from the training mean, in training standard deviations.

    Mean and standard deviation are learned only from the first `train_end`
    points (the clean training period), so nothing from the test period leaks in.
    """
    train = s.iloc[:train_end]
    mu, sd = train.mean(), train.std()
    return (s - mu).abs() / sd
