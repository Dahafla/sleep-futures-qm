import numpy as np
import pandas as pd
from .config import TARGET_SLEEP_HOURS


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Input: daily df indexed by date with columns including:
        hours_slept, sleep_index, bedtime_minutes, sleep_efficiency

    Output: df with:
        - features in feature_cols
        - target      = next-day SleepIndex (numeric)
        - target_cls  = next-day direction: +1 (above target), -1 (at/below target)
    """
    df = df.copy()

    # Rolling means of SleepIndex (keep it simple: 3 & 7 days)
    for window in (3, 7):
        df[f"sleep_index_rolling_{window}d"] = (
            df["sleep_index"].rolling(window).mean()
        )

    # Sleep deficit: how much below target (0 if at/above target)
    df["sleep_deficit"] = np.maximum(0.0, TARGET_SLEEP_HOURS - df["hours_slept"])

    # Circadian drift: deviation from median bedtime
    median_bedtime = df["bedtime_minutes"].median()
    df["circadian_drift"] = df["bedtime_minutes"] - median_bedtime

    # Weekend indicator (Sat/Sun -> 1)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Day-of-week one hot
    dow = pd.get_dummies(df.index.dayofweek, prefix="dow", dtype=int)
    dow.index = df.index
    df = pd.concat([df, dow], axis=1)

    # Numeric target: next-day SleepIndex
    df["target"] = df["sleep_index"].shift(-1)

    # Binary direction: +1 if next-day SleepIndex > 0, else -1
    df["target_cls"] = np.where(df["target"] > 0, 1, -1)

    feature_cols = [
        "sleep_index",
        "hours_slept",
        "sleep_efficiency",
        "sleep_deficit",
        "circadian_drift",
        "is_weekend",
        "sleep_index_rolling_3d",
        "sleep_index_rolling_7d",
    ] + list(dow.columns)

    df = df[feature_cols + ["target", "target_cls"]].dropna()

    return df, feature_cols
