from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .config import TEST_SIZE_FRACTION, RANDOM_STATE


@dataclass
class ModelResults:
    model: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_cls: pd.Series     # -1 or 1
    y_test_cls: pd.Series
    y_test_reg: pd.Series      # numeric SleepIndex target (t+1)
    y_pred_cls: pd.Series      # predicted direction -1 or 1
    accuracy: float
    directional_accuracy: float   # same as accuracy here


def train_sleep_model(
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> ModelResults:
    """
    Train a CatBoost binary classifier to predict DIRECTION of next-day SleepIndex.

    target_cls: +1 (SleepIndex>0), -1 (SleepIndex<=0)
    """
    df = df_features.copy()

    # Chronological split
    n = len(df)
    test_size = max(int(n * TEST_SIZE_FRACTION), 1)
    train_size = n - test_size

    if train_size <= 10:
        train = df
        test = df.iloc[0:0]  # empty
    else:
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

    X_train = train[feature_cols]
    y_train_cls = train["target_cls"]

    X_test = test[feature_cols]
    y_test_cls = test["target_cls"]

    # Numeric SleepIndex target for payoff
    y_test_reg = test["target"] if len(test) > 0 else train["target"]

    # Ensure we have at least 2 classes in train
    unique_classes = sorted(y_train_cls.dropna().unique().tolist())

    if len(unique_classes) <= 1:
        # Baseline: always predict this class
        constant_class = unique_classes[0]

        if len(X_test) == 0:
            X_test_eval = X_train
            y_test_cls_eval = y_train_cls
            y_test_reg_eval = train["target"]
        else:
            X_test_eval = X_test
            y_test_cls_eval = y_test_cls
            y_test_reg_eval = y_test_reg

        y_pred_cls = pd.Series(
            constant_class, index=X_test_eval.index, name="y_pred_cls"
        )

        accuracy = float((y_pred_cls == y_test_cls_eval).mean())
        directional_accuracy = accuracy

        return ModelResults(
            model="constant_classifier",
            X_train=X_train,
            X_test=X_test_eval,
            y_train_cls=y_train_cls,
            y_test_cls=y_test_cls_eval,
            y_test_reg=y_test_reg_eval,
            y_pred_cls=y_pred_cls,
            accuracy=accuracy,
            directional_accuracy=directional_accuracy,
        )

    # Map {-1,1} -> {0,1} for CatBoost
    cls_to_int = {-1: 0, 1: 1}
    int_to_cls = {v: k for k, v in cls_to_int.items()}
    y_train_int = y_train_cls.map(cls_to_int)

    # Validation slice from end of training
    val_fraction = 0.2
    val_size = max(int(len(X_train) * val_fraction), 1)
    train_core = X_train.iloc[:-val_size] if len(X_train) > val_size else X_train
    train_core_y = y_train_int.iloc[:-val_size] if len(y_train_int) > val_size else y_train_int

    val_core = X_train.iloc[-val_size:]
    val_core_y = y_train_int.iloc[-val_size:]

    train_pool = Pool(train_core, train_core_y)
    eval_pool = Pool(val_core, val_core_y)

    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        iterations=1000,
        early_stopping_rounds=50,
        verbose=False,
    )

    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    # If no test set, evaluate on train
    if len(X_test) == 0:
        X_test_eval = X_train
        y_test_cls_eval = y_train_cls
        y_test_reg_eval = train["target"]
    else:
        X_test_eval = X_test
        y_test_cls_eval = y_test_cls
        y_test_reg_eval = y_test_reg

    # Predict class probabilities, then threshold at 0.5
    proba = model.predict_proba(X_test_eval)[:, 1]  # prob of class "1" (up)
    y_pred_int = (proba >= 0.5).astype(int)
    y_pred_cls = pd.Series(y_pred_int, index=X_test_eval.index).map(int_to_cls)
    y_pred_cls.name = "y_pred_cls"

    accuracy = float((y_pred_cls == y_test_cls_eval).mean())
    directional_accuracy = accuracy  # no neutrals now

    return ModelResults(
        model=model,
        X_train=X_train,
        X_test=X_test_eval,
        y_train_cls=y_train_cls,
        y_test_cls=y_test_cls_eval,
        y_test_reg=y_test_reg_eval,
        y_pred_cls=y_pred_cls,
        accuracy=accuracy,
        directional_accuracy=directional_accuracy,
    )
