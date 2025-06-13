# utils.py
from typing import Tuple, Type
import numpy as np
import pandas as pd
import math


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_proportion: float = 0.75
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly partition features and targets into training and test subsets.
    """
    indices = list(X.index)
    total = len(indices)
    train_count = math.ceil(train_proportion * total)
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    return X_train, y_train, X_test, y_test


def run_estimator(
    estimator_cls: Type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    **kwargs
) -> np.ndarray:
    """
    Instantiate, fit, and predict using the provided estimator class and parameters.
    """
    model = estimator_cls(**kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_eval)
