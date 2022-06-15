from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    splits_arr_x = np.array_split(X, cv)
    splits_arr_y = np.array_split(y, cv)
    train_score_arr = []
    test_score_arr = []
    # check if its okay that the split is no randonly.
    for i in range(cv):
        train_x = np.concatenate(np.delete(splits_arr_x, i, axis=0))
        train_y = np.concatenate(np.delete(splits_arr_y, i, axis=0))
        test_x = splits_arr_x[i]
        test_y = splits_arr_y[i]
        estimator.fit(train_x, train_y)
        train_data_pred = estimator.predict(train_x)
        test_data_pred = estimator.predict(test_x)
        # 2 different prediction models- one for train score and one for validtion score
        train_score_arr.append(scoring(train_y, train_data_pred))
        test_score_arr.append(scoring(test_y, test_data_pred))

    return np.mean(train_score_arr), np.mean(test_score_arr)
