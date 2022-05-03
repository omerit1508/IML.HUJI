from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error as MSE


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        dic_mse = {}
        samp, feat_num = X.shape
        for feature in range(feat_num):
            thr1, thr_err1 = self._find_threshold(X[:, feature], y, 1)
            thr2, thr_err2 = self._find_threshold(X[:, feature], y, -1)
            if thr_err1 <= thr_err2:
                dic_mse[thr_err1] = [thr1, feature, 1]
            else:
                dic_mse[thr_err2] = [thr2, feature, -1]
        min_feat = min(dic_mse)
        self.threshold_ = dic_mse[min_feat][0]
        self.j_ = dic_mse[min_feat][1]
        self.sign_ = dic_mse[min_feat][2]





    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        size = X.shape[0]
        y_pred = np.zeros(size)
        for i in range(size):
            if X[i][self.j_] >= self.threshold_:
                y_pred[i] = self.sign_
            else:
                y_pred[i] = -self.sign_
        return y_pred


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> tuple:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thr_err = 1.0
        thr = 0.0
        samp_size = values.size
        new_arr = np.zeros(samp_size)
        for i in range(samp_size):
            for j in range(samp_size):
                if values[j] >= values[i]:
                    new_arr[j] = sign
                else:
                    new_arr[j] = -sign
            temp_mse = MSE(labels, new_arr, True)
            if temp_mse < thr_err:
                thr_err = temp_mse
                thr = values[i]
        return thr, thr_err




    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return MSE(y, y_pred)
