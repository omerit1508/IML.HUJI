from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        # check all the labels
        self.mu_ = []
        for label in self.classes_:
            self.mu_.append(np.mean(X[y == label], axis=0))
        self.mu_ = np.asarray(self.mu_)

        # for the cov
        self.cov_ = np.zeros((X.shape[1],X.shape[1]))
        for label, mv in zip(self.classes_, self.mu_):
            temp_cov = np.zeros((X.shape[1], X.shape[1]))
            for sample in X[y == label]:
                sample, mv = sample.reshape(2,1), mv.reshape(2,1)
                temp_cov += (sample-mv).dot((sample-mv).T)
            self.cov_ += temp_cov
        self.cov_ = self.cov_/(X.shape[0] -self.classes_.size)
        #create the inv matrix
        self._cov_inv = inv(self.cov_)

        # pi
        num_of_samples = np.size(y)
        unique, counts = np.unique(y, return_counts=True)
        arr = np.asarray((unique, counts)).T
        self.pi_ = np.zeros(np.size(self.classes_))
        for i in range(np.size(self.pi_)):
            self.pi_[i] = (arr[i][1] / num_of_samples)




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # predict is with the linear ak, bk
        # ls = []
        # for label in self.classes_:
        #     a = self._cov_inv @ self.mu_[label]
        #     b = np.log(self.pi_[label]) -0.5* self.mu_[label] @ self._cov_inv @self.mu_[label]
        #     ls.append(X @ np.transpose(a) + b)
        # lis = np.asarray(ls).T
        # print(lis.shape) 300X3
        arr = self.likelihood(X)

        return np.argmax(arr, axis=1)
    # for each line, argmax of the feature

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        ls = []
        for label in self.classes_:
            a = self._cov_inv @ self.mu_[label]
            b = np.log(self.pi_[label]) - 0.5 * self.mu_[
                label] @ self._cov_inv @ self.mu_[label]
            ls.append(X @ np.transpose(a) + b)
        lis = np.asarray(ls).T
        return lis

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
        from ...metrics import misclassification_error
        y_prediction = self.predict(X)
        return misclassification_error(y, y_prediction)
