import numpy as np
from IMLearn import BaseModule


class L2(BaseModule):
    """
    Class representing the L2 module

    Represents the function: f(w)=||w||^2_2
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L2 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        return np.linalg.norm(self.weights, ord=2) ** 2

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L2 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L2 derivative with respect to self.weights at point self.weights
        """
        #
        return 2*self.weights


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L1 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        return np.linalg.norm(self.weights, ord=1)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L1 derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L1 derivative with respect to self.weights at point self.weights
        """
        # choose the sub gradient to be zero
        grad = np.sign(self.weights)
        return grad


class LogisticModule(BaseModule):
    """
    Class representing the logistic regression objective function

    Represents the function: f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a logistic regression module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, X: np.ndarray, y: np.ndarray,
                       **kwargs) -> np.ndarray:
        """
        Compute the output value of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # - (1/m) sum_i^m[y*<x_i,w> - log(sigmoid(<x_i,w>))]

        # total = 0
        # for i in range(X.size):
        #     total += y[i] * np.inner(X[i], self.weights) - np.log(
        #         1 / (1 + np.exp(-np.inner(X[i], self.weights))))
        # return np.asarray(-1 / X.size * total)
        m = X.shape[0]
        sigmoid = 1 + np.exp(np.inner(X, self.weights))
        to_sum = y * np.inner(X, self.weights) - np.log(sigmoid)
        return np.asarray(- (1 / m) * np.sum(to_sum))

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray,
                         **kwargs) -> np.ndarray:
        """
        Compute the gradient of the logistic regression objective function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (n_features,)
            Derivative of function with respect to self.weights at point self.weights
        """
        divert = np.exp(self.weights @ X.T) / (1 + np.exp(self.weights @ X.T))
        return - (y @ X - X.T @ divert) / X.shape[0]


class RegularizedModule(BaseModule):
    """
    Class representing a general regularized objective function of the format:
                                    f(w) = F(w) + lambda*R(w)
    for F(w) being some fidelity function, R(w) some regularization function and lambda
    the regularization parameter
    """

    def __init__(self,
                 fidelity_module: BaseModule,
                 regularization_module: BaseModule,
                 lam: float = 1.,
                 weights: np.ndarray = None,
                 include_intercept: bool = True):
        """
        Initialize a regularized objective module instance

        Parameters:
        -----------
        fidelity_module: BaseModule
            Module to be used as a fidelity term

        regularization_module: BaseModule
            Module to be used as a regularization term

        lam: float, default=1
            Value of regularization parameter

        weights: np.ndarray, default=None
            Initial value of weights

        include_intercept: bool default=True
            Should fidelity term (and not regularization term) include an intercept or not
        """
        super().__init__()
        self.fidelity_module_, self.regularization_module_, self.lam_ = fidelity_module, regularization_module, lam
        self.include_intercept_ = include_intercept

        if weights is not None:
            self.weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the regularized objective function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """

        return self.fidelity_module_.compute_output(**kwargs) + self.lam_*self.regularization_module_.compute_output(**kwargs)

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            Derivative with respect to self.weights at point self.weights
        """
        # self.fidelity_module_.weights = self.weights
        # self.regularization_module_.weights = self.weights
        # return self.fidelity_module_.compute_jacobian(**kwargs) + self.lam_*self.regularization_module_.compute_jacobian(**kwargs)

        regular_model = self.regularization_module_.compute_jacobian(**kwargs)
        if self.include_intercept_:
            regular_model = np.insert(regular_model, 0, 0)
        return self.fidelity_module_.compute_jacobian(**kwargs) + self.lam_ * regular_model

    @property
    def weights(self):
        """
        Wrapper property to retrieve module parameter

        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self.weights_

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters

        In case self.include_intercept_ is set to True, weights[0] is regarded as the intercept
        and is not passed to the regularization module

        Parameters
        ----------
        weights: ndarray of shape (n_in, n_out)
            Weights to set for module
        """
        self.weights_ = weights
        self.fidelity_module_.weights_ = weights
        if self.include_intercept_:
            self.regularization_module_.weights = np.delete(weights, 0, axis=0)
        else:
            self.regularization_module_.weights = weights
