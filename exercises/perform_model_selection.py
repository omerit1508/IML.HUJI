from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, \
    LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import sys

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y_true = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y_noise = y_true + eps
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(
                                                            y_noise),
                                                        2 / 3)
    fig1 = go.Figure([go.Scatter(x=np.concatenate(np.asarray(train_x)),
                                 y=np.asarray(train_y), mode="markers",
                                 marker=dict(color="red")
                                 , name="train"),
                      go.Scatter(x=np.concatenate(np.asarray(test_x)),
                                 y=np.asarray(test_y), mode="markers",
                                 marker=dict(color="blue"), name="test"),
                      go.Scatter(x=X, y=y_true, mode="lines+markers",
                                 marker=dict(color="black"),
                                 name="true")])
    fig1.update_layout(
        title=f"Question 1 - True data set and split data set with noise {noise} and number of sample {n_samples}")
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_ls = []
    val_ls = []
    x_ = np.linspace(0, 10, 11)

    for i in range(11):
        temp_train, temp_val = cross_validate(PolynomialFitting(i),
                                              np.asarray(train_x),
                                              np.asarray(train_y),
                                              mean_square_error)
        train_ls.append(temp_train)
        val_ls.append(temp_val)

    fig2 = go.Figure([go.Scatter(x=x_,
                                 y=np.asarray(train_ls), mode="lines+markers",
                                 marker=dict(color="red")
                                 , name="train list error"),
                      go.Scatter(x=x_,
                                 y=np.asarray(val_ls), mode="lines+markers",
                                 marker=dict(color="blue"), name="validation list error")])
    fig2.update_layout(
        title=f"Question 2 - train and validation error as the function of k with noise {noise} and number of sample {n_samples}")
    fig2.show()



    #Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    best_k = np.argmin(val_ls)
    min_val_err = val_ls[best_k]
    # finding the best k

    poly = PolynomialFitting(best_k)
    poly.fit(np.concatenate(np.asarray(train_x)), np.asarray(train_y))
    y_pred = poly.predict(np.concatenate(np.asarray(test_x)))
    k_test_err = mean_square_error(np.asarray(test_y), y_pred)

    print("The best k* is: " ,best_k, "and it's test_err is: ", np.round(k_test_err, 2))
    print("The validation error of the k* is: ", min_val_err)



def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data_x, data_y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = split_train_test(
        pd.DataFrame(data_x), pd.Series(data_y), n_samples / data_y.size)
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = 2 ** np.linspace(-4, 2, n_evaluations)
    ridge_train_err_ls = []
    ridge_valid_err_ls = []
    lasso_train_err_ls = []
    lasso_valid_err_ls = []
    for lam in lambdas:
        ridge_train_err, ridge_validation_err = cross_validate(
            RidgeRegression(lam), np.asarray(train_x),
            np.asarray(train_y), mean_square_error)
        ridge_train_err_ls.append(
            ridge_train_err), ridge_valid_err_ls.append(
            ridge_validation_err)
        lasso_train_err, lasso_validation_err = cross_validate(
            Lasso(alpha=lam, max_iter=10000), np.asarray(train_x),
            np.asarray(train_y), mean_square_error)
        lasso_train_err_ls.append(
            lasso_train_err), lasso_valid_err_ls.append(
            lasso_validation_err)
    # print(ridge_train_err_ls)
    fig3 = go.Figure([go.Scatter(x=lambdas,
                                 y=ridge_train_err_ls,
                                 mode="lines+markers",
                                 marker=dict(color="red")
                                 , name="ridge train err"),
                      go.Scatter(x=lambdas,
                                 y=ridge_valid_err_ls,
                                 mode="lines+markers",
                                 marker=dict(color="blue"),
                                 name="ridge validatdion err"),
                      go.Scatter(x=lambdas,
                                 y=lasso_valid_err_ls,
                                 mode="lines+markers",
                                 marker=dict(color="green"),
                                 name="lasso validatdion err"),
                      go.Scatter(x=lambdas, y=lasso_train_err_ls,
                                 mode="lines+markers",
                                 marker=dict(color="black"),
                                 name="lasso train err")])
    fig3.update_layout(
        title="Question 7 - Train and validation error as a function of lambda")
    fig3.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_ind_ridge = np.argmin(ridge_valid_err_ls)
    min_ind_lasso = np.argmin(lasso_valid_err_ls)
    lambda_ridge = lambdas[min_ind_ridge]
    lambda_lasso = lambdas[min_ind_lasso]

    ridge = RidgeRegression(lambda_ridge)
    ridge.fit(train_x, train_y)
    test_err_ridge = ridge.loss(test_x, test_y)

    lasso = Lasso(alpha=lambda_lasso, max_iter=50000)
    lasso.fit(train_x, train_y)
    y_pred = lasso.predict(test_x)
    test_err_lasso = mean_square_error(test_y, y_pred)

    lin_reg = LinearRegression()
    lin_reg.fit(train_x, train_y)
    test_err_lin_reg = lin_reg.loss(test_x, test_y)

    print("The ridge test error is: ", test_err_ridge, " and min lambda is: ", lambdas[min_ind_ridge])
    print("The lasso test error is : ", test_err_lasso, " and min lambda is: ", lambdas[min_ind_lasso])
    print("The linear regression test error is: ", test_err_lin_reg)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 5)
    select_regularization_parameter()
