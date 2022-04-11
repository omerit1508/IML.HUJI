from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
import sys
from utils import *
from scipy.stats import norm


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    s = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimator.fit(s)
    print("Uni_estimator mean and var:", (estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    m = 100
    x_axis = np.linspace(10, 1000, m)
    y_axis = []
    for i in x_axis:
        temp = s[0:int(i)]
        estimator_temp = UnivariateGaussian()
        estimator_temp.fit(temp)
        y_axis.append(np.abs(estimator_temp.mu_ - 10))

    fig1 = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=x_axis, y=y_axis, mode='lines',
                                marker=dict(color="black"),
                                showlegend=False, name='question 2')])
    fig1.update_yaxes(title_text='Loss')
    fig1.update_xaxes(title_text='Size Of Samples')
    fig1.update_layout(
        title_text="Question 2- connection between loss "
                   "and size of sample")

    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_arr = estimator.pdf(s)
    print(s)
    print(pdf_arr)

    fig2 = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=s, y=pdf_arr, mode='markers',
                                marker=dict(color="black"),
                                showlegend=False, name='question 3')])
    fig2.update_yaxes(title_text='PDFs values', title_font_size=20)
    fig2.update_xaxes(title_text='Sample Values', title_font_size=20)
    fig2.update_layout(
        title_text="$Question 3- PDF values - expecting to "
                   "see Normal distribution$",
        title_font_size=30)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0], [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    s2 = np.random.multivariate_normal(mu, sigma, 1000)

    estimator_multi = MultivariateGaussian()
    estimator_multi.fit(s2)
    print("Multi_estimator mean:", estimator_multi.mu_)
    print("Multi_estimator cov:", estimator_multi.cov_)
    pd2 = estimator_multi.pdf(s2)
    sc = estimator_multi.log_likelihood(mu, sigma, s2)


    # Question 5 - Likelihood evaluation
    ls = []
    mx = -10000
    t1, t2 = 0, 0
    f = np.linspace(-10, 10, 200)
    for i in f:
        temp = []
        for j in f:
            con_mu = np.array([i, 0, j, 0])
            val = estimator_multi.log_likelihood(con_mu, sigma, s2)
            temp.append(val)
            if val > mx:
                mx = val
                t1, t2 = round(i, 4), round(j, 4)
        ls.append(temp)

    fig = go.Figure(data=go.Heatmap(
        z=ls,
        x=f,
        y=f
    ))
    fig.update_layout(
        title="Loglikelihood Heatmap",
        xaxis_title="f3 values",
        yaxis_title="f1 valus"
    )
    fig.show()

    # Question 6 - Maximum likelihood
    print("max loglikelihood= ",  round(mx, 4))
    print("pair of values is:", t1, t2)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    n = np.array(
        [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1,
         -3, 1, -4, 1, 2, 1,
         -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3,
         -1, 0, 3, 5, 0, -2])
    r = UnivariateGaussian()
    r.fit(n)
    # print(r.log_likelihood(10, 1, n))
