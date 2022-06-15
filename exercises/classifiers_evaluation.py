from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
pio.renderers.default= "browser"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset("C:\\Users\\omeri\\Documents\\Omer\\Semester D\\IML\\IML.HUJI\\datasets\\" + f)
        # print(X)
        # print(Y)
        def callaback_func(per:Perceptron, x:np.ndarray,y:int):
            losses.append(per.loss(X, Y))
            # calculate the loss in the fit


        # Fit Perceptron and record loss in each fit iteration
        losses = []
        per = Perceptron(include_intercept=True,callback=callaback_func) #maybe not false
        per.fit(X, Y)

        # Plot figure of loss as function of fitting iteration
        # print(losses)
        fig = go.Figure((go.Scatter(x=np.linspace(1, 1000, 1000),
                                    y=np.asarray(losses), mode="lines",
                                    name="Mean Prediction",
                                    marker=dict(color="blue", )),))
        fig.update_layout(
            title_text='Question 3.1 - Loss as a function of iterations')
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, Y = load_dataset("C:\\Users\\omeri\\Documents\\Omer\\Semester D\\IML\\IML.HUJI\\datasets\\" + f)


        # Fit models and predict over training set
        print(X)
        print(Y)
        lda = LDA()
        lda.fit(X, Y)
        lda_pred = lda.predict(X)
        GNB = GaussianNaiveBayes()
        GNB.fit(X, Y)
        gnb_pred = GNB.predict(X)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc = accuracy(Y, lda_pred)
        gnb_acc = accuracy(Y, gnb_pred)
        print(lda_acc, gnb_acc)

        # Add traces for data-points setting symbols and colors
        symbols = np.array(["circle", "square", "triangle-up"])

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Gaussian Naive Bayes with accuracy {gnb_acc}",
        f"Linear Discriminant Analysis with accuracy {lda_acc}"))

        # Add traces for data-points setting symbols and colors
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array(
            [-.4, .4])
        fig.add_traces([decision_surface(GNB.predict, lims[0], lims[1],
                                         showscale=False),
                        decision_surface(lda.predict, lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=Y,
                                               symbol=symbols[Y]),
                                   showlegend=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=Y,
                                               symbol=symbols[Y]),
                                   showlegend=False)],
                       rows=[1, 1, 1, 1], cols=[1, 2, 1, 2]) \
            .update_layout(
            title_text=f"Part 3 - Bayes Classifiers - {f.split('.')[0]}", )

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode='markers',
                       marker=dict(color="black", symbol="x"),
                       showlegend=False), row=1, col=2, )
        fig.add_trace(
            go.Scatter(x=GNB.mu_[:, 0], y=GNB.mu_[:, 1], mode='markers',
                       marker=dict(color="black",
                                   symbol="x"),
                       showlegend=False), row=1, col=1, )


        # Add ellipses depicting the covariances of the fitted Gaussians

        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(GNB.mu_[0], np.diag(GNB.vars_[0])),
                      row=1, col=1, )
        fig.add_trace(get_ellipse(GNB.mu_[1], np.diag(GNB.vars_[1])),
                      row=1, col=1, )
        fig.add_trace(get_ellipse(GNB.mu_[2], np.diag(GNB.vars_[2])),
                      row=1, col=1, )

        # fig.show()






if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    # for the quiz question
    # s = [(0,0),(1,0),(2,1),(3,1),(4,1),(5,1),(6,2),(7,2)]
    x1 = [0,1,2,3,4,5,6,7]
    y1 = [0,0,1,1,1,1,2,2]
    gnb1 = GaussianNaiveBayes()
    gnb1.fit(np.asarray(x1), np.asarray(y1))
    # print(gnb1.pi_)
    # print(gnb1.mu_)
    # S2 = {([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1),
    #      ([3, 4], 1)}
    x2 = [[1, 1], [1, 2], [2, 3] , [2, 4] , [3, 3], [3, 4]]
    y2 = [0,0,1,1,1,1]
    gnb2 = GaussianNaiveBayes()
    gnb2.fit(np.asarray(x2), np.asarray(y2))
    # print(gnb2.vars_)

