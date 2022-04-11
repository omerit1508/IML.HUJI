from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from collections import defaultdict
import sys
from utils import *
from IMLearn.metrics import mean_square_error as MSE


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    read = pd.read_csv(filename)
    zip_and_price = read[['zipcode', 'price']].copy()
    # creating copy of zip and prices to re vectorize zip code
    price_dic = defaultdict(list) #dictionary for price-zip dictonary
    for idx, row in zip_and_price.iterrows():
        price_dic[row['zipcode']].append(row['price']) # https://stackoverflow.com/questions/17426292/how-to-create-a-dictionary-of-two-pandas-dataframe-columns
    for i in price_dic:
        price_dic[i] = sum(price_dic[i]) // len(price_dic[i])
        float(price_dic[i])
    # changing the values of the dictionary to the mean sum of the prises
    # on the area of the same zipcode
    zips = read['zipcode'].copy()
    for i in range(len(zips)):
        if zips[i] > 0:
            zips[i] = price_dic[zips[i]]
    # chnaging zipcode to the values from the diction
    read.insert(0, 'average price by zipcode ', zips)
    # print(read['zipcodes_new'])
    read_drop = read.drop(columns=['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode'])
    read_drop['yr_renovated'] = read_drop.apply(
        lambda x: x['yr_built'] if x['yr_renovated'] == 0 else x[
            'yr_renovated'], axis=1)
    # feeling the year renoavted if didnt renovated to its year of built
    read_drop.dropna(inplace=True)
    # droping na values for the futrue- the pdf
    price = read_drop['price'].copy()  # double [[]] for column- bad
    read_drop = read_drop.drop(columns='price')
    return read_drop, price




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    ls = []
    y_std = y.std()
    # runing for each column
    for feature in X:
        df = X[[feature]].copy()
        df.insert(1, 'y', y)
        c_std = X[feature].std()
        corr = (df.cov() / (y_std * c_std))[feature][1]
        ls.append([feature, (df.cov() / (y_std * c_std))[feature][1]])
        # s_np = X[column].to_numpy()
        # print(s_np)
        # y_np = y.to_numpy().transpose()
        # print(y_np)
        print(X[feature])
        print(y)
        x_axis = feature
        fig = go.Figure(layout=dict(title=f'The correlation between {feature} and prices is {corr}'))
        fig.add_trace(go.Scatter(x=X[feature].to_numpy(), y=y.to_numpy(), name=feature, mode="markers",
                       showlegend=True))
        fig.update_yaxes(title_text='House Prices(M)', title_font_size=20)
        fig.update_xaxes(title_text=x_axis, title_font_size=20)
        fig.show()
        fig.write_image(output_path + f'\\{feature}.jpg')


    # print(ls)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, Y = load_data('C:\\Users\\omeri\\Documents\\Omer\\Semester D\\IML\\IML.HUJI\\datasets\\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(X, Y,"C:\\Users\\omeri\\Documents\\Omer\\Semester D\\IML course\\path")

    # Question 3 - Split samples into training- and testing sets.
    split_train_test(X, Y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    reg = LinearRegression()
    #creating linear reg instance, to use

    x_train, y_train, x_test, y_test = split_train_test(X, Y)

    av_loss, var_loss = [], []
    for p in range(10, 101):
        loss = []
        for time in range(10):
            indexes = np.random.randint(0, x_train.shape[0]-1, int((p/100)* x_train.shape[0]))
            train_sample_x = np.asarray(x_train)[indexes]
            train_sample_y = np.asarray(y_train)[indexes]
            fitted_model = LinearRegression().fit(train_sample_x,train_sample_y)
            loss.append(fitted_model.loss(x_test, y_test))
        av_loss.append(np.sum(np.asarray(loss))/10)
        var_loss.append(np.std(np.asarray(loss)))

    mean_pred, var_pred = np.asarray(av_loss), np.asarray(var_loss)
    x = np.linspace(10, 101,91)
    graph_5 = go.Figure((go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=x, y=mean_pred-2*var_pred, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=x, y=mean_pred+2*var_pred, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),))
    graph_5.update_layout( title_text="Question 4- Mean loss as a function of p%",
        title_font_size=30)
    graph_5.show()


    #test for quis
    # t_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    # t_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011,
    #                    345856.57131275, 563867.1347574, 395102.94362135])
    # print(MSE(t_true,t_pred))










