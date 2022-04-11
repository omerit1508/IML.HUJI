import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import sys
from utils import *


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df.drop(df[df.Temp < -20].index, inplace=True)
    # clean low temperature under -20
    # print(df["DayOfYear"])
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    read = load_data("C:\\Users\\omeri\Documents\\Omer\\Semester D\\IML\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    read['Year'] = read['Year'].astype(str)
    # to make it by year value, creating stringss
    israel_df = read[read.Country == 'Israel']
    graph = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year")
    graph.update_layout(
        title_text='Question 2 - Temperature as a function of the day of year (1995-2007)')
    graph.show()

    new_israel_df = israel_df.groupby('Month').agg('std')
    # print(new_israel_df)
    graph2 = px.bar(new_israel_df, y="Temp")
    graph2.update_layout(
        title_text='Question 2 - Temperature STD as a function of the Month')
    graph2.show()


    # Question 3 - Exploring differences between countries
    grouped_multiple = read.groupby(['Month', 'Country']).Temp.agg(['mean', 'std']).reset_index()
    # print(grouped_multiple)
    graph3 = px.line(grouped_multiple, x=["Month"], y="mean", error_y='std', color="Country")
    graph3.update_layout(
        title_text='Question 3 - The mean and standard deviation of temperatures in every country according to the months')
    graph3.show()

    # Question 4 - Fitting model for different values of `k`
    x_train, y_train, x_test, y_test = split_train_test(israel_df['DayOfYear'], israel_df['Temp'])
    ls_deg = []
    ls_loss = []
    for deg in range(1,11):
        pol = PolynomialFitting(deg)
        pol.fit(x_train.to_numpy(), y_train.to_numpy())
        loss = pol.loss(x_test.to_numpy(), y_test.to_numpy())
        ls_deg.append(deg)
        ls_loss.append(round(loss,2))

    np_deg = np.asarray(ls_deg)
    np_loss = np.asarray(ls_loss)
    print(ls_loss)

    graph4 = px.bar(x=np_deg, y=np_loss)
    graph4.update_layout(
        title_text='Question 4 - The loss as a function of different degrees of Polynome')
    graph4.update_yaxes(title_text='Loss', title_font_size=20)
    graph4.update_xaxes(title_text="deg of Polynome", title_font_size=20)
    graph4.show()


    # Question 5 - Evaluating fitted model on different countries
    poly_2 = PolynomialFitting(5)
    poly_2.fit(israel_df['DayOfYear'], israel_df['Temp'])
    loss_ls = []
    # countries = ['Jordan', 'South Africa', 'The Netherlands']
    countries = read.Country.unique().tolist()
    countries.remove("Israel")
    for country in countries:
        country_df = read[read.Country == country]
        loss = poly_2.loss(country_df['DayOfYear'], country_df['Temp'])
        loss_ls.append(loss)
    graph5 = px.bar(x=countries, y=loss_ls)
    graph5.update_layout(
        title_text='Question 5 - The loss of other countries based on a fitted model of Israel')
    graph5.update_yaxes(title_text='Loss', title_font_size=20)
    graph5.update_xaxes(title_text="country", title_font_size=20)
    graph5.show()
