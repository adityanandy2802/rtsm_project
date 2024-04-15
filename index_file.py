import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf, pacf
import statsmodels as sm
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import streamlit as st
from helper_functions import lag, plot_bar, plot_line, plot_train_test, plot_predictions, plot_chart
from models import AR, ARMA, evaluate_arima_model, sarimax

from datasets import passengers_url, shampoo_url, stockmarket_url

st.title("Evaluating Time Series Models Across Diverse Datasets")
options_dict = {"Airplane Passengers Data": passengers_url, "Shampoo Data": shampoo_url, "Stock Market Data": stockmarket_url}
options = ["", "Airplane Passengers Data", "Shampoo Data", "Stock Market Data"]
selected_dataset = st.selectbox("Select a Time Series dataset:", options)

if selected_dataset == "Airplane Passengers Data":
    results = {}
    url = options_dict[selected_dataset]
    plot_chart(url, "#Passengers", "Months", "Passenger Data")
    results["AR"] = AR(url, "#Passengers", "Months", "Passenger Data")
    st.write(results["AR"])

    results["ARMA"] = ARMA(url, "#Passengers", "Months", "Passenger Data")
    st.write(results["ARMA"])

    results["ARIMA"] = evaluate_arima_model(url, variable = "#Passengers", xparam = "Months", title = "Passenger Data", frcst_stp = 16)
    st.write(results["ARIMA"])

    results["SARIMA"] = sarimax(url, variable = "#Passengers", xparam = "Months", title = "Passenger Data")
    st.write(results["SARIMA"])

    st.plotly_chart(plot_bar(x = list(results.keys()), y = list(results.values()), title = "Results", x_title = "Models", y_title = "MAPE", hline = False))

if selected_dataset == "Shampoo Data":
    results = {}
    url = options_dict[selected_dataset]
    results["AR"] = AR(url, "Sales of shampoo over a three year period", "Months", "Shampoo Data")
    st.write(results["AR"])

    results["ARMA"] = ARMA(url, "Sales of shampoo over a three year period", "Months", "Shampoo Data")
    st.write(results["ARMA"])

    results["ARIMA"] = evaluate_arima_model(url, variable = "Sales of shampoo over a three year period", xparam = "Months", title = "Shampoo Data", frcst_stp = 1)
    st.write(results["ARIMA"])

    results["SARIMA"] = sarimax(url, variable = "Sales of shampoo over a three year period", xparam = "Months", title = "Shampoo Data")
    st.write(results["SARIMA"])

    st.plotly_chart(plot_bar(x = list(results.keys()), y = list(results.values()), title = "Results", x_title = "Models", y_title = "MAPE", hline = False))

if selected_dataset == "Stock Market Data":
    results = {}
    url = options_dict[selected_dataset]
    results["AR"] = AR(url, "Close Price", "TimeStamp", "Stock Market Data")
    st.write(results["AR"])

    results["ARMA"] = ARMA(url, "Close Price", "TimeStamp", "Stock Market Data")
    st.write(results["ARMA"])

    results["ARIMA"] = evaluate_arima_model(url, variable = "Close Price", xparam = "TimeStamp", title = "Stock Market Data", frcst_stp = 1)
    st.write(results["ARIMA"])

    results["SARIMA"] = sarimax(url, variable = "Close Price", xparam = "TimeStamp", title = "Stock Market Data")
    st.write(results["SARIMA"])

    st.plotly_chart(plot_bar(x = list(results.keys()), y = list(results.values()), title = "Results", x_title = "Models", y_title = "MAPE", hline = False))

