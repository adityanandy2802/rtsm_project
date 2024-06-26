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
from helper_functions import lag, plot_bar, plot_line, plot_train_test, plot_predictions

import warnings
warnings.filterwarnings("ignore")

def AR(url = None, variable = None, xparam = None, title = None):
    true_df = pd.read_csv(url)
    df = true_df.copy()
    time_series_data = true_df[variable].values

    st.subheader("Autoregressive Model")
    nlags = st.slider(
                label = "Lags",  
                min_value = 0, 
                max_value = len(true_df),   
                value = len(true_df) - 10, 
                key = "nlags_slider_key",      
                step = 1,               
            )
    if (nlags > len(true_df) / 2):
        st.write("Please choose N lags less than 50'%' of sample size")
        return
    acf_values = acf(time_series_data, nlags=nlags)[1:]
    pacf_values = pacf(time_series_data, nlags=nlags)[1:]

    l_space = [i for i in range(1, len(acf_values) + 1)]
    pacf_cutoff = 0.05
    fig = plot_bar(x = l_space, y = pacf_values, title = "PACF Values", x_title = "Lag", y_title = "PACF Values", cutoff = pacf_cutoff)

    st.plotly_chart(fig)

    lag_list = []
    for i in range(len(pacf_values)):
        if abs(pacf_values[i]) > pacf_cutoff:
            lag_list.append(i + 1)
    col_list, df = lag(df, variable, lag_list)
    
    df = df.dropna()

    x = df[col_list]
    y = df[variable]

    TRAIN_SIZE = st.slider(
                label="Train Size",  
                min_value = 0, 
                max_value = len(true_df) - nlags,  
                value = len(true_df) - nlags - 1,  
                key="ar_slider_key",      
                step = 1,               
            )
    x_train, y_train = x.iloc[: TRAIN_SIZE], y.iloc[: TRAIN_SIZE]
    x_test, y_test = x.iloc[TRAIN_SIZE: ], y.iloc[TRAIN_SIZE: ]
    st.plotly_chart(plot_train_test(y_train, y_test))

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    train_range = [i for i in range(len(y_train))]
    test_range = [i for i in range(len(y_train), len(y_train) + len(y_test))]
    pred_range = test_range
    st.plotly_chart(plot_predictions(train_range,\
                                    test_range,\
                                    pred_range,\
                                    y_train,\
                                    y_test,\
                                    predictions,\
                                    title = "AR results",\
                                    xlabel = xparam,\
                                    ylabel = variable
                                ))

    error = mape(df[variable].iloc[pred_range], predictions)
    return error

def ARMA(url = None, variable = None, xparam = None, title = None):
    true_df = pd.read_csv(url)
    df = true_df.copy()

    st.subheader("Autoregressive Moving Average (ARMA)")
    window_size = st.slider(
                label="Window Size",  
                min_value = 0, 
                max_value = 20,  
                value = 10,  
                key="arma_slider_key",      
                step = 1,               
            )
    df['rolling_mean'] = df[variable].rolling(window = window_size).mean()
    df.dropna(inplace = True)
    st.plotly_chart(plot_predictions(x_train = [i for i in range(len(df[variable]))],\
                                    x_test = [i for i in range(len(df[variable]))],\
                                    y_train = df[variable],\
                                    y_test = df["rolling_mean"],\
                                    title = "Trendline",\
                                    xlabel = xparam,\
                                    ylabel = variable
                                    ))
    df["error"] = df[variable] - df["rolling_mean"]
    st.plotly_chart(plot_line(x = [i for i in range(len(df["error"]))], y = df["error"], title = "Deviation from Rolling Mean", x_title = xparam, y_title = variable))
    
    df_save = df

    nlags = st.slider(
                label = "Lags",  
                min_value = 0, 
                max_value = len(df),   
                value = 5, 
                key = "nlags_arma_slider_key",      
                step = 1,               
            )
    pacf_values = pacf(df["error"].values, nlags=20)[1:]
    st.plotly_chart(plot_bar(x = [i for i in range(1, nlags + 1)],\
                            y = pacf_values,\
                            title = "PACF Values for Error Term",\
                            x_title = xparam,\
                            y_title = variable
                    ))
    lag_list = []
    for i in range(len(pacf_values)):
        if abs(pacf_values[i]) > 0.05:
            lag_list.append(i + 1)
    col_list, df_error = lag(df, "error", lag_list)

    df = df_save
    pacf_values_trend = pacf(df["rolling_mean"].values, nlags = nlags)[1:]
    st.plotly_chart(plot_bar(x = [i for i in range(1, nlags + 1)],\
                            y = pacf_values_trend,\
                            title = "PACF Values for Trend",\
                            x_title = xparam,\
                            y_title = variable
                    ))
    trend_lag_list = []
    for i in range(len(pacf_values)):
        if abs(pacf_values[i]) > 0.05:
            trend_lag_list.append(i + 1)
    trend_col_list, df_trend = lag(df, "rolling_mean", trend_lag_list)

    df_error.dropna(inplace = True)
    x = df_error[col_list]
    y = df_error["error"]

    df_trend.dropna(inplace = True)
    trend_x = df_trend[trend_col_list]
    trend_y = df_trend["rolling_mean"]

    TRAIN_SIZE = st.slider(
                label="Train Size",  
                min_value = 0, 
                max_value = len(true_df) - nlags,  
                value = len(true_df) - nlags - 1,  
                key="arma_train_slider_key",      
                step = 1,               
            )
    x_train, y_train = x.iloc[: TRAIN_SIZE], y.iloc[: TRAIN_SIZE]
    x_test, y_test = x.iloc[TRAIN_SIZE: ], y.iloc[TRAIN_SIZE: ]

    trend_x_train, trend_y_train = trend_x.iloc[: TRAIN_SIZE], trend_y.iloc[: TRAIN_SIZE]
    trend_x_test, trend_y_test = trend_x.iloc[TRAIN_SIZE: ], trend_y.iloc[TRAIN_SIZE: ]

    model_error = LinearRegression()
    model_error.fit(x_train, y_train)
    model_trend_line = LinearRegression()
    model_trend_line.fit(trend_x_train, trend_y_train)

    error_pred = model_error.predict(x_test)
    trend_line_pred = model_trend_line.predict(trend_x_test)
    trend_line_pred = trend_line_pred[-min(len(trend_line_pred), len(error_pred)): ]
    error_pred = error_pred[-min(len(trend_line_pred), len(error_pred)): ]

    prediction = trend_line_pred + error_pred

    train_range = [i for i in range(TRAIN_SIZE)]
    test_range = [i for i in range(TRAIN_SIZE, TRAIN_SIZE + len(y_test))]
    pred_range = [i for i in range(TRAIN_SIZE + len(y_test) - min(len(trend_line_pred), len(error_pred)), TRAIN_SIZE + len(y_test))]
    st.plotly_chart(plot_predictions(x_train = train_range,\
                                    x_test = test_range,\
                                    pred_range = pred_range,\
                                    y_train = df[variable].iloc[train_range],\
                                    y_test = df[variable].iloc[test_range],\
                                    predictions = prediction,\
                                    title = "ARMA Results",\
                                    xlabel = xparam,\
                                    ylabel = variable
                        ))
    error = mape(df[variable].iloc[pred_range], prediction)
    return error


def evaluate_arima_model(url = None, xparam = None, variable = None, frcst_stp = None, title = None):
    true_df = pd.read_csv(url)
    df = true_df.copy()
    X = true_df[variable].values

    st.subheader("ARIMA Model")
    p = st.slider(
                label="Choose p value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="arima_p_slider_key",      
                step = 1,               
            )
    
    q = st.slider(
                label="Choose q value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="arima_q_slider_key",      
                step = 1,               
            )
    
    d = st.slider(
                label="Choose d value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="arima_d_slider_key",      
                step = 1,               
            )
    
    arima_order = (p, d, q)

    TRAIN_SIZE = len(df[variable])
    
    train, test = X[0: TRAIN_SIZE], X[TRAIN_SIZE: ]
    history = [x for x in train]
   
    predictions = list()
    model = sm.tsa.arima.model.ARIMA(history, order = arima_order)
    model_fit = model.fit()

    pred = model_fit.predict(start = 1, end = len(df[variable]) - 1, typ = "levels")

    train_range = [i for i in range(1, len(train))]
    pred_range = train_range

    # st.plotly_chart(plot_train_test(train, pred))
    st.plotly_chart(plot_predictions(x_train = train_range,\
                                    pred_range = pred_range,\
                                    y_train = train[1:],\
                                    predictions = pred,\
                                    title = "ARIMA results",\
                                    xlabel = xparam,\
                                    ylabel = variable
                                ))

    error = mape(train[1:], pred)
    return error

def sarimax(url = None, xparam = None, variable = None, title = None):
    true_df = pd.read_csv(url)
    df = true_df.copy()
    y = true_df[variable].values

    st.subheader("SARIMAX Model")
    # decomposition = sm.tsa.seasonal_decompose(y, model='additive', period = 30)
    # p = d = q = range(0, 2)
    # pdq = list(itertools.product(p, d, q))
    # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))] 
    # for param in pdq:
    #     for param_seasonal in seasonal_pdq:
    #         try:
    #             mod = sm.api.tsa.statespace.SARIMAX(endog = train.Rides,\
    #                                             trend='n',\
    #                                             order=(1,0,1),\
    #                                             seasonal_order=(1,0,1,12))

    #             results = mod.fit()

    #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
    #         except:
    #             continue

    p = st.slider(
                label="Choose p value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="sarima_p_slider_key",      
                step = 1,               
            )
    
    d = st.slider(
                label="Choose d value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="sarima_d_slider_key",      
                step = 1,               
            )
    
    q = st.slider(
                label="Choose q value",  
                min_value = 0, 
                max_value = 15,  
                value = 2,  
                key="sarima_q_slider_key",      
                step = 1,               
            )
    
    mod = sm.api.tsa.statespace.SARIMAX(y,
                                    order = (p, d, q),
                                    seasonal_order=(1, 0, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()

    start_forecast = st.slider(
                label = "Forecast start",  
                min_value = 0, 
                max_value = len(y),  
                value = 5,  
                key = "sarima_forecast_start",      
                step = 1,               
            )
    pred = results.get_prediction(start=start_forecast, dynamic=False)
    # pred_ci = pred.conf_int()

    st.plotly_chart(plot_predictions(\
                    x_train = [i for i in range(len(y))],\
                    pred_range = [i for i in range(start_forecast, start_forecast + len(pred.predicted_mean))],\
                    y_train = y,\
                    predictions = pred.predicted_mean,\
                    title = "Sarimax Results",\
                    xlabel = xparam,\
                    ylabel = variable
                ))

    error = mape(y[start_forecast: ], pred.predicted_mean)
    return error