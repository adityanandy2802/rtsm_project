import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def lag(df, col_name, lag_list):
  col_list = []
  for num in lag_list:
    col_list.append("lag_" + str(num))
  for col in col_list:
    a = ""
    for i in col[::-1]:
      if i.isdigit():
        a += i
    a = a[::-1]
    df[col] = df[col_name].shift(int(a))
  return col_list, df

def plot_chart(url = None, variable = None, xparam = None, title = None):
    true_df = pd.read_csv(url)
    df = true_df.copy()
    time_series_data = true_df[variable].values

    fig = plot_line(y = true_df[variable], title = title, x_title = xparam, y_title = variable)
    st.plotly_chart(fig)

def plot_line(fig = None, x = None, y = None, x_title = None, y_title = None, title = None, vis = True):
    if fig is None:
        fig = px.line(
        x = x,
        y = y
        )
        fig.update_layout(title = title, xaxis_title = x_title, yaxis_title = y_title)
    return fig

def plot_bar(fig = None, x = None, y = None, title = None, x_title = None, y_title = None, cutoff = 0.05, vis = True, threshold_text = "", hline = True):
  trace = go.Bar(x = x, y = y)
  fig = go.Figure(data = [trace])
  fig.update_layout(title = title, xaxis_title = x_title, yaxis_title = y_title)
  if hline:
    fig.add_hline(y = cutoff, line_dash="dash", line_color = "red", annotation_text = threshold_text) # upper threshold
    fig.add_hline(y = -cutoff, line_dash="dash", line_color = "red", annotation_text = threshold_text) #lower threshold

  return fig

def plot_train_test(y_train = None, y_test = None, title = None, xlabel = None, ylabel = None):
  fig1 = px.line(x = [i for i in range (1, len(y_train) + 1)], y = y_train, labels = "Train Set")
  fig1.update_traces(line_color = "green")
  fig2 = px.line(x = [i for i in range(len(y_train), len(y_train) + len(y_test))], y = y_test, labels = "Test Set")
  fig2.update_traces(line_color = "red")
  fig = fig1.update_layout(showlegend=True)
  fig = fig.add_traces(fig2.data)
  fig.update_layout(showlegend = True)
  return fig

def plot_predictions(x_train = None, x_test = None, pred_range = None, y_train = None, y_test = None, predictions = None, title = None, xlabel = None, ylabel = None):
    fig1 = px.line(x = x_train, y = y_train, labels = "Train Set")
    fig1.update_traces(line_color = "green")
    fig2 = px.line(x = x_test, y = y_test, labels = "Test Set")
    fig2.update_traces(line_color = "red")
    fig3 = px.line(x = pred_range, y = predictions, labels = "Test Set")
    fig3.update_traces(line_color = "orange")

    fig = fig1.update_layout(showlegend=True)
    fig = fig.add_traces(fig2.data)
    fig = fig.add_traces(fig3.data)
    fig.update_layout(showlegend = True, title = title, xaxis_title = xlabel, yaxis_title = ylabel)

    return fig
