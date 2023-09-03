from datetime import date, timedelta

import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

date_temp = date.today()
date_of_today = date_temp.strftime("%Y-%m-%d")

start_date = date_temp - timedelta(days=300)
start_date = start_date.strftime("%Y-%m-%d")

date_of_month_later = date_temp + timedelta(days=30)
date_of_month_later = date_of_month_later.strftime("%Y-%m-%d")

date_of_week_later = date_temp + timedelta(days=7)
date_of_week_later = date_of_week_later.strftime("%Y-%m-%d")

date_of_day_later = date_temp + timedelta(days=1)
date_of_day_later = date_of_day_later.strftime("%Y-%m-%d")

st.title("The Stocks Prediction App")
selected_stock = st.text_input(label="Stock Symbol: ", value="TSLA")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start_date, date_of_today)
    data.reset_index(inplace=True)
    return data

try:
    data = load_data(selected_stock)
    stock_price_of_today = str(data.iloc[-1]["Close"])
    data = data[["Date", "Close"]]
    data = data.rename(columns={"Date":"ds", "Close":"y"})

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(data)

    future = m.make_future_dataframe(periods=33)
    forecast = m.predict(future)

    forecast['ds'] = pd.to_datetime(forecast['ds'])

    filtered_row = forecast[forecast['ds'] == date_of_month_later]

    stock_price_of_month_later = filtered_row['yhat'].values[0]

    filtered_row = forecast[forecast['ds'] == date_of_week_later]

    stock_price_of_week_later = filtered_row['yhat'].values[0]

    filtered_row = forecast[forecast['ds'] == date_of_day_later]

    stock_price_of_day_later = filtered_row['yhat'].values[0]

    st.write(f"Stock Forecast Till The Next Month: {stock_price_of_month_later}")
    st.write(f"Stock Forecast Till The Next Week: {stock_price_of_week_later}")
    st.write(f"Stock Forecast Till The Next Day: {stock_price_of_day_later}")
    st.write(f"Stock Forecast Today: {stock_price_of_today}")

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
except Exception as error:
    st.write(f"Please Write A Valid Stock Symbol! {error}")
