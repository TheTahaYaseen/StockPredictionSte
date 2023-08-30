import subprocess
import os

# Determine the path to the packages.toml file
toml_path = os.path.join(os.path.dirname(__file__), 'packages.toml')

# Install packages listed in packages.toml
def install_packages_from_toml():
    try:
        subprocess.run(['pip', 'install', '-r', toml_path])
        print("Packages installed successfully.")
    except Exception as e:
        print("Error installing packages:", e)
install_packages_from_toml()

from datetime import date, timedelta

import sys
import subprocess

import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import requests
from bs4 import BeautifulSoup
import pandas as pd

response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text

soup = BeautifulSoup(response, "html.parser")

tables = soup.find_all('table')

company_symbols = []

for row in tables[0].tbody.find_all("tr"):
    col = row.find_all("td")
    if (col != []):
        symbol = col[0].text.strip().replace('\n','')
        company_symbols.append(symbol)

start_date = "2000-01-01"

date_temp = date.today()
date_of_today = date_temp.strftime("%Y-%m-%d")

date_of_year_later = date_temp + timedelta(days=365)
date_of_year_later = date_of_year_later.strftime("%Y-%m-%d")

date_of_month_later = date_temp + timedelta(days=30)
date_of_month_later = date_of_month_later.strftime("%Y-%m-%d")

date_of_week_later = date_temp + timedelta(days=7)
date_of_week_later = date_of_week_later.strftime("%Y-%m-%d")

date_of_day_later = date_temp + timedelta(days=1)
date_of_day_later = date_of_day_later.strftime("%Y-%m-%d")


st.title("The Stocks Prediction App")
stocks = tuple(company_symbols)

selected_stock = st.selectbox("Select Stock For Prediction", stocks)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start_date, date_of_today)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

df_training = data[["Date", "Close"]]
df_training = df_training.rename(columns={"Date":"ds", "Close":"y"})

m = Prophet()
m.fit(df_training)

future = m.make_future_dataframe(periods=375)
forecast = m.predict(future)

forecast['ds'] = pd.to_datetime(forecast['ds'])

filtered_row = forecast[forecast['ds'] == date_of_year_later]

stock_price_of_year_later = filtered_row['yhat'].values[0]

filtered_row = forecast[forecast['ds'] == date_of_month_later]

stock_price_of_month_later = filtered_row['yhat'].values[0]

filtered_row = forecast[forecast['ds'] == date_of_week_later]

stock_price_of_week_later = filtered_row['yhat'].values[0]

filtered_row = forecast[forecast['ds'] == date_of_day_later]

stock_price_of_day_later = filtered_row['yhat'].values[0]

st.write(f"Stock Forecast Till The Next Year: {stock_price_of_year_later}")
st.write(f"Stock Forecast Till The Next Month: {stock_price_of_month_later}")
st.write(f"Stock Forecast Till The Next Week: {stock_price_of_week_later}")
st.write(f"Stock Forecast Till The Next Day: {stock_price_of_day_later}")

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
