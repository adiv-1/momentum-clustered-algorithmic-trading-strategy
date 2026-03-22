import datetime as dt
import urllib.request

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    sp500 = pd.read_html(urllib.request.urlopen(req))[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    return sp500['Symbol'].unique().tolist()


def download_price_data(symbols, years=20):
    end_date = dt.datetime.now()
    start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * years)
    df = yf.download(tickers=symbols, start=start_date, end=end_date)
    return df


def reshape_price_data(df):
    data_list = []
    for ticker in df.columns.get_level_values(1).unique():
        ticker_df = df.xs(ticker, level=1, axis=1).copy()
        ticker_df['Ticker'] = ticker
        data_list.append(ticker_df)

    df = pd.concat(data_list).reset_index()
    df = df.set_index(['Date', 'Ticker'])

    if 'Adj Close' in df.columns:
        df = df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
    else:
        df = df[['Close', 'High', 'Low', 'Open', 'Volume']]

    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    return df


def download_spy_data(start='2005-01-01'):
    end = dt.date.today().strftime('%Y-%m-%d')
    spy = yf.download(tickers='SPY', start=start, end=end)
    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    return spy


def get_fama_french_factors(start='2010'):
    factor_data = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start
    )[0].drop('RF', axis=1)
    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample('ME').last().div(100)
    factor_data.index.name = 'date'
    return factor_data
