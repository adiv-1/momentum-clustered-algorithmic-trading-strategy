import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


# ---------------------------------------------------------------------------
# Indicator implementations (no pandas_ta dependency)
# ---------------------------------------------------------------------------

def _rsi(close, length=20):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bbands(close, length=20):
    mid = close.rolling(window=length).mean()
    std = close.rolling(window=length).std()
    low = mid - 2 * std
    high = mid + 2 * std
    return low, mid, high


def _atr(high, low, close, length=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, min_periods=length).mean()


def _macd_line(close, fast=12, slow=26):
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def add_technical_indicators(df):
    df['garman_klass_vol'] = (
        ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2
        - (2 * np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open'])) ** 2)
    )

    df['rsi'] = df.groupby(level=1)['close'].transform(lambda x: _rsi(x, length=20))

    log_close = np.log1p(df['close'])
    df['bb_low'] = df.groupby(level=1)['close'].transform(
        lambda x: _bbands(np.log1p(x), length=20)[0]
    )
    df['bb_mid'] = df.groupby(level=1)['close'].transform(
        lambda x: _bbands(np.log1p(x), length=20)[1]
    )
    df['bb_high'] = df.groupby(level=1)['close'].transform(
        lambda x: _bbands(np.log1p(x), length=20)[2]
    )

    def _compute_atr(stock_data):
        atr = _atr(stock_data['high'], stock_data['low'], stock_data['close'], length=14)
        return atr.sub(atr.mean()).div(atr.std())

    def _compute_macd(close):
        macd = _macd_line(close, fast=12, slow=26)
        return macd.sub(macd.mean()).div(macd.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(_compute_atr)
    df['macd'] = df.groupby(level=1, group_keys=False)['close'].apply(_compute_macd)
    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6

    return df


def aggregate_to_monthly(df):
    last_cols = [
        c for c in df.columns.unique(0)
        if c not in ['dollar_volume', 'volume', 'open', 'high', 'low']
    ]

    data = pd.concat([
        df.groupby([pd.Grouper(level='date', freq='ME'), 'ticker'])['dollar_volume']
          .mean()
          .to_frame('dollar_volume'),
        df[last_cols].groupby([pd.Grouper(level='date', freq='ME'), 'ticker']).last(),
    ], axis=1).dropna()

    return data


def filter_top_liquid(data, top_n=150):
    data['dollar_volume'] = data.groupby(level='ticker')['dollar_volume'].transform(
        lambda x: x.rolling(window=5 * 12, min_periods=12).mean()
    )
    data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
    data = data[data['dollar_vol_rank'] < top_n + 1].drop(
        ['dollar_volume', 'dollar_vol_rank'], axis=1
    )
    return data


def calculate_returns(df):
    lags = [1, 2, 3, 6, 9, 12]
    outlier_cutoff = 0.005

    for lag in lags:
        df[f'return_{lag}m'] = (
            df['close']
            .pct_change(lag)
            .pipe(lambda x: x.clip(
                lower=x.quantile(outlier_cutoff),
                upper=x.quantile(1 - outlier_cutoff),
            ))
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    return df


def compute_rolling_betas(data, factor_data):
    factor_data = factor_data.join(data['return_1m']).sort_index()

    observations = factor_data.groupby(level=1).size()
    valid_stocks = observations[observations >= 10]
    factor_data = factor_data[
        factor_data.index.get_level_values('ticker').isin(valid_stocks.index)
    ]

    betas = (
        factor_data.groupby(level=1, group_keys=False)
        .apply(lambda x: RollingOLS(
            endog=x['return_1m'],
            exog=sm.add_constant(x.drop('return_1m', axis=1)),
            window=min(24, x.shape[0]),
            min_nobs=len(x.columns) + 1,
        ).fit(params_only=True).params.drop('const', axis=1))
    )

    data = data.join(betas.groupby('ticker').shift())

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(
        lambda x: x.fillna(x.mean())
    )

    data = data.drop('close', axis=1).dropna()
    return data
