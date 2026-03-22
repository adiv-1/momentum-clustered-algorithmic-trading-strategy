import numpy as np
import pandas as pd


def moving_average_strategy(prices, short_window=20, long_window=50):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices['price']
    signals['short_ma'] = prices['price'].rolling(window=short_window, min_periods=1).mean()
    signals['long_ma'] = prices['price'].rolling(window=long_window, min_periods=1).mean()

    signals['signal'] = 0.0
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0
    )

    signals['positions'] = signals['signal'].diff()
    signals['returns'] = np.log(signals['price'] / signals['price'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
    return signals
