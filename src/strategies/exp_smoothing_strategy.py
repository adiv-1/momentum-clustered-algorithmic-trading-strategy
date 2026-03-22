import numpy as np
import pandas as pd


def exponential_smoothing_strategy(prices, span=20):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices['price']
    signals['ema'] = prices['price'].ewm(span=span, adjust=False).mean()

    signals['signal'] = np.where(signals['price'] > signals['ema'], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    signals['returns'] = np.log(signals['price'] / signals['price'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
    return signals
