"""Quick smoke test — 30 stocks, 6 years, no LSTM."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.fetcher import reshape_price_data, download_spy_data, get_fama_french_factors
from src.features.indicators import (
    add_technical_indicators, aggregate_to_monthly,
    filter_top_liquid, calculate_returns, compute_rolling_betas,
)
from src.models.clustering import apply_kmeans_clustering, select_momentum_cluster
from src.backtest.engine import run_backtest
from src.strategies.sma_strategy import moving_average_strategy
from src.strategies.exp_smoothing_strategy import exponential_smoothing_strategy
from src.visualization.plots import (
    plot_cluster_visualization, plot_strategy_comparison,
    plot_unsupervised_returns, compute_metrics,
)

TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 'PG',
    'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM', 'ABBV', 'CVX', 'LLY', 'AVGO',
    'COST', 'MCD', 'PEP', 'CSCO', 'ADBE', 'WMT', 'TMO', 'ACN', 'NEE', 'NKE',
]

print('[1] Downloading 6yr daily data...')
raw = yf.download(tickers=TICKERS, start='2018-01-01', end='2024-01-01')
df = reshape_price_data(raw)
print(f'    shape: {df.shape}')

print('[2] Computing indicators...')
df = add_technical_indicators(df)

print('[3] Monthly aggregation + liquidity filter...')
data = aggregate_to_monthly(df)
data = filter_top_liquid(data, top_n=30)
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
print(f'    shape: {data.shape}')

print('[4] Rolling Fama-French betas...')
factor_data = get_fama_french_factors(start='2018')
data = compute_rolling_betas(data, factor_data)
print(f'    shape after betas: {data.shape}')

print('[5] K-Means clustering...')
data = apply_kmeans_clustering(data)
print('    cluster distribution:')
print(data['cluster'].value_counts().to_string())
plot_cluster_visualization(data, output_path='cluster_visualization.png', plot_limit=5)

print('[6] Momentum cluster selection...')
fixed_dates = select_momentum_cluster(data, cluster_id=3)
print(f'    months in strategy: {len(fixed_dates)}')

print('[7] Portfolio optimization backtest...')
stocks = data.index.get_level_values('ticker').unique().tolist()
new_df = yf.download(
    tickers=stocks,
    start=data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12),
    end=data.index.get_level_values('date').unique()[-1],
)
returns_dataframe = np.log(new_df['Close']).diff()
portfolio_df = run_backtest(fixed_dates, returns_dataframe, new_df)
print(f'    portfolio trading days: {len(portfolio_df)}')

print('[8] Benchmark strategies...')
spy = download_spy_data(start='2005-01-01')
spy_ret = np.log(spy[['Close']]).diff().dropna()
if isinstance(spy_ret.columns, pd.MultiIndex):
    spy_ret.columns = spy_ret.columns.droplevel(1)
spy_ret.columns = ['SPY Buy & Hold']

portfolio_df2 = portfolio_df.merge(spy_ret, left_index=True, right_index=True)

spy_prices = spy[['Close']].copy()
if isinstance(spy_prices.columns, pd.MultiIndex):
    spy_prices.columns = spy_prices.columns.droplevel(1)
spy_prices.columns = ['price']

sma_signals = moving_average_strategy(spy_prices)
exp_signals = exponential_smoothing_strategy(spy_prices)

common_index = portfolio_df2.index.intersection(sma_signals.index)
cumulative_returns = pd.DataFrame(index=common_index)
cumulative_returns['Unsupervised Learning'] = (
    np.exp(np.log1p(portfolio_df2.loc[common_index, 'Strategy Return']).cumsum()) - 1
)
cumulative_returns['SPY Buy & Hold'] = (
    np.exp(np.log1p(portfolio_df2.loc[common_index, 'SPY Buy & Hold']).cumsum()) - 1
)
cumulative_returns['SMA Crossover'] = (
    np.exp(sma_signals.loc[common_index, 'strategy_returns'].fillna(0).cumsum()) - 1
)
cumulative_returns['Exp Smoothing'] = (
    np.exp(exp_signals.loc[common_index, 'strategy_returns'].fillna(0).cumsum()) - 1
)

print('[9] Saving plots...')
plot_unsupervised_returns(portfolio_df2, output_path='unsupervised_strategy.png')
plot_strategy_comparison(cumulative_returns, output_path='strategy_comparison.png')

return_series_map = {
    'Unsupervised Learning': portfolio_df2.loc[common_index, 'Strategy Return'],
    'SPY Buy & Hold': portfolio_df2.loc[common_index, 'SPY Buy & Hold'],
    'SMA Crossover': sma_signals.loc[common_index, 'strategy_returns'],
    'Exp Smoothing': exp_signals.loc[common_index, 'strategy_returns'],
}
metrics = compute_metrics(cumulative_returns, return_series_map)
metrics.to_csv('performance_metrics.csv')

print('\n=== Performance Metrics ===')
print(metrics.to_string())
print('\nSMOKE TEST PASSED')
