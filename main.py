"""
Momentum-Clustered Algorithmic Trading Strategy
================================================
Orchestrates the full pipeline:
  1. Fetch S&P 500 universe and 20-year price history
  2. Compute technical indicators (Garman-Klass, RSI, Bollinger Bands, ATR, MACD)
  3. Aggregate to monthly frequency; filter top-150 liquid stocks
  4. Compute 1–12 month momentum returns + Fama-French rolling factor betas
  5. K-Means clustering with RSI-anchored centroids (target RSI 40/55/65/80)
  6. Select high-momentum cluster → Efficient Frontier portfolio each month
  7. Backtest and compare vs SPY Buy & Hold, SMA, EMA, and LSTM strategies
  8. Output images and a performance metrics table
"""

import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

from src.backtest.engine import run_backtest
from src.data.fetcher import (
    download_price_data,
    download_spy_data,
    get_fama_french_factors,
    get_sp500_symbols,
    reshape_price_data,
)
from src.features.indicators import (
    add_technical_indicators,
    aggregate_to_monthly,
    calculate_returns,
    compute_rolling_betas,
    filter_top_liquid,
)
from src.models.clustering import apply_kmeans_clustering, select_momentum_cluster
from src.models.lstm_model import lstm_strategy
from src.strategies.exp_smoothing_strategy import exponential_smoothing_strategy
from src.strategies.sma_strategy import moving_average_strategy
from src.visualization.plots import (
    compute_metrics,
    plot_cluster_visualization,
    plot_strategy_comparison,
    plot_unsupervised_returns,
)


def main():
    os.makedirs('final_output', exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Universe and raw price data                                       #
    # ------------------------------------------------------------------ #
    print('[1/9] Fetching S&P 500 symbols...')
    symbols_list = get_sp500_symbols()

    print('[2/9] Downloading 20-year daily price history...')
    raw_df = download_price_data(symbols_list, years=20)

    print('[3/9] Reshaping to long format...')
    df = reshape_price_data(raw_df)

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    print('[4/9] Computing technical indicators...')
    df = add_technical_indicators(df)

    print('[5/9] Aggregating to monthly, filtering top-150 by liquidity...')
    data = aggregate_to_monthly(df)
    data = filter_top_liquid(data, top_n=150)
    data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    # ------------------------------------------------------------------ #
    # 3. Fama-French rolling factor betas                                  #
    # ------------------------------------------------------------------ #
    print('[6/9] Computing rolling Fama-French factor betas...')
    factor_data = get_fama_french_factors(start='2010')
    data = compute_rolling_betas(data, factor_data)

    # ------------------------------------------------------------------ #
    # 4. K-Means clustering → momentum cluster selection                   #
    # ------------------------------------------------------------------ #
    print('[7/9] Applying K-Means clustering with RSI-anchored centroids...')
    data = apply_kmeans_clustering(data)

    print('       Saving cluster visualization...')
    plot_cluster_visualization(data, output_path='final_output/cluster_visualization.png', plot_limit=5)

    print('       Selecting high-momentum cluster (RSI ~ 80)...')
    fixed_dates = select_momentum_cluster(data, cluster_id=3)

    # ------------------------------------------------------------------ #
    # 5. Portfolio construction and backtest                               #
    # ------------------------------------------------------------------ #
    print('[8/9] Downloading prices for portfolio optimization...')
    stocks = data.index.get_level_values('ticker').unique().tolist()
    new_df = yf.download(
        tickers=stocks,
        start=data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12),
        end=data.index.get_level_values('date').unique()[-1],
    )

    returns_dataframe = np.log(new_df['Close']).diff()

    print('       Running monthly rebalancing backtest...')
    portfolio_df = run_backtest(fixed_dates, returns_dataframe, new_df)

    # ------------------------------------------------------------------ #
    # 6. Benchmark and comparison strategies                               #
    # ------------------------------------------------------------------ #
    print('[9/9] Running comparison strategies...')
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

    sma_signals = moving_average_strategy(spy_prices, short_window=20, long_window=50)
    exp_signals = exponential_smoothing_strategy(spy_prices, span=20)

    print('       Training LSTM (this may take 30-60 minutes)...')
    lstm_signals = lstm_strategy(spy_prices, seq_length=20, hidden_size=50, retrain_freq=21)

    # ------------------------------------------------------------------ #
    # 7. Build cumulative return table                                     #
    # ------------------------------------------------------------------ #
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
    cumulative_returns['LSTM'] = (
        np.exp(lstm_signals.loc[common_index, 'strategy_returns'].fillna(0).cumsum()) - 1
    )

    # ------------------------------------------------------------------ #
    # 8. Save output images                                                #
    # ------------------------------------------------------------------ #
    plot_unsupervised_returns(portfolio_df2, output_path='final_output/unsupervised_strategy.png')
    plot_strategy_comparison(cumulative_returns, output_path='final_output/strategy_comparison.png')

    # ------------------------------------------------------------------ #
    # 9. Performance metrics                                               #
    # ------------------------------------------------------------------ #
    return_series_map = {
        'Unsupervised Learning': portfolio_df2.loc[common_index, 'Strategy Return'],
        'SPY Buy & Hold': portfolio_df2.loc[common_index, 'SPY Buy & Hold'],
        'SMA Crossover': sma_signals.loc[common_index, 'strategy_returns'],
        'Exp Smoothing': exp_signals.loc[common_index, 'strategy_returns'],
        'LSTM': lstm_signals.loc[common_index, 'strategy_returns'],
    }

    metrics = compute_metrics(cumulative_returns, return_series_map)
    metrics.to_csv('final_output/performance_metrics.csv')

    print('\n=== Performance Metrics ===')
    print(metrics.to_string())
    print('\nOutput files saved to final_output/:')
    print('  final_output/cluster_visualization.png')
    print('  final_output/unsupervised_strategy.png')
    print('  final_output/strategy_comparison.png')
    print('  final_output/performance_metrics.csv')


if __name__ == '__main__':
    main()
