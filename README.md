# Momentum-Clustered Algorithmic Trading Strategy

A systematic long-only equity strategy that combines **unsupervised machine learning**, **factor models**, and **portfolio optimization** to outperform the S&P 500. The strategy selects high-momentum stocks each month using K-Means clustering and allocates capital via Efficient Frontier max-Sharpe optimization.

---

## Strategy Overview

Every month the pipeline:

1. **Screens** the S&P 500 universe to the top-150 most liquid stocks (5-year rolling average dollar volume).
2. **Engineers** eight cross-sectional features per stock: Garman-Klass volatility, RSI, Bollinger Bands (low/mid/high), normalized ATR, normalized MACD, and six momentum return windows (1, 2, 3, 6, 9, 12 months).
3. **Augments** each stock's feature vector with its rolling 24-month Fama-French 5-factor betas (Mkt-RF, SMB, HML, RMW, CMA), capturing systematic risk exposure.
4. **Clusters** stocks into four groups via K-Means with centroid initialization anchored to RSI values of 40, 55, 65, and 80. This forces a stable, interpretable cluster assignment month-over-month.
5. **Selects** only the high-momentum cluster (RSI ≈ 80), then rebalances into that basket using PyPortfolioOpt's Efficient Frontier max-Sharpe optimizer (single-stock cap of 10%; equal weights as a fallback).
6. **Tracks** daily P&L using log returns weighted by the optimized allocation.

The hypothesis: stocks that cluster near RSI 80 exhibit persistent short-term price momentum, and capitalizing on that momentum with disciplined position sizing produces risk-adjusted returns above a passive S&P 500 index.

---

## Project Structure

```
momentum-clustered-algorithmic-trading-strategy/
│
├── main.py                        # Entry point — runs the full pipeline
│
├── src/
│   ├── data/
│   │   └── fetcher.py             # S&P 500 symbols, yfinance downloads, Fama-French factors
│   │
│   ├── features/
│   │   └── indicators.py          # Technical indicators, monthly aggregation, rolling betas
│   │
│   ├── models/
│   │   ├── clustering.py          # K-Means with RSI-anchored centroids
│   │   ├── portfolio.py           # Efficient Frontier max-Sharpe optimization
│   │   └── lstm_model.py          # PyTorch LSTM for directional price prediction
│   │
│   ├── strategies/
│   │   ├── sma_strategy.py        # 20/50-day SMA crossover benchmark
│   │   └── exp_smoothing_strategy.py  # 20-day EMA benchmark
│   │
│   ├── backtest/
│   │   └── engine.py              # Monthly rebalancing backtest loop
│   │
│   └── visualization/
│       └── plots.py               # Cumulative return charts, cluster scatter plots, metrics
│
├── requirements.txt
└── algo_trading_learning_strategy.ipynb   # Original research notebook
```

---

## Key Technical Concepts

### Garman-Klass Volatility
Estimates intraday volatility more efficiently than close-to-close standard deviation by incorporating open, high, low, and close prices:

$$\sigma^2_{GK} = \frac{1}{2}\left(\ln\frac{H}{L}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C}{O}\right)^2$$

### RSI-Anchored K-Means Centroids
Rather than random initialization, centroids are pre-seeded so that cluster assignment is consistent across months. The four centroid RSI values (40, 55, 65, 80) correspond to oversold, neutral, moderately overbought, and strongly overbought regimes — making cluster 3 a reliable momentum signal.

### Rolling Fama-French Betas
For each stock a 24-month rolling OLS regression estimates its sensitivity to the five Fama-French factors. These betas are shifted forward one month before joining the feature matrix to prevent look-ahead bias.

### Efficient Frontier Optimization
Given the selected cluster's stocks and the prior 12 months of prices, the optimizer maximizes the Sharpe ratio subject to:
- Individual stock weight ∈ [1/(2N), 10%]  
- Weights sum to 1 (fully invested)

If optimization fails (e.g., insufficient price history), the portfolio falls back to equal weighting.

### LSTM Price Forecasting
A two-layer LSTM is retrained every 21 trading days (~monthly) on a rolling 252-day window. On each day, the model predicts the next day's close; a long-only signal is generated when the prediction exceeds the current price.

---

## Results

Running `main.py` produces three output images:

| File | Description |
|------|-------------|
| `cluster_visualization.png` | Scatter plots of RSI vs ATR for each K-Means cluster across five sample months |
| `unsupervised_strategy.png` | Cumulative return of the unsupervised strategy vs SPY Buy & Hold |
| `strategy_comparison.png` | Side-by-side comparison of all five strategies over the common date range |

And a CSV:

| File | Description |
|------|-------------|
| `performance_metrics.csv` | Annualized return, volatility, Sharpe ratio, max drawdown, win rate, Calmar ratio |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation varies by platform. Visit [pytorch.org](https://pytorch.org) if you need GPU support.

---

## Usage

```bash
python main.py
```

The full run (20-year S&P 500 history + LSTM training) takes approximately **1–2 hours** depending on hardware. The LSTM retrains monthly over a multi-year backtest window and is the dominant cost.

---

## Comparison Strategies

| Strategy | Description |
|----------|-------------|
| **Unsupervised Learning** | This project's K-Means momentum strategy |
| **SPY Buy & Hold** | Passive benchmark — fully invested in SPY |
| **SMA Crossover** | Go long when 20-day SMA > 50-day SMA on SPY |
| **Exp Smoothing (EMA)** | Go long when SPY price > 20-day EMA |
| **LSTM** | Go long when LSTM predicts next-day price rise on SPY |

---

## License

MIT

Python-based algorithmic trading project that uses technical indicators, K-Means clustering, and Efficient Frontier portfolio optimization to build and backtest monthly rebalanced stock strategies against SPY
