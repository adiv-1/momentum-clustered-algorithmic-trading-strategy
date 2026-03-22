# Momentum-Clustered Algorithmic Trading Strategy

I built this project to learn how machine learning can be applied to stock trading. The basic idea is to pick high-momentum stocks each month using K-Means clustering, then figure out how much to invest in each one using portfolio optimization. I wanted to see if this could beat just holding the S&P 500.

---

## What it does

Each month the strategy runs through a few steps:

1. Pulls all S&P 500 stocks and filters down to the top 150 most liquid ones (based on average dollar volume over 5 years).
2. Calculates a bunch of technical indicators for each stock — RSI, Bollinger Bands, ATR, MACD, Garman-Klass volatility, and momentum returns over 1, 2, 3, 6, 9, and 12 months.
3. Adds Fama-French factor betas (5 factors, 24-month rolling window) to capture each stock's systematic risk exposure.
4. Runs K-Means clustering to group stocks into 4 clusters. The clusters are seeded at RSI values of 40, 55, 65, and 80 so we get consistant results month to month.
5. Picks stocks in the highest RSI cluster (cluster 3, RSI ≈ 80) — the idea being these stocks have the most upward momentum.
6. Optimizes portfolio weights using Efficient Frontier to maximize Sharpe ratio. Each stock is capped at 10% of the portfolio.
7. Calculates daily returns based on those weights.

The hypothesis is pretty simple: stocks with high RSI tend to keep going up in the short term. So we ride that momentum every month and rebalance when it changes.

---

## Project Structure

```
momentum-clustered-algorithmic-trading-strategy/
│
├── main.py                        # runs everything
│
├── src/
│   ├── data/
│   │   └── fetcher.py             # downloads S&P 500 data, SPY, and Fama-French factors
│   │
│   ├── features/
│   │   └── indicators.py          # computes all the technical indicators and rolling betas
│   │
│   ├── models/
│   │   ├── clustering.py          # K-Means clustering with RSI-anchored centroids
│   │   ├── portfolio.py           # Efficient Frontier optimization
│   │   └── lstm_model.py          # PyTorch LSTM for price direction prediction
│   │
│   ├── strategies/
│   │   ├── sma_strategy.py        # 20/50-day moving average crossover
│   │   └── exp_smoothing_strategy.py  # 20-day EMA strategy
│   │
│   ├── backtest/
│   │   └── engine.py              # the main backtest loop
│   │
│   └── visualization/
│       └── plots.py               # all the charts and metrics
│
├── requirements.txt
└── algo_trading_learning_strategy.ipynb   # original research notebook
```

---

## The Technical Stuff

### Garman-Klass Volatility
This is a better way to estimate volatility than just looking at daily close prices. It uses the open, high, low, and close to get a more accurate picture of how much a stock moved intraday.

$$\sigma^2_{GK} = \frac{1}{2}\left(\ln\frac{H}{L}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C}{O}\right)^2$$

### RSI-Anchored K-Means
The problem with regular K-Means is it uses random initialization, so the cluster labels change every month. I fixed this by seeding the centroids at RSI values of 40, 55, 65, and 80. That way cluster 3 always means "high momentum" and the strategy is consistent.

### Fama-French Rolling Betas
For each stock I run a rolling 24-month OLS regression against the 5 Fama-French factors (market risk, size, value, profitability, investment). These betas get shifted forward one month so we're not cheating by using future data.

### Portfolio Optimization
Once we have the cluster 3 stocks, we use the last 12 months of price data to find the weights that maximize the Sharpe ratio. Each stock is bounded between `1/(2N)` and `10%`. If the optimizer fails we just go equal weight.

### LSTM
There's also an LSTM model that retrains every 21 trading days on a rolling 1-year window. It predicts the next day's price direction for SPY and goes long when it expects an increase. Its mostly there as a comparison benchmark.

---

## Results

Running `main.py` saves these files to the project root:

| File | What it shows |
|------|---------------|
| `cluster_visualization.png` | RSI vs ATR scatter plots for each cluster across 5 months |
| `unsupervised_strategy.png` | Our strategy vs SPY Buy & Hold |
| `strategy_comparison.png` | All 5 strategies compared side by side |
| `performance_metrics.csv` | Sharpe ratio, annualized return, max drawdown, win rate, etc. |

---

## Installation

```bash
pip install -r requirements.txt
```

> PyTorch install can vary depending on your machine. Check [pytorch.org](https://pytorch.org) if you run into issues.

---

## How to run it

```bash
python main.py
```

Fair warning — the full run takes around 1-2 hours. Most of that is the LSTM retraining every month over 20 years of data. If you just want to test the pipeline quickly, run `smoke_test.py` instead — that uses 30 stocks and 6 years.

---

## What we're comparing against

| Strategy | Description |
|----------|-------------|
| **Unsupervised Learning** | our K-Means momentum strategy |
| **SPY Buy & Hold** | just hold SPY, the passive benchmark |
| **SMA Crossover** | buy when the 20-day MA crosses above the 50-day MA |
| **Exp Smoothing** | buy when SPY price is above its 20-day EMA |
| **LSTM** | buy when the LSTM predicts the next day will be higher |

---

## License

MIT
