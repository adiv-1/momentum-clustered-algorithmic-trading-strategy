import numpy as np
import pandas as pd

from src.models.portfolio import optimize_weights


def run_backtest(fixed_dates, returns_dataframe, new_df):
    portfolio_df = pd.DataFrame()
    close_df = new_df['Close']

    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.droplevel(0)

    for start_date in fixed_dates.keys():
        try:
            end_date = (
                pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
            ).strftime('%Y-%m-%d')
            cols = fixed_dates[start_date]

            opt_start = (
                pd.to_datetime(start_date) - pd.DateOffset(months=12)
            ).strftime('%Y-%m-%d')
            opt_end = (
                pd.to_datetime(start_date) - pd.DateOffset(days=1)
            ).strftime('%Y-%m-%d')

            optimization_df = close_df.loc[opt_start:opt_end, cols]

            try:
                weights = optimize_weights(
                    prices=optimization_df,
                    lower_bound=round(1 / (len(optimization_df.columns) * 2), 3),
                )
                weights = pd.DataFrame(weights, index=pd.Series(0))
            except Exception:
                print(f'Max Sharpe failed for {start_date}, using equal weights')
                weights = pd.DataFrame(
                    [1 / len(optimization_df.columns)] * len(optimization_df.columns),
                    index=optimization_df.columns.tolist(),
                    columns=pd.Series(0),
                ).T

            temp_df = returns_dataframe[start_date:end_date]
            if temp_df.empty:
                continue
            temp_df = (
                temp_df.stack()
                .to_frame('return')
                .reset_index(level=0)
                .merge(
                    weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                    left_index=True,
                    right_index=True,
                )
                .reset_index()
                .set_index(['Date', 'Ticker'])
                .unstack()
                .stack()
            )
            temp_df.index.names = ['date', 'ticker']
            temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']
            temp_df = (
                temp_df.groupby(level=0)['weighted_return']
                .sum()
                .to_frame('Strategy Return')
            )
            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

        except Exception as e:
            print(e)

    return portfolio_df
