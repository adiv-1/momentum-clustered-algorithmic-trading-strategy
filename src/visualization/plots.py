import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


def plot_cluster_visualization(data, output_path='cluster_visualization.png', plot_limit=5):
    dates = data.index.get_level_values('date').unique().tolist()[:plot_limit]
    fig, axes = plt.subplots(1, len(dates), figsize=(5 * len(dates), 4))
    if len(dates) == 1:
        axes = [axes]

    plt.style.use('ggplot')
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'black'}
    label_map = {0: 'RSI~40', 1: 'RSI~55', 2: 'RSI~65', 3: 'RSI~80'}

    for idx, d in enumerate(dates):
        ax = axes[idx]
        g = data.xs(d, level=0)
        for cid, color in color_map.items():
            cluster = g[g['cluster'] == cid]
            ax.scatter(cluster['atr'], cluster['rsi'], color=color,
                       label=label_map[cid], alpha=0.7, s=30)
        ax.set_title(d.strftime('%Y-%m'), fontsize=10)
        ax.set_xlabel('ATR')
        ax.set_ylabel('RSI')
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle('K-Means Cluster Visualization (ATR vs RSI)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_unsupervised_returns(portfolio_df2, output_path='unsupervised_strategy.png'):
    plt.style.use('ggplot')
    cumulative = np.exp(np.log1p(portfolio_df2).cumsum()) - 1
    cumulative.plot(figsize=(16, 8))
    plt.title('Unsupervised Learning Trading Strategy vs SPY Buy & Hold')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_strategy_comparison(cumulative_returns, output_path='strategy_comparison.png'):
    plt.figure(figsize=(18, 10))
    plt.style.use('ggplot')

    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)

    plt.title('Comparison of Trading Strategies: Cumulative Returns Over Time',
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def compute_metrics(cumulative_returns, return_series_map):
    index_labels = [
        'Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)',
        'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)',
        'Avg Daily Return (bps)', 'Best Month (%)', 'Worst Month (%)', 'Calmar Ratio',
    ]
    metrics = pd.DataFrame(index=index_labels)

    for col, returns in return_series_map.items():
        returns = returns.fillna(0)
        total_return = cumulative_returns[col].iloc[-1] * 100
        years = len(returns) / 252
        annualized_return = ((1 + cumulative_returns[col].iloc[-1]) ** (1 / years) - 1) * 100
        annualized_vol = returns.std() * np.sqrt(252) * 100
        sharpe = (
            (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            if returns.std() != 0 else 0
        )
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.expanding().max()
        max_drawdown = ((cum_ret - running_max) / running_max).min() * 100
        win_rate = (returns > 0).sum() / len(returns) * 100
        avg_daily_bps = returns.mean() * 10000
        monthly = returns.resample('ME').sum()
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics[col] = [
            f'{total_return:.2f}', f'{annualized_return:.2f}', f'{annualized_vol:.2f}',
            f'{sharpe:.4f}', f'{max_drawdown:.2f}', f'{win_rate:.2f}',
            f'{avg_daily_bps:.2f}', f'{monthly.max() * 100:.2f}',
            f'{monthly.min() * 100:.2f}', f'{calmar:.4f}',
        ]

    return metrics
