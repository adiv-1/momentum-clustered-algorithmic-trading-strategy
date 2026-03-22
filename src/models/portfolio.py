from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)
    ef = EfficientFrontier(
        expected_returns=returns,
        cov_matrix=cov,
        weight_bounds=(lower_bound, 0.1),
        solver='SCS',
    )
    ef.max_sharpe()
    return ef.clean_weights()


def equal_weights(cols):
    n = len(cols)
    return {col: 1 / n for col in cols}
