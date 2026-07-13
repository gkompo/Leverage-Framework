"""Returns calculation module.

Computes monthly log returns and cumulative returns from price data.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('regime_leverage')


def compute_log_returns(prices):
    """Compute monthly log returns from a price series.
    
    r_t = log(P_t / P_(t-1))
    
    Parameters
    ----------
    prices : pd.Series
        Monthly price series.
    
    Returns
    -------
    pd.Series
        Monthly log returns.
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna()


def compute_simple_returns(prices):
    """Compute simple monthly returns from a price series."""
    return prices.pct_change().dropna()


def compute_cumulative_returns(returns, log_returns=True):
    """Compute cumulative returns from a return series.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    log_returns : bool
        If True, treats input as log returns.
    
    Returns
    -------
    pd.Series
        Cumulative return series (starting at 1.0).
    """
    if log_returns:
        cum_ret = np.exp(returns.cumsum())
    else:
        cum_ret = (1 + returns).cumprod()
    return cum_ret


def compute_excess_returns(returns, risk_free_rate=0.0):
    """Compute excess returns over risk-free rate.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    risk_free_rate : float
        Annual risk-free rate (converted to monthly internally).
    
    Returns
    -------
    pd.Series
        Excess returns.
    """
    monthly_rf = (1 + risk_free_rate) ** (1/12) - 1
    return returns - monthly_rf


def compute_wealth_index(returns, initial_wealth=1.0, log_returns=True):
    """Compute wealth index from returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    initial_wealth : float
        Starting portfolio value.
    log_returns : bool
        If True, treats input as log returns.
    
    Returns
    -------
    pd.Series
        Wealth index series.
    """
    cum = compute_cumulative_returns(returns, log_returns)
    return initial_wealth * cum
