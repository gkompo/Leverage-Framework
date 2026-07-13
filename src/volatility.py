"""Volatility estimation module.

Computes rolling volatility measures, volatility percentiles,
drawdowns, and rolling maximum drawdowns.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('regime_leverage')


def rolling_volatility(returns, window=12):
    """Calculate rolling annualized volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    window : int
        Rolling window in months (default 12).
    
    Returns
    -------
    pd.Series
        Annualized rolling volatility.
    """
    return returns.rolling(window=window).std() * np.sqrt(12)


def volatility_percentile(vol_series, lookback=120):
    """Calculate the percentile rank of current volatility.
    
    Parameters
    ----------
    vol_series : pd.Series
        Volatility series.
    lookback : int
        Lookback window in months for percentile calculation.
    
    Returns
    -------
    pd.Series
        Percentile rank (0-1) of volatility.
    """
    def _percentile_rank(window):
        if len(window) < 2:
            return np.nan
        current = window.iloc[-1]
        return (window < current).sum() / (len(window) - 1)
    
    return vol_series.rolling(window=lookback, min_periods=24).apply(
        _percentile_rank, raw=False
    )


def compute_drawdown(prices):
    """Compute drawdown series from price or wealth series.
    
    Parameters
    ----------
    prices : pd.Series
        Price or portfolio value series.
    
    Returns
    -------
    pd.Series
        Drawdown series (negative values).
    """
    rolling_max = prices.expanding().max()
    drawdown = prices / rolling_max - 1
    return drawdown


def rolling_max_drawdown(prices, window=24):
    """Calculate rolling maximum drawdown over a window.
    
    Parameters
    ----------
    prices : pd.Series
        Price or portfolio value series.
    window : int
        Rolling window in months.
    
    Returns
    -------
    pd.Series
        Rolling maximum drawdown (negative values).
    """
    def _max_drawdown(window_prices):
        if len(window_prices) < 2:
            return 0.0
        cum_max = np.maximum.accumulate(window_prices)
        dd = window_prices / cum_max - 1
        return dd.min()
    
    return prices.rolling(window=window, min_periods=2).apply(
        _max_drawdown, raw=True
    )


def volatility_change(vol_series, periods=1):
    """Compute change in volatility.
    
    Parameters
    ----------
    vol_series : pd.Series
        Volatility series.
    periods : int
        Number of periods for differencing.
    
    Returns
    -------
    pd.Series
        Change in volatility.
    """
    return vol_series.diff(periods)


def compute_volatility_features(returns, prices):
    """Compute all volatility features for regime detection.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    prices : pd.Series
        Monthly price series.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: vol_12m, vol_24m, vol_pctile,
        vol_change, drawdown, rolling_mdd
    """
    vol_12m = rolling_volatility(returns, window=12)
    vol_24m = rolling_volatility(returns, window=24)
    vol_pctile = volatility_percentile(vol_12m)
    vol_chg = volatility_change(vol_12m, periods=3)
    dd = compute_drawdown(prices)
    r_mdd = rolling_max_drawdown(prices, window=24)
    
    features = pd.DataFrame({
        'vol_12m': vol_12m,
        'vol_24m': vol_24m,
        'vol_pctile': vol_pctile,
        'vol_change': vol_chg,
        'drawdown': dd,
        'rolling_mdd': r_mdd
    }, index=returns.index)
    
    logger.info(f'Computed volatility features: {len(features)} observations')
    return features
