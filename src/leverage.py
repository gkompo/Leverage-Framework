"""Dynamic leverage strategy implementations.

Implements five leverage strategies:
1. Buy and Hold (1x)
2. Fixed Leverage (configurable)
3. Volatility Targeting
4. Regime-Based Leverage
5. Recovery Acceleration Model
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('regime_leverage')

# Default regime leverage mappings
DEFAULT_REGIME_LEVERAGE = {
    'Low Volatility': 1.75,
    'Normal Market': 1.0,
    'High Volatility': 0.5,
    'Crisis': 0.125,
    'Recovery': 1.375
}


def buy_and_hold_leverage(returns):
    """Strategy 1: Buy and hold (constant 1x leverage).
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    
    Returns
    -------
    pd.Series
        Leverage series (all 1.0).
    """
    return pd.Series(1.0, index=returns.index, name='leverage_bh')


def fixed_leverage(returns, leverage_ratio=2.0):
    """Strategy 2: Fixed constant leverage.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    leverage_ratio : float
        Fixed leverage multiplier.
    
    Returns
    -------
    pd.Series
        Leverage series.
    """
    return pd.Series(leverage_ratio, index=returns.index, 
                     name=f'leverage_fixed_{leverage_ratio}x')


def volatility_targeting_leverage(returns, vol_series, target_vol=0.15, 
                                   max_leverage=3.0, min_leverage=0.1):
    """Strategy 3: Volatility targeting.
    
    L_t = target_vol / current_vol
    
    Uses lagged volatility to avoid look-ahead bias.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    vol_series : pd.Series
        Annualized rolling volatility (must be lagged by caller or uses shift).
    target_vol : float
        Target annualized volatility (default 15%).
    max_leverage : float
        Maximum leverage cap.
    min_leverage : float
        Minimum leverage floor.
    
    Returns
    -------
    pd.Series
        Leverage series.
    """
    # Use lagged volatility (available at time t-1)
    lagged_vol = vol_series.shift(1)
    leverage = target_vol / lagged_vol
    leverage = leverage.clip(lower=min_leverage, upper=max_leverage)
    leverage = leverage.fillna(1.0)
    leverage.name = 'leverage_voltarget'
    return leverage


def regime_leverage(returns, regimes, regime_map=None):
    """Strategy 4: Regime-based leverage.
    
    Uses lagged regime assignment to avoid look-ahead bias.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns.
    regimes : pd.Series
        Regime labels.
    regime_map : dict, optional
        Mapping from regime name to leverage. Uses defaults if None.
    
    Returns
    -------
    pd.Series
        Leverage series.
    """
    if regime_map is None:
        regime_map = DEFAULT_REGIME_LEVERAGE
    
    # Use lagged regime (available at time t-1)
    lagged_regimes = regimes.shift(1)
    leverage = lagged_regimes.map(regime_map)
    leverage = leverage.reindex(returns.index)
    leverage = leverage.fillna(1.0)
    leverage.name = 'leverage_regime'
    return leverage


def recovery_acceleration_leverage(returns, prices, vol_series, regimes,
                                     base_leverage=1.0, max_boost=0.75):
    """Strategy 5: Recovery acceleration model.
    
    Computes a recovery score based on:
    - Volatility decline (is vol decreasing?)
    - Positive momentum (recent returns)
    - Distance from previous peak (recovery potential)
    
    Increases leverage gradually during recovery periods.
    Uses only lagged information to avoid look-ahead bias.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    prices : pd.Series
        Price series.
    vol_series : pd.Series
        Annualized rolling volatility.
    regimes : pd.Series
        Regime labels.
    base_leverage : float
        Base leverage in non-recovery periods.
    max_boost : float
        Maximum additional leverage during recovery.
    
    Returns
    -------
    pd.Series
        Leverage series.
    pd.Series
        Recovery score series.
    """
    # All computations use lagged values
    lagged_vol = vol_series.shift(1)
    lagged_regimes = regimes.shift(1)
    
    # Component 1: Volatility decline score
    vol_change_3m = (lagged_vol - lagged_vol.shift(3)) / lagged_vol.shift(3)
    vol_decline_score = (-vol_change_3m).clip(lower=0, upper=1)
    vol_decline_score = vol_decline_score.fillna(0)
    
    # Component 2: Positive momentum score (3-month return)
    momentum = returns.rolling(3).sum().shift(1)
    # Avoid zero division
    rolling_momentum_vol = momentum.abs().rolling(60).mean().shift(1)
    momentum_score = (momentum / rolling_momentum_vol.replace(0, 1e-6)).clip(-1, 1)
    momentum_score = (momentum_score + 1) / 2  # Scale to [0, 1]
    momentum_score = momentum_score.fillna(0.5)
    
    # Component 3: Distance from peak (recovery potential)
    aligned_prices = prices.reindex(returns.index)
    rolling_peak = aligned_prices.expanding().max().shift(1)
    distance_from_peak = 1 - (aligned_prices.shift(1) / rolling_peak)
    distance_from_peak = distance_from_peak.clip(lower=0, upper=1)
    distance_from_peak = distance_from_peak.fillna(0)
    
    # Recovery score: weighted combination
    recovery_score = (
        0.35 * vol_decline_score + 
        0.35 * momentum_score + 
        0.30 * distance_from_peak
    )
    
    # Apply regime leverage first, then boost during recovery
    leverage = lagged_regimes.map(DEFAULT_REGIME_LEVERAGE).fillna(base_leverage)
    
    # Apply recovery boost when in Recovery regime
    is_recovery = (lagged_regimes == 'Recovery').astype(float)
    recovery_boost = is_recovery * recovery_score * max_boost
    
    leverage = leverage + recovery_boost
    
    # Reindex and fillna
    leverage = leverage.reindex(returns.index)
    leverage = leverage.fillna(base_leverage)
    leverage = leverage.clip(lower=0.0, upper=3.0)
    leverage.name = 'leverage_recovery'
    
    recovery_score = recovery_score.reindex(returns.index).fillna(0.0)
    recovery_score.name = 'recovery_score'
    
    return leverage, recovery_score


def generate_all_strategies(returns, prices, vol_series, regimes):
    """Generate leverage series for all strategies.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    prices : pd.Series
        Price series.
    vol_series : pd.Series
        Rolling annualized volatility.
    regimes : pd.Series
        Regime labels.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with leverage for each strategy.
    """
    strategies = pd.DataFrame(index=returns.index)
    
    strategies['Buy & Hold'] = buy_and_hold_leverage(returns)
    strategies['Fixed 1x'] = fixed_leverage(returns, 1.0)
    strategies['Fixed 2x'] = fixed_leverage(returns, 2.0)
    strategies['Vol Target'] = volatility_targeting_leverage(
        returns, vol_series, target_vol=0.15
    )
    strategies['Regime'] = regime_leverage(returns, regimes)
    rec_lev, rec_score = recovery_acceleration_leverage(
        returns, prices, vol_series, regimes
    )
    strategies['Recovery Accel'] = rec_lev
    
    logger.info(f'Generated {len(strategies.columns)} leverage strategies')
    return strategies
