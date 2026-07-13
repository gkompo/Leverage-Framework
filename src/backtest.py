"""Backtesting engine for leveraged strategies.

Implements portfolio simulation with:
- Monthly rebalancing
- Transaction costs
- No look-ahead bias

Portfolio evolution:
    V_t = V_(t-1) * (1 + L_t * r_t) - TC
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('regime_leverage')


DEFAULT_TRANSACTION_COST = 0.0005  # 0.05%


def simulate_portfolio(returns, leverage, initial_value=1.0,
                       transaction_cost=DEFAULT_TRANSACTION_COST):
    """Simulate leveraged portfolio.
    
    V_t = V_(t-1) * (1 + L_t * r_t) - |delta_L| * V_(t-1) * TC
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    leverage : pd.Series
        Leverage series (must be aligned with returns).
    initial_value : float
        Starting portfolio value.
    transaction_cost : float
        Proportional transaction cost per unit leverage change.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: portfolio_value, portfolio_return,
        leverage, turnover, costs
    """
    # Align series
    common_idx = returns.index.intersection(leverage.index)
    returns = returns.loc[common_idx]
    leverage = leverage.loc[common_idx]
    
    # Convert log returns to simple returns for portfolio simulation
    simple_returns = np.exp(returns) - 1
    
    n = len(returns)
    portfolio_values = np.zeros(n)
    portfolio_returns = np.zeros(n)
    turnover = np.zeros(n)
    costs = np.zeros(n)
    
    if n == 0:
        return pd.DataFrame()
        
    portfolio_values[0] = initial_value * (1 + leverage.iloc[0] * simple_returns.iloc[0])
    portfolio_returns[0] = leverage.iloc[0] * simple_returns.iloc[0]
    
    for t in range(1, n):
        # Leverage change (turnover)
        lev_change = abs(leverage.iloc[t] - leverage.iloc[t-1])
        turnover[t] = lev_change
        
        # Transaction cost
        tc = lev_change * portfolio_values[t-1] * transaction_cost
        costs[t] = tc
        
        # Portfolio return for this period
        port_ret = leverage.iloc[t] * simple_returns.iloc[t]
        portfolio_returns[t] = port_ret
        
        # Update portfolio value
        portfolio_values[t] = portfolio_values[t-1] * (1 + port_ret) - tc
        
        # Floor at zero (bankruptcy)
        if portfolio_values[t] <= 0:
            portfolio_values[t] = 0
            portfolio_values[t+1:] = 0
            break
    
    result = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'portfolio_return': portfolio_returns,
        'leverage': leverage.values,
        'turnover': turnover,
        'costs': costs
    }, index=common_idx)
    
    return result


def run_backtest(returns, leverage_df, initial_value=1.0,
                 transaction_cost=DEFAULT_TRANSACTION_COST):
    """Run backtests for all strategies.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    leverage_df : pd.DataFrame
        DataFrame with leverage columns for each strategy.
    initial_value : float
        Starting portfolio value.
    transaction_cost : float
        Transaction cost.
    
    Returns
    -------
    dict
        Dictionary mapping strategy names to backtest DataFrames.
    """
    results = {}
    
    for strategy in leverage_df.columns:
        logger.info(f'Running backtest: {strategy}')
        result = simulate_portfolio(
            returns, leverage_df[strategy], 
            initial_value, transaction_cost
        )
        results[strategy] = result
    
    logger.info(f'Completed {len(results)} backtests')
    return results


def extract_portfolio_values(backtest_results):
    """Extract portfolio value series from backtest results.
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results dictionary.
    
    Returns
    -------
    pd.DataFrame
        Portfolio values for all strategies.
    """
    values = {}
    for name, df in backtest_results.items():
        if not df.empty:
            values[name] = df['portfolio_value']
    return pd.DataFrame(values)


def extract_portfolio_returns(backtest_results):
    """Extract portfolio return series from backtest results.
    
    Parameters
    ----------
    backtest_results : dict
        Backtest results dictionary.
    
    Returns
    -------
    pd.DataFrame
        Portfolio returns for all strategies.
    """
    rets = {}
    for name, df in backtest_results.items():
        if not df.empty:
            rets[name] = df['portfolio_return']
    return pd.DataFrame(rets)
