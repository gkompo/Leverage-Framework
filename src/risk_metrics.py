"""Risk and performance metrics.

Computes comprehensive portfolio performance metrics including:
- Annual return
- Annual volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Worst/best month
- Average leverage
- Turnover
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger('regime_leverage')


def annual_return(returns, periods_per_year=12):
    """Calculate annualized return."""
    if len(returns) == 0:
        return np.nan
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or total_return <= 0:
        return -1.0
    return total_return ** (1 / n_years) - 1


def annual_volatility(returns, periods_per_year=12):
    """Calculate annualized volatility."""
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=12):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    std_val = excess.std()
    if std_val < 1e-8 or np.isnan(std_val):
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / std_val


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=12):
    """Calculate annualized Sortino ratio."""
    if len(returns) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std < 1e-8 or np.isnan(downside_std):
        return np.inf
    ann_ret = annual_return(returns, periods_per_year)
    return (ann_ret - risk_free_rate) / downside_std



def max_drawdown(returns):
    """Calculate maximum drawdown from return series."""
    if len(returns) == 0:
        return 0.0
    wealth = (1 + returns).cumprod()
    running_max = wealth.expanding().max()
    drawdown = wealth / running_max - 1
    return drawdown.min()


def max_drawdown_from_values(portfolio_values):
    """Calculate maximum drawdown from portfolio value series."""
    if len(portfolio_values) == 0:
        return 0.0
    running_max = portfolio_values.expanding().max()
    drawdown = portfolio_values / running_max - 1
    return drawdown.min()


def calmar_ratio(returns, periods_per_year=12):
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    ann_ret = annual_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.inf
    return ann_ret / mdd


def compute_all_metrics(backtest_result, strategy_name=None):
    """Compute all performance metrics for a backtest result.
    
    Parameters
    ----------
    backtest_result : pd.DataFrame
        Backtest result with columns: portfolio_value, portfolio_return,
        leverage, turnover, costs.
    strategy_name : str, optional
        Name of the strategy.
    
    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    if backtest_result.empty:
        return {}
    returns = backtest_result['portfolio_return']
    values = backtest_result['portfolio_value']
    
    metrics = {
        'strategy': strategy_name or 'Unknown',
        'annual_return': annual_return(returns),
        'annual_volatility': annual_volatility(returns),
        'sharpe_ratio': sharpe_ratio(returns),
        'sortino_ratio': sortino_ratio(returns),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns),
        'worst_month': returns.min(),
        'best_month': returns.max(),
        'avg_leverage': backtest_result['leverage'].mean(),
        'total_turnover': backtest_result['turnover'].sum(),
        'total_costs': backtest_result['costs'].sum(),
        'final_value': values.iloc[-1],
        'total_months': len(returns),
        'pct_positive_months': (returns > 0).mean()
    }
    
    return metrics


def compute_performance_table(backtest_results):
    """Compute performance table for all strategies.
    
    Parameters
    ----------
    backtest_results : dict
        Dictionary of backtest results.
    
    Returns
    -------
    pd.DataFrame
        Performance metrics for all strategies.
    """
    all_metrics = []
    for name, result in backtest_results.items():
        metrics = compute_all_metrics(result, strategy_name=name)
        if metrics:
            all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    if not df.empty:
        df = df.set_index('strategy')
    
    logger.info(f'Computed performance metrics for {len(df)} strategies')
    return df
