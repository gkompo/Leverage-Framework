"""Leverage backtest experiment.

Runs backtests for all leverage strategies and saves performance results.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns
from src.volatility import rolling_volatility, compute_volatility_features
from src.regimes import detect_regimes
from src.leverage import generate_all_strategies
from src.backtest import run_backtest
from src.risk_metrics import compute_performance_table
from src.utils import setup_logging, ensure_directory, save_dataframe


def run_leverage_backtest():
    """Execute leverage backtest experiment."""
    logger = setup_logging()
    logger.info('=' * 60)
    logger.info('EXPERIMENT: Leverage Backtest')
    logger.info('=' * 60)
    
    # Load data
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    
    # Volatility features and regimes
    vol_features = compute_volatility_features(returns, prices)
    vol_12m = vol_features['vol_12m']
    
    # Detect regimes
    regimes, regime_stats, _, _ = detect_regimes(returns, prices, vol_features)
    
    # Generate all leverage strategies
    leverage_df = generate_all_strategies(returns, prices, vol_12m, regimes)
    
    # Run backtests
    backtest_results = run_backtest(returns, leverage_df)
    
    # Compute performance metrics
    performance = compute_performance_table(backtest_results)
    
    # Save results
    ensure_directory('results')
    save_dataframe(performance, 'results/performance.csv')
    
    # Save portfolio values
    portfolio_values = pd.DataFrame({
        name: result['portfolio_value'] 
        for name, result in backtest_results.items()
        if not result.empty
    })
    save_dataframe(portfolio_values, 'results/portfolio_values.csv')
    
    # Print performance summary
    logger.info('\nPerformance Summary:')
    display_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio', 
                    'sortino_ratio', 'max_drawdown', 'calmar_ratio']
    available_cols = [c for c in display_cols if c in performance.columns]
    logger.info(f'\n{performance[available_cols].to_string()}')
    
    return performance, backtest_results


if __name__ == '__main__':
    run_leverage_backtest()
