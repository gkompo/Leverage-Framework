"""Regime detection experiment.

Runs the full regime detection pipeline and saves results.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns
from src.volatility import compute_volatility_features
from src.regimes import detect_regimes
from src.utils import setup_logging, ensure_directory, save_dataframe


def run_regime_detection():
    """Execute regime detection experiment."""
    logger = setup_logging()
    logger.info('=' * 60)
    logger.info('EXPERIMENT: Regime Detection')
    logger.info('=' * 60)
    
    # Load data
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    
    # Compute volatility features
    vol_features = compute_volatility_features(returns, prices)
    
    # Detect regimes
    regimes, regime_stats, gmm, scaler = detect_regimes(
        returns, prices, vol_features, n_regimes=5
    )
    
    # Save results
    ensure_directory('results')
    
    # Regime statistics
    save_dataframe(regime_stats, 'results/regime_statistics.csv')
    
    # Regime time series
    regime_ts = pd.DataFrame({
        'date': regimes.index,
        'regime': regimes.values
    })
    save_dataframe(regime_ts, 'results/regime_timeseries.csv', index=False)
    
    # Print summary
    logger.info('\nRegime Statistics:')
    logger.info(f'\n{regime_stats.to_string()}')
    
    return regimes, regime_stats


if __name__ == '__main__':
    run_regime_detection()
"""Regime detection experiment script."""
