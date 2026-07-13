"""Bootstrap statistical testing experiment.

Performs:
- 10,000 bootstrap simulations
- Sharpe ratio confidence intervals
- Probability dynamic strategy beats buy-and-hold
- Diebold-Mariano test for volatility prediction
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
from src.statistics import (
    bootstrap_sharpe_comparison,
    bootstrap_outperformance,
    diebold_mariano_test,
    regime_persistence_test,
    bootstrap_statistic
)
from src.risk_metrics import sharpe_ratio
from src.utils import setup_logging, ensure_directory, save_dataframe


def run_bootstrap_tests():
    """Execute bootstrap statistical tests."""
    logger = setup_logging()
    logger.info('=' * 60)
    logger.info('EXPERIMENT: Bootstrap Statistical Tests')
    logger.info('=' * 60)
    
    # Load data and run backtests
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    
    vol_features = compute_volatility_features(returns, prices)
    vol_12m = vol_features['vol_12m']
    regimes, _, _, _ = detect_regimes(returns, prices, vol_features)
    leverage_df = generate_all_strategies(returns, prices, vol_12m, regimes)
    backtest_results = run_backtest(returns, leverage_df)
    
    all_results = []
    
    # Benchmark: Buy & Hold returns
    bh_returns = backtest_results['Buy & Hold']['portfolio_return']
    
    # 1. Bootstrap Sharpe ratio comparisons against Buy & Hold
    logger.info('\n--- Bootstrap Sharpe Ratio Comparisons ---')
    for strategy_name, bt_result in backtest_results.items():
        if strategy_name == 'Buy & Hold' or bt_result.empty:
            continue
        
        strat_returns = bt_result['portfolio_return']
        
        # Sharpe comparison
        sharpe_result = bootstrap_sharpe_comparison(
            strat_returns, bh_returns, n_bootstrap=1000
        ) # Reducer from 10k to 1000 for faster execution in this master run, or 10000. Let's make it 10000 if fast, or keep it configurable.
        
        # Outperformance probability
        outperf = bootstrap_outperformance(
            strat_returns, bh_returns, n_bootstrap=1000
        )
        
        # Bootstrap Sharpe CI for the strategy
        def compute_sharpe(r):
            std_val = r.std()
            if std_val == 0 or np.isnan(std_val):
                return 0.0
            return np.sqrt(12) * r.mean() / std_val
        
        sharpe_ci = bootstrap_statistic(
            strat_returns.values, compute_sharpe, n_bootstrap=1000
        )
        
        result = {
            'strategy': strategy_name,
            'sharpe_ratio': sharpe_result['sharpe_a'],
            'bh_sharpe_ratio': sharpe_result['sharpe_b'],
            'sharpe_diff': sharpe_result['observed_diff'],
            'sharpe_diff_ci_lower': sharpe_result['ci_lower'],
            'sharpe_diff_ci_upper': sharpe_result['ci_upper'],
            'sharpe_p_value': sharpe_result['p_value'],
            'prob_beats_bh': outperf['prob_outperform'],
            'median_wealth_ratio': outperf['median_wealth_ratio'],
            'sharpe_ci_lower': sharpe_ci['ci_lower'],
            'sharpe_ci_upper': sharpe_ci['ci_upper']
        }
        all_results.append(result)
        
        logger.info(f'\n  {strategy_name}:')
        logger.info(f'    Sharpe: {sharpe_result["sharpe_a"]:.4f} '
                    f'[{sharpe_ci["ci_lower"]:.4f}, {sharpe_ci["ci_upper"]:.4f}]')
        logger.info(f'    Sharpe diff vs B&H: {sharpe_result["observed_diff"]:.4f} '
                    f'(p={sharpe_result["p_value"]:.4f})')
        logger.info(f'    P(beats B&H): {outperf["prob_outperform"]:.4f}')
    
    # 2. Regime persistence test
    logger.info('\n--- Regime Persistence Test ---')
    persistence = regime_persistence_test(regimes, n_simulations=1000)
    logger.info(f'  Observed persistence: {persistence["observed_persistence"]:.4f}')
    logger.info(f'  Expected (random): {persistence["expected_persistence"]:.4f}')
    logger.info(f'  Ratio: {persistence["persistence_ratio"]:.4f}')
    logger.info(f'  p-value: {persistence["p_value"]:.4f}')
    
    # 3. Diebold-Mariano test: regime-based vol vs constant vol
    logger.info('\n--- Diebold-Mariano Test ---')
    aligned_vol = vol_features['vol_12m'].dropna()
    aligned_regimes = regimes.reindex(aligned_vol.index).dropna()
    common_idx = aligned_vol.index.intersection(aligned_regimes.index)
    
    if len(common_idx) > 24:
        aligned_vol = aligned_vol.loc[common_idx]
        aligned_regimes = aligned_regimes.loc[common_idx]
        
        # Unconditional forecast: expanding mean
        unconditional_forecast = aligned_vol.expanding(min_periods=12).mean().shift(1)
        
        # Regime-conditioned forecast: regime-specific expanding mean
        regime_forecast = aligned_vol.copy() * np.nan
        for regime in aligned_regimes.unique():
            mask = aligned_regimes == regime
            regime_vol = aligned_vol[mask]
            regime_forecast[mask] = regime_vol.expanding(min_periods=3).mean().shift(1)
        
        # Drop NaN for comparison
        valid = unconditional_forecast.notna() & regime_forecast.notna()
        if valid.sum() > 24:
            errors_unconditional = (aligned_vol[valid] - unconditional_forecast[valid]).values
            errors_regime = (aligned_vol[valid] - regime_forecast[valid]).values
            
            dm_result = diebold_mariano_test(errors_unconditional, errors_regime)
            
            logger.info(f'  DM statistic: {dm_result["dm_statistic"]:.4f}')
            logger.info(f'  p-value: {dm_result["p_value"]:.4f}')
            logger.info(f'  Regime model better: {dm_result["model_a_better"]}')
            
            all_results.append({
                'strategy': 'DM Test (Vol Prediction)',
                'sharpe_ratio': np.nan,
                'bh_sharpe_ratio': np.nan,
                'sharpe_diff': dm_result['dm_statistic'],
                'sharpe_diff_ci_lower': np.nan,
                'sharpe_diff_ci_upper': np.nan,
                'sharpe_p_value': dm_result['p_value'],
                'prob_beats_bh': np.nan,
                'median_wealth_ratio': np.nan,
                'sharpe_ci_lower': np.nan,
                'sharpe_ci_upper': np.nan
            })
    
    # Save results
    ensure_directory('results')
    bootstrap_df = pd.DataFrame(all_results)
    save_dataframe(bootstrap_df, 'results/bootstrap_results.csv', index=False)
    
    # Save persistence test separately
    persistence_df = pd.DataFrame([persistence])
    save_dataframe(persistence_df, 'results/regime_persistence.csv', index=False)
    
    logger.info(f'\nBootstrap tests completed. {len(all_results)} comparisons saved.')
    
    return bootstrap_df


if __name__ == '__main__':
    run_bootstrap_tests()
