"""Crisis period analysis.

Analyzes leverage behavior and portfolio performance during
historical crisis periods:
- 1929 Great Depression
- 1937 Recession
- 1973 Oil Crisis
- 1987 Black Monday
- 2000 Dot-com Bubble
- 2008 Global Financial Crisis
- 2020 COVID-19 Crash
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns
from src.volatility import rolling_volatility, compute_volatility_features, compute_drawdown
from src.regimes import detect_regimes
from src.leverage import generate_all_strategies
from src.backtest import run_backtest
from src.risk_metrics import max_drawdown_from_values
from src.utils import setup_logging, ensure_directory, save_dataframe


# Crisis periods: (name, start, end, recovery_end)
CRISIS_PERIODS = [
    ('1929 Great Depression', '1929-09', '1932-06', '1936-12'),
    ('1937 Recession', '1937-03', '1938-03', '1940-12'),
    ('1973 Oil Crisis', '1973-01', '1974-10', '1976-12'),
    ('1987 Black Monday', '1987-08', '1987-12', '1989-06'),
    ('2000 Dot-com Bubble', '2000-03', '2002-10', '2006-06'),
    ('2008 Financial Crisis', '2007-10', '2009-03', '2012-12'),
    ('2020 COVID Crash', '2020-02', '2020-03', '2020-12'),
]


def analyze_crisis_period(name, start, end, recovery_end, 
                          backtest_results, leverage_df, regimes):
    """Analyze a single crisis period.
    
    Parameters
    ----------
    name : str
        Crisis name.
    start, end, recovery_end : str
        Period boundaries.
    backtest_results : dict
        Backtest results.
    leverage_df : pd.DataFrame
        Leverage values.
    regimes : pd.Series
        Regime labels.
    
    Returns
    -------
    dict
        Crisis analysis metrics.
    """
    results = {'crisis': name}
    
    for strategy_name, bt_result in backtest_results.items():
        if bt_result.empty:
            continue
        values = bt_result['portfolio_value']
        leverage = bt_result['leverage']
        
        # Before crisis (6 months)
        try:
            before_start = pd.Timestamp(start) - pd.DateOffset(months=6)
            before_mask = (values.index >= before_start) & (values.index < pd.Timestamp(start))
            if before_mask.any():
                results[f'{strategy_name}_leverage_before'] = leverage[before_mask].mean()
            else:
                results[f'{strategy_name}_leverage_before'] = np.nan
        except Exception:
            results[f'{strategy_name}_leverage_before'] = np.nan
        
        # During crisis
        try:
            during_mask = (values.index >= pd.Timestamp(start)) & (values.index <= pd.Timestamp(end))
            if during_mask.any():
                crisis_values = values[during_mask]
                results[f'{strategy_name}_max_dd'] = max_drawdown_from_values(crisis_values)
                results[f'{strategy_name}_leverage_during'] = leverage[during_mask].mean()
                results[f'{strategy_name}_wealth_lost'] = (
                    crisis_values.iloc[-1] / crisis_values.iloc[0] - 1
                )
            else:
                results[f'{strategy_name}_max_dd'] = np.nan
                results[f'{strategy_name}_leverage_during'] = np.nan
                results[f'{strategy_name}_wealth_lost'] = np.nan
        except Exception:
            results[f'{strategy_name}_max_dd'] = np.nan
            results[f'{strategy_name}_leverage_during'] = np.nan
            results[f'{strategy_name}_wealth_lost'] = np.nan
        
        # Recovery
        try:
            recovery_mask = (values.index > pd.Timestamp(end)) & (values.index <= pd.Timestamp(recovery_end))
            if recovery_mask.any():
                results[f'{strategy_name}_leverage_recovery'] = leverage[recovery_mask].mean()
                # Recovery duration: months to reach pre-crisis peak
                # Let's define the peak as the max value prior to start
                pre_crisis_peak = values[values.index <= pd.Timestamp(start)].max()
                recovered = values[values.index > pd.Timestamp(end)]
                recovery_months = 0
                recovered_flag = False
                for idx, val in recovered.items():
                    recovery_months += 1
                    if val >= pre_crisis_peak:
                        recovered_flag = True
                        break
                results[f'{strategy_name}_recovery_months'] = recovery_months if recovered_flag else len(recovered)
            else:
                results[f'{strategy_name}_leverage_recovery'] = np.nan
                results[f'{strategy_name}_recovery_months'] = np.nan
        except Exception:
            results[f'{strategy_name}_leverage_recovery'] = np.nan
            results[f'{strategy_name}_recovery_months'] = np.nan
    
    return results


def run_crisis_analysis():
    """Execute crisis analysis experiment."""
    logger = setup_logging()
    logger.info('=' * 60)
    logger.info('EXPERIMENT: Crisis Analysis')
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
    
    # Analyze each crisis
    all_results = []
    for crisis_name, start, end, recovery_end in CRISIS_PERIODS:
        logger.info(f'\nAnalyzing: {crisis_name}')
        try:
            result = analyze_crisis_period(
                crisis_name, start, end, recovery_end,
                backtest_results, leverage_df, regimes
            )
            all_results.append(result)
            logger.info(f'  Completed analysis for {crisis_name}')
        except Exception as e:
            logger.warning(f'  Could not analyze {crisis_name}: {e}')
    
    # Save results
    ensure_directory('results')
    crisis_df = pd.DataFrame(all_results)
    if not crisis_df.empty:
        crisis_df = crisis_df.set_index('crisis')
    save_dataframe(crisis_df, 'results/crisis_results.csv')
    
    logger.info(f'\nCrisis analysis completed for {len(all_results)} periods')
    
    return crisis_df


if __name__ == '__main__':
    run_crisis_analysis()
