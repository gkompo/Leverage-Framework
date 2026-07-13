"""Master execution script.

Runs the complete research pipeline:
1. Data loading
2. Regime estimation
3. Leverage backtests
4. Crisis analysis
5. Bootstrap tests
6. Generate paper tables
"""

import os
import sys
import time
import logging

# Ensure we're running from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, ensure_directory


def generate_latex_tables():
    """Generate LaTeX tables from results for the paper."""
    import pandas as pd
    
    logger = logging.getLogger('regime_leverage')
    tables_dir = os.path.join('paper', 'tables')
    ensure_directory(tables_dir)
    
    # Table 1: Regime Statistics
    try:
        regime_stats = pd.read_csv('results/regime_statistics.csv', index_col=0)
        latex = regime_stats.to_latex(
            float_format='%.4f',
            caption='Volatility Regime Statistics',
            label='tab:regime_stats'
        )
        with open(os.path.join(tables_dir, 'regime_statistics.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: regime_statistics.tex')
    except Exception as e:
        logger.warning(f'Could not generate regime stats table: {e}')
    
    # Table 2: Performance Comparison
    try:
        performance = pd.read_csv('results/performance.csv', index_col=0)
        display_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio',
                        'sortino_ratio', 'max_drawdown', 'calmar_ratio',
                        'avg_leverage']
        available_cols = [c for c in display_cols if c in performance.columns]
        latex = performance[available_cols].to_latex(
            float_format='%.4f',
            caption='Strategy Performance Comparison',
            label='tab:performance'
        )
        with open(os.path.join(tables_dir, 'performance.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: performance.tex')
    except Exception as e:
        logger.warning(f'Could not generate performance table: {e}')
    
    # Table 3: Crisis Results
    try:
        crisis = pd.read_csv('results/crisis_results.csv', index_col=0)
        latex = crisis.to_latex(
            float_format='%.4f',
            caption='Crisis Period Analysis',
            label='tab:crisis'
        )
        with open(os.path.join(tables_dir, 'crisis_results.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: crisis_results.tex')
    except Exception as e:
        logger.warning(f'Could not generate crisis table: {e}')
    
    # Table 4: Bootstrap Results
    try:
        bootstrap = pd.read_csv('results/bootstrap_results.csv')
        latex = bootstrap.to_latex(
            float_format='%.4f',
            index=False,
            caption='Bootstrap Statistical Tests',
            label='tab:bootstrap'
        )
        with open(os.path.join(tables_dir, 'bootstrap_results.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: bootstrap_results.tex')
    except Exception as e:
        logger.warning(f'Could not generate bootstrap table: {e}')
        
    # Table 5: GMM vs GHMM Fit Comparison
    try:
        fit_comp = pd.read_csv('results/regime_model_fit.csv', index_col=0)
        latex = fit_comp.to_latex(
            float_format='%.4f',
            caption='GMM vs GHMM Model Fit on 1D Returns',
            label='tab:fit_comparison'
        )
        with open(os.path.join(tables_dir, 'regime_model_fit.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: regime_model_fit.tex')
    except Exception as e:
        logger.warning(f'Could not generate model fit table: {e}')
        
    # Table 6: GMM vs GHMM Performance
    try:
        perf_comp = pd.read_csv('results/regime_comparison_performance.csv', index_col=0)
        display_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio',
                        'sortino_ratio', 'max_drawdown', 'calmar_ratio']
        available_cols = [c for c in display_cols if c in perf_comp.columns]
        latex = perf_comp[available_cols].to_latex(
            float_format='%.4f',
            caption='Strategy Performance: GMM vs GHMM Regimes',
            label='tab:comparison_performance'
        )
        with open(os.path.join(tables_dir, 'regime_comparison_performance.tex'), 'w') as f:
            f.write(latex)
        logger.info('Generated table: regime_comparison_performance.tex')
    except Exception as e:
        logger.warning(f'Could not generate comparison performance table: {e}')


def main():
    """Run complete research pipeline."""
    logger = setup_logging()
    start_time = time.time()
    
    logger.info('=' * 70)
    logger.info('DYNAMIC OPTIMAL LEVERAGE UNDER VOLATILITY REGIMES')
    logger.info('A Regime-Switching Framework for Risk Allocation')
    logger.info('=' * 70)
    
    # Ensure output directories exist
    ensure_directory('results')
    ensure_directory(os.path.join('paper', 'figures'))
    ensure_directory(os.path.join('paper', 'tables'))
    
    # Step 1: Data Loading
    logger.info('\n' + '=' * 50)
    logger.info('STEP 1: Data Loading')
    logger.info('=' * 50)
    from src.data_loader import load_shiller_data
    data = load_shiller_data()
    logger.info(f'Loaded {len(data)} observations')
    
    # Step 2: Regime Estimation
    logger.info('\n' + '=' * 50)
    logger.info('STEP 2: Regime Estimation')
    logger.info('=' * 50)
    from experiments.regime_detection import run_regime_detection
    regimes, regime_stats = run_regime_detection()
    
    # Step 3: Leverage Backtests
    logger.info('\n' + '=' * 50)
    logger.info('STEP 3: Leverage Backtests')
    logger.info('=' * 50)
    from experiments.leverage_backtest import run_leverage_backtest
    performance, backtest_results = run_leverage_backtest()
    
    # Step 4: Crisis Analysis
    logger.info('\n' + '=' * 50)
    logger.info('STEP 4: Crisis Analysis')
    logger.info('=' * 50)
    from experiments.crisis_analysis import run_crisis_analysis
    crisis_df = run_crisis_analysis()
    
    # Step 5: Bootstrap Tests
    logger.info('\n' + '=' * 50)
    logger.info('STEP 5: Bootstrap Statistical Tests')
    logger.info('=' * 50)
    from experiments.bootstrap import run_bootstrap_tests
    bootstrap_df = run_bootstrap_tests()
    
    # Step 6: GMM vs GHMM Regime Comparison
    logger.info('\n' + '=' * 50)
    logger.info('STEP 6: GMM vs GHMM Regime Comparison')
    logger.info('=' * 50)
    from experiments.regime_comparison import run_comparison
    run_comparison()
    
    # Step 7: Generate Paper Tables
    logger.info('\n' + '=' * 50)
    logger.info('STEP 7: Generate Paper Tables')
    logger.info('=' * 50)
    generate_latex_tables()
    
    # Summary
    elapsed = time.time() - start_time
    logger.info('\n' + '=' * 70)
    logger.info(f'Research project completed successfully')
    logger.info(f'Total execution time: {elapsed:.1f} seconds')
    logger.info('=' * 70)
    logger.info('\nOutput files:')
    for root, dirs, files in os.walk('results'):
        for f in files:
            logger.info(f'  {os.path.join(root, f)}')
    for root, dirs, files in os.walk(os.path.join('paper', 'tables')):
        for f in files:
            logger.info(f'  {os.path.join(root, f)}')
    
    print('\nResearch project completed successfully')


if __name__ == '__main__':
    main()
