"""03 - Backtest Analysis

Visualizes backtest results:
- Portfolio value comparison
- Leverage over time
- Drawdown comparison
- Crisis zoom-ins

Usage:
    python notebooks/03_backtest_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns
from src.volatility import rolling_volatility, compute_drawdown
from src.regimes import detect_regimes
from src.leverage import generate_all_strategies
from src.backtest import run_backtest, extract_portfolio_values
from src.risk_metrics import compute_performance_table
from src.utils import ensure_directory


STRATEGY_COLORS = {
    'Buy & Hold': '#95a5a6',
    'Fixed 1x': '#7f8c8d',
    'Fixed 2x': '#2c3e50',
    'Vol Target': '#3498db',
    'Regime': '#e74c3c',
    'Recovery Accel': '#9b59b6'
}


def main():
    output_dir = os.path.join('paper', 'figures')
    ensure_directory(output_dir)
    
    # Load data and run backtests
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    vol_features = compute_volatility_features(returns, prices)
    vol_12m = vol_features['vol_12m']
    regimes, _, _, _ = detect_regimes(returns, prices, vol_features)
    leverage_df = generate_all_strategies(returns, prices, vol_12m, regimes)
    backtest_results = run_backtest(returns, leverage_df)
    performance = compute_performance_table(backtest_results)
    
    print('=' * 60)
    print('BACKTEST ANALYSIS')
    print('=' * 60)
    print(f'\nPerformance Summary:')
    print(performance[['annual_return', 'annual_volatility', 'sharpe_ratio', 
                       'max_drawdown', 'calmar_ratio']].to_string())
    
    # Extract portfolio values
    port_values = extract_portfolio_values(backtest_results)
    
    # Figure 1: Portfolio Value Comparison
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    for strategy in port_values.columns:
        color = STRATEGY_COLORS.get(strategy, '#333333')
        linewidth = 2.0 if strategy in ['Regime', 'Recovery Accel'] else 1.0
        alpha = 1.0 if strategy in ['Buy & Hold', 'Regime', 'Recovery Accel'] else 0.6
        ax.plot(port_values.index, port_values[strategy], 
                label=strategy, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_yscale('log')
    ax.set_title('Portfolio Value Comparison (Log Scale)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {output_dir}/portfolio_comparison.png')
    
    # Figure 2: Leverage Over Time
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=100, sharex=True)
    
    for i, strategy in enumerate(['Vol Target', 'Regime', 'Recovery Accel']):
        if strategy in leverage_df.columns:
            color = STRATEGY_COLORS.get(strategy, '#333333')
            axes[i].plot(leverage_df.index, leverage_df[strategy], 
                        color=color, linewidth=0.8, alpha=0.8)
            axes[i].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            axes[i].set_title(f'{strategy} Leverage', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Leverage')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leverage_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/leverage_timeseries.png')
    
    # Figure 3: Drawdown Comparison
    fig, ax = plt.subplots(figsize=(16, 6), dpi=100)
    for strategy in ['Buy & Hold', 'Regime', 'Recovery Accel']:
        if strategy in port_values.columns:
            dd = compute_drawdown(port_values[strategy])
            color = STRATEGY_COLORS.get(strategy, '#333333')
            ax.plot(dd.index, dd.values, label=strategy, color=color, linewidth=0.8)
    ax.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/drawdown_comparison.png')
    
    # Figure 4: Performance bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=100)
    
    metrics = ['sharpe_ratio', 'max_drawdown', 'calmar_ratio']
    titles = ['Sharpe Ratio', 'Maximum Drawdown', 'Calmar Ratio']
    
    for ax_sub, metric, title in zip(axes, metrics, titles):
        values = performance[metric]
        colors = [STRATEGY_COLORS.get(s, '#333333') for s in values.index]
        ax_sub.barh(range(len(values)), values.values, color=colors, alpha=0.8)
        ax_sub.set_yticks(range(len(values)))
        ax_sub.set_yticklabels(values.index, fontsize=9)
        ax_sub.set_title(title, fontsize=12, fontweight='bold')
        ax_sub.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/performance_bars.png')
    
    print('\nBacktest analysis complete.')


if __name__ == '__main__':
    from src.volatility import compute_volatility_features
    main()
