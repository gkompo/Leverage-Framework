"""02 - Regime Analysis

Visualizes and analyzes volatility regimes:
- Regime timeline
- Regime-conditioned return distributions
- Transition matrix
- Feature space visualization

Usage:
    python notebooks/02_regime_analysis.py
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
from src.volatility import compute_volatility_features, rolling_volatility
from src.regimes import detect_regimes
from src.utils import ensure_directory


REGIME_COLORS = {
    'Low Volatility': '#27ae60',
    'Normal Market': '#3498db',
    'High Volatility': '#f39c12',
    'Crisis': '#e74c3c',
    'Recovery': '#9b59b6'
}


def main():
    output_dir = os.path.join('paper', 'figures')
    ensure_directory(output_dir)
    
    # Load data and detect regimes
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    vol_features = compute_volatility_features(returns, prices)
    regimes, regime_stats, _, _ = detect_regimes(returns, prices, vol_features)
    
    print('=' * 60)
    print('REGIME ANALYSIS')
    print('=' * 60)
    print(f'\nRegime Statistics:')
    print(regime_stats.to_string())
    
    # Figure 1: Regime Timeline
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=100)
    
    # Price with regime coloring
    aligned_prices = prices.reindex(regimes.index)
    axes[0].plot(aligned_prices.index, aligned_prices.values, color='black', linewidth=0.5, alpha=0.5)
    for regime_name, color in REGIME_COLORS.items():
        mask = regimes == regime_name
        if mask.any():
            axes[0].fill_between(aligned_prices.index, 0, aligned_prices.values,
                                where=mask, alpha=0.3, color=color, label=regime_name)
    axes[0].set_yscale('log')
    axes[0].set_title('S&P 500 with Volatility Regimes', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (log scale)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Volatility with regimes
    vol_12m = rolling_volatility(returns, window=12)
    aligned_vol = vol_12m.reindex(regimes.index)
    axes[1].plot(aligned_vol.index, aligned_vol.values, color='black', linewidth=0.5)
    for regime_name, color in REGIME_COLORS.items():
        mask = regimes == regime_name
        if mask.any():
            axes[1].fill_between(aligned_vol.index, 0, aligned_vol.values,
                                where=mask, alpha=0.3, color=color)
    axes[1].set_title('12-Month Rolling Volatility by Regime', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Annualized Volatility')
    axes[1].grid(True, alpha=0.3)
    
    # Regime labels over time
    regime_codes = regimes.map({v: i for i, v in enumerate(REGIME_COLORS.keys())})
    axes[2].scatter(regimes.index, regime_codes, c=[REGIME_COLORS.get(r, 'gray') for r in regimes],
                    s=2, alpha=0.7)
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(list(REGIME_COLORS.keys()), fontsize=8)
    axes[2].set_title('Regime Classification Over Time', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {output_dir}/regime_timeline.png')
    
    # Figure 2: Regime-conditioned return distributions
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=100, sharey=True)
    aligned_returns = returns.reindex(regimes.index)
    
    for i, (regime_name, color) in enumerate(REGIME_COLORS.items()):
        mask = regimes == regime_name
        if mask.any():
            regime_returns = aligned_returns[mask].dropna()
            axes[i].hist(regime_returns, bins=40, color=color, alpha=0.7, density=True, edgecolor='white')
            axes[i].axvline(regime_returns.mean(), color='black', linestyle='--', linewidth=1.5)
            axes[i].set_title(f'{regime_name}\n(n={mask.sum()})', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Monthly Return')
            if i == 0:
                axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Return Distributions by Regime', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/regime_returns.png')
    
    # Transition matrix
    unique_regimes = list(REGIME_COLORS.keys())
    transition_matrix = pd.DataFrame(0.0, index=unique_regimes, columns=unique_regimes)
    
    for i in range(len(regimes) - 1):
        from_regime = regimes.iloc[i]
        to_regime = regimes.iloc[i + 1]
        if from_regime in unique_regimes and to_regime in unique_regimes:
            transition_matrix.loc[from_regime, to_regime] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(row_sums.replace(0, 1), axis=0)
    
    print(f'\nTransition Matrix:')
    print(transition_matrix.round(3).to_string())
    
    # Save transition matrix
    ensure_directory('results')
    transition_matrix.to_csv('results/transition_matrix.csv')
    print(f'Saved: results/transition_matrix.csv')
    
    print('\nRegime analysis complete.')


if __name__ == '__main__':
    main()
