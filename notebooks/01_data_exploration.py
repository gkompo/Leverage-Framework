"""01 - Data Exploration

Explores the Shiller S&P 500 dataset:
- Data overview and summary statistics
- Price history visualization
- Return distribution analysis
- Volatility analysis

Usage:
    python notebooks/01_data_exploration.py
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
from src.utils import ensure_directory


def main():
    # Setup
    output_dir = os.path.join('paper', 'figures')
    ensure_directory(output_dir)
    
    # Load data
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    
    print('=' * 60)
    print('DATA EXPLORATION: Shiller S&P 500')
    print('=' * 60)
    
    # Summary statistics
    print(f'\nDate range: {prices.index[0]} to {prices.index[-1]}')
    print(f'Observations: {len(prices)}')
    print(f'\nPrice Statistics:')
    print(prices.describe())
    
    print(f'\nMonthly Log Return Statistics:')
    print(returns.describe())
    print(f'Skewness: {returns.skew():.4f}')
    print(f'Kurtosis: {returns.kurtosis():.4f}')
    
    # Figure 1: Price History
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
    
    axes[0].plot(prices.index, prices.values, color='#1a5276', linewidth=0.8)
    axes[0].set_yscale('log')
    axes[0].set_title('S&P 500 Price History (Log Scale)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Returns
    axes[1].bar(returns.index, returns.values, color=np.where(returns >= 0, '#27ae60', '#e74c3c'),
                width=25, alpha=0.7)
    axes[1].set_title('Monthly Log Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    
    # Rolling volatility
    vol_12m = rolling_volatility(returns, window=12)
    vol_24m = rolling_volatility(returns, window=24)
    axes[2].plot(vol_12m.index, vol_12m.values, label='12-month', color='#e74c3c', linewidth=0.8)
    axes[2].plot(vol_24m.index, vol_24m.values, label='24-month', color='#3498db', linewidth=0.8)
    axes[2].set_title('Rolling Annualized Volatility', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {output_dir}/data_overview.png')
    
    # Figure 2: Return distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    axes[0].hist(returns.values, bins=80, density=True, color='#3498db', alpha=0.7, edgecolor='white')
    x = np.linspace(returns.min(), returns.max(), 200)
    from scipy.stats import norm
    axes[0].plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', linewidth=2, label='Normal')
    axes[0].set_title('Return Distribution vs Normal', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Monthly Log Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot
    from scipy.stats import probplot
    probplot(returns.values, dist='norm', plot=axes[1])
    axes[1].set_title('QQ Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/return_distribution.png')
    
    # Figure 3: Drawdown
    dd = compute_drawdown(prices)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
    ax.fill_between(dd.index, dd.values, 0, color='#e74c3c', alpha=0.5)
    ax.plot(dd.index, dd.values, color='#c0392b', linewidth=0.5)
    ax.set_title('Historical Drawdowns', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdowns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_dir}/drawdowns.png')
    
    print('\nData exploration complete.')


if __name__ == '__main__':
    main()
