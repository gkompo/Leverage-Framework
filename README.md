# Dynamic Optimal Leverage Under Volatility Regimes

> **A Regime-Switching Framework for Risk Allocation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-black.svg)](https://peps.python.org/pep-0008/)

---

## Overview

This repository contains the complete research code, data pipeline, and academic paper for the study:

**"Dynamic Optimal Leverage Under Volatility Regimes: A Regime-Switching Framework for Risk Allocation"**

The research investigates whether a volatility-regime-based dynamic leverage strategy can improve long-term risk-adjusted returns while reducing drawdowns compared with traditional fixed leverage and volatility targeting approaches.

## Research Questions

1. **Do volatility regimes contain useful information about future risk?** — We test whether market volatility exhibits persistent regime behavior using Gaussian Mixture Models.
2. **Can crisis regimes be detected early enough to reduce portfolio losses?** — We evaluate whether regime signals provide timely warnings before major drawdowns.
3. **Does increasing leverage during recovery regimes improve long-term compounded returns?** — We implement a recovery acceleration model that gradually increases exposure during market recoveries.
4. **Is dynamic leverage statistically superior to constant leverage?** — We conduct bootstrap hypothesis tests with 10,000 simulations to assess statistical significance.
5. **How much drawdown reduction is achieved during historical crises?** — We analyze seven major market crises from 1929 to 2020.

## Methodology

### Volatility Regime Detection

We employ a **5-regime Gaussian Mixture Model (GMM)** trained on volatility features:
- 12-month rolling realized volatility
- 3-month volatility change
- Current drawdown from peak

The GMM identifies five distinct market states: Low Volatility, Normal Market, High Volatility, Crisis, and Recovery.

### Leverage Strategies

- **Buy \& Hold**: Constant 1x exposure.
- **Fixed Leverage**: Constant 1x and 2x.
- **Volatility Targeting**: $L_t = \sigma_{\text{target}} / \hat{\sigma}_t$.
- **Regime Leverage**: Regime-conditioned leverage multipliers.
- **Recovery Acceleration**: Regime leverage + recovery score boost.

### Statistical Tests

- **Bootstrap confidence intervals** (10,000 iterations)
- **Sharpe ratio significance tests**
- **Diebold-Mariano test** for predictive accuracy
- **Regime persistence test** vs. random permutation null

## Repository Structure

```
Dynamic-Regime-Leverage/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── run_all.py                   # Master execution script
│
├── data/                        # Data directory (auto-populated)
│   └── shiller_data.xls         # Shiller S&P 500 dataset
│
├── src/                         # Core research library
│   ├── data_loader.py           # Data loading and cleaning
│   ├── returns.py               # Return calculations
│   ├── volatility.py            # Volatility estimation
│   ├── regimes.py               # GMM regime detection
│   ├── leverage.py              # Leverage strategy implementations
│   ├── backtest.py              # Backtesting engine
│   ├── risk_metrics.py          # Performance metrics
│   ├── statistics.py            # Statistical testing
│   └── utils.py                 # Utility functions
│
├── experiments/                 # Experiment scripts
│   ├── regime_detection.py      # Regime detection pipeline
│   ├── leverage_backtest.py     # Strategy backtests
│   ├── crisis_analysis.py       # Historical crisis study
│   └── bootstrap.py             # Bootstrap statistical tests
│
├── results/                     # Generated results (CSV)
│   ├── regime_statistics.csv
│   ├── performance.csv
│   ├── crisis_results.csv
│   └── bootstrap_results.csv
│
├── paper/                       # Academic paper (LaTeX)
│   ├── main.tex                 # Main paper source
│   ├── references.bib           # Bibliography
│   ├── figures/                 # Generated figures
│   └── tables/                  # Generated LaTeX tables
│
├── notebooks/                   # Jupyter notebooks / Scripts
│   ├── 01_data_exploration.py   # Data exploration
│   ├── 02_regime_analysis.py    # Regime visualization
│   └── 03_backtest_analysis.py  # Backtest results
│
└── tests/                       # Unit tests
    ├── test_data.py
    ├── test_backtest.py
    └── test_metrics.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Dynamic-Regime-Leverage.git
cd Dynamic-Regime-Leverage

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

Run the complete research pipeline with a single command:

```bash
python run_all.py
```

This will run all data extraction, regime fitting, strategy backtests, stats significance testing, and output LaTeX tables directly to the `paper/tables/` folder.

## License

MIT License. See LICENSE for details.
