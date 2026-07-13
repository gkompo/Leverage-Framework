"""Utility functions for the Dynamic Regime Leverage project."""

import os
import logging
import numpy as np
import pandas as pd


def setup_logging(level=logging.INFO):
    """Configure project-wide logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('regime_leverage')


def ensure_directory(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def convert_european_decimal(value):
    """Convert European-style decimal (comma) to float."""
    if isinstance(value, str):
        value = value.replace(',', '.')
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def annualize_return(monthly_return, periods=12):
    """Annualize a monthly return."""
    return (1 + monthly_return) ** periods - 1


def annualize_volatility(monthly_vol, periods=12):
    """Annualize monthly volatility."""
    return monthly_vol * np.sqrt(periods)


def save_dataframe(df, filepath, index=True):
    """Save DataFrame to CSV, creating directories as needed."""
    ensure_directory(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)
    logging.getLogger('regime_leverage').info(f'Saved: {filepath}')


def format_pct(value, decimals=2):
    """Format a decimal as percentage string."""
    return f'{value * 100:.{decimals}f}%'


def format_number(value, decimals=4):
    """Format a number to fixed decimal places."""
    return f'{value:.{decimals}f}'
