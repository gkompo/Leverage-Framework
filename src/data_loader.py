"""Data loading module for Shiller S&P 500 dataset.

Loads the Robert Shiller S&P 500 monthly dataset from the local Excel file.
Handles European decimal conversion and date parsing.
"""

import os
import logging
import numpy as np
import pandas as pd
import urllib.request

from .utils import convert_european_decimal

logger = logging.getLogger('regime_leverage')

SHILLER_URL = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'shiller_data.xls')


def download_shiller_data(save_path=None):
    """Download Shiller dataset if not present locally."""
    save_path = save_path or DEFAULT_DATA_PATH
    if os.path.exists(save_path):
        logger.info(f'Shiller data already exists at {save_path}')
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f'Downloading Shiller data from {SHILLER_URL}...')
    try:
        # User agents can prevent 403 errors from some servers
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(SHILLER_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info(f'Saved Shiller data to {save_path}')
    except Exception as e:
        logger.error(f'Failed to download Shiller data: {e}')
        raise e
    return save_path


def load_shiller_data(filepath=None):
    """Load and clean the Shiller S&P 500 dataset.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the Excel file. Downloads if not present.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: Date, P, D, E, CPI
    """
    filepath = filepath or DEFAULT_DATA_PATH
    if not os.path.exists(filepath):
        download_shiller_data(filepath)
    
    logger.info(f'Loading Shiller data from {filepath}')
    
    # Read Excel with header at row 8 (0-indexed: skiprows=7)
    df = pd.read_excel(filepath, sheet_name='Data', skiprows=7)
    
    # Select and rename columns
    required_cols = ['Date', 'P', 'D', 'E', 'CPI']
    
    # Check if all required columns are in the dataframe
    if all(col in df.columns for col in required_cols):
        relevant = df[required_cols].copy()
    else:
        # Fallback to positional selection of first 5 columns
        relevant = df.iloc[:, :5].copy()
        relevant.columns = required_cols
    
    # Convert European decimals for numeric columns
    for col in ['P', 'D', 'E', 'CPI']:
        if col in relevant.columns:
            relevant[col] = relevant[col].apply(convert_european_decimal)
    
    # Parse dates - Shiller uses YYYY.MM format (e.g. 1871.01 for Jan 1871, 1871.1 for Oct 1871)
    def parse_shiller_date(d):
        try:
            d_str = f"{float(d):.4f}" # Ensure precision
            parts = d_str.split('.')
            year = int(parts[0])
            # The decimal part is the month. Shiller format is:
            # .01 = Jan, .02 = Feb, ..., .10 = Oct, .11 = Nov, .12 = Dec
            # Since float conversion might make it e.g. 1871.1 for Oct, we must handle it carefully
            dec_part = parts[1][:2]
            if len(dec_part) == 1:
                dec_part += '0'
            month = int(dec_part)
            if month == 0:
                month = 1
            if month > 12:
                month = 12
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, TypeError, IndexError):
            return pd.NaT
    
    relevant['Date'] = relevant['Date'].apply(parse_shiller_date)
    
    # Drop rows with missing critical data
    relevant = relevant.dropna(subset=['Date', 'P'])
    relevant = relevant[relevant['P'] > 0]
    
    # Sort by date and reset index
    relevant = relevant.sort_values('Date').reset_index(drop=True)
    
    logger.info(f'Loaded {len(relevant)} monthly observations from '
                f'{relevant["Date"].iloc[0].strftime("%Y-%m")} to '
                f'{relevant["Date"].iloc[-1].strftime("%Y-%m")}')
    
    return relevant
