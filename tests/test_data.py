"""Tests for data loading module."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns, compute_cumulative_returns
from src.utils import convert_european_decimal


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        try:
            cls.data = load_shiller_data()
            cls.data_loaded = True
        except Exception:
            cls.data = None
            cls.data_loaded = False
    
    def test_data_loads(self):
        """Test that data loads without errors."""
        if not self.data_loaded:
            self.skipTest('Data file not available')
        self.assertIsNotNone(self.data)
    
    def test_data_has_required_columns(self):
        """Test that data contains required columns."""
        if not self.data_loaded:
            self.skipTest('Data file not available')
        required = ['Date', 'P']
        for col in required:
            self.assertIn(col, self.data.columns)
    
    def test_data_has_positive_prices(self):
        """Test that all prices are positive."""
        if not self.data_loaded:
            self.skipTest('Data file not available')
        self.assertTrue((self.data['P'] > 0).all())
    
    def test_data_dates_are_sorted(self):
        """Test that dates are sorted chronologically."""
        if not self.data_loaded:
            self.skipTest('Data file not available')
        dates = self.data['Date']
        self.assertTrue((dates.diff().dropna() >= pd.Timedelta(0)).all())
    
    def test_data_has_sufficient_observations(self):
        """Test that we have enough data for analysis."""
        if not self.data_loaded:
            self.skipTest('Data file not available')
        self.assertGreater(len(self.data), 100)


class TestReturns(unittest.TestCase):
    """Tests for return calculations."""
    
    def test_log_returns_calculation(self):
        """Test log returns are calculated correctly."""
        prices = pd.Series([100, 110, 105, 115])
        returns = compute_log_returns(prices)
        expected = np.log(np.array([110/100, 105/110, 115/105]))
        np.testing.assert_array_almost_equal(returns.values, expected)
    
    def test_log_returns_length(self):
        """Test that log returns have correct length."""
        prices = pd.Series([100, 110, 105, 115])
        returns = compute_log_returns(prices)
        self.assertEqual(len(returns), 3)
    
    def test_cumulative_returns(self):
        """Test cumulative return calculation."""
        log_returns = pd.Series([0.01, 0.02, -0.01])
        cum = compute_cumulative_returns(log_returns, log_returns=True)
        expected = np.exp(np.cumsum([0.01, 0.02, -0.01]))
        np.testing.assert_array_almost_equal(cum.values, expected)


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_european_decimal_conversion(self):
        """Test European decimal conversion."""
        self.assertAlmostEqual(convert_european_decimal('1,5'), 1.5)
        self.assertAlmostEqual(convert_european_decimal('100,25'), 100.25)
    
    def test_normal_decimal(self):
        """Test that normal decimals are handled."""
        self.assertAlmostEqual(convert_european_decimal(1.5), 1.5)
        self.assertAlmostEqual(convert_european_decimal('1.5'), 1.5)
    
    def test_invalid_value(self):
        """Test that invalid values return NaN."""
        result = convert_european_decimal('abc')
        self.assertTrue(np.isnan(result))


if __name__ == '__main__':
    unittest.main()
