"""Tests for risk metrics module."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_metrics import (
    annual_return, annual_volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, calmar_ratio
)


class TestRiskMetrics(unittest.TestCase):
    """Tests for risk metrics calculations."""
    
    def setUp(self):
        """Create test return series."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.005, 0.03, 120))
    
    def test_annual_return_positive(self):
        """Test annualized return for positive series."""
        positive_returns = pd.Series([0.01] * 12)
        result = annual_return(positive_returns)
        expected = (1.01 ** 12) - 1
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_annual_volatility(self):
        """Test annualized volatility calculation."""
        monthly_vol = self.returns.std()
        result = annual_volatility(self.returns)
        expected = monthly_vol * np.sqrt(12)
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_sharpe_ratio_positive_mean(self):
        """Test Sharpe ratio is positive for positive mean returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        result = sharpe_ratio(positive_returns)
        self.assertGreater(result, 0)
    
    def test_sharpe_ratio_zero_vol(self):
        """Test Sharpe ratio handles zero volatility."""
        constant_returns = pd.Series([0.01] * 10)
        result = sharpe_ratio(constant_returns)
        self.assertEqual(result, 0.0)
    
    def test_max_drawdown_negative(self):
        """Test that max drawdown is negative."""
        result = max_drawdown(self.returns)
        self.assertLessEqual(result, 0)
    
    def test_max_drawdown_known(self):
        """Test max drawdown for a known sequence."""
        returns = pd.Series([0.1, -0.3, 0.05, 0.05])
        result = max_drawdown(returns)
        # Peak at 1.1, then drops to 1.1*0.7 = 0.77, dd = 0.77/1.1-1 = -0.3
        self.assertAlmostEqual(result, -0.3, places=4)
    
    def test_sortino_ratio_positive(self):
        """Test Sortino ratio for positive mean series."""
        result = sortino_ratio(self.returns)
        # Should be finite for typical data
        self.assertTrue(np.isfinite(result))
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        result = calmar_ratio(self.returns)
        self.assertTrue(np.isfinite(result))
    
    def test_no_loss_sortino(self):
        """Test Sortino ratio when there are no losses."""
        positive_only = pd.Series([0.01, 0.02, 0.015])
        result = sortino_ratio(positive_only)
        self.assertEqual(result, np.inf)


if __name__ == '__main__':
    unittest.main()
