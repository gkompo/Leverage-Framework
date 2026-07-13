"""Tests for backtesting engine."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import simulate_portfolio
from src.leverage import buy_and_hold_leverage, fixed_leverage


class TestBacktest(unittest.TestCase):
    """Tests for backtesting functionality."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n = 60  # 5 years monthly
        dates = pd.date_range('2000-01-01', periods=self.n, freq='MS')
        self.returns = pd.Series(
            np.random.normal(0.005, 0.04, self.n),
            index=dates,
            name='returns'
        )
    
    def test_buy_and_hold_no_costs(self):
        """Test buy-and-hold with zero transaction costs."""
        leverage = buy_and_hold_leverage(self.returns)
        result = simulate_portfolio(
            self.returns, leverage, 
            initial_value=1.0, transaction_cost=0.0
        )
        self.assertEqual(len(result), len(self.returns))
        # With 1x leverage and no costs, should match simple compounding
        simple_returns = np.exp(self.returns) - 1
        expected_final = np.prod(1 + simple_returns)
        self.assertAlmostEqual(
            result['portfolio_value'].iloc[-1], expected_final, places=4
        )
    
    def test_portfolio_starts_correctly(self):
        """Test that portfolio starts near initial value."""
        leverage = buy_and_hold_leverage(self.returns)
        result = simulate_portfolio(self.returns, leverage, initial_value=100.0)
        # First value should be initial * (1 + first return)
        first_ret = np.exp(self.returns.iloc[0]) - 1
        expected = 100.0 * (1 + first_ret)
        self.assertAlmostEqual(result['portfolio_value'].iloc[0], expected, places=2)
    
    def test_fixed_leverage_increases_volatility(self):
        """Test that 2x leverage increases portfolio volatility."""
        lev_1x = buy_and_hold_leverage(self.returns)
        lev_2x = fixed_leverage(self.returns, 2.0)
        
        result_1x = simulate_portfolio(self.returns, lev_1x, transaction_cost=0.0)
        result_2x = simulate_portfolio(self.returns, lev_2x, transaction_cost=0.0)
        
        vol_1x = result_1x['portfolio_return'].std()
        vol_2x = result_2x['portfolio_return'].std()
        
        self.assertGreater(vol_2x, vol_1x)
    
    def test_transaction_costs_reduce_value(self):
        """Test that transaction costs reduce portfolio value."""
        leverage = pd.Series(
            np.where(np.arange(len(self.returns)) % 2 == 0, 1.0, 2.0),
            index=self.returns.index
        )
        result_no_cost = simulate_portfolio(
            self.returns, leverage, transaction_cost=0.0
        )
        result_with_cost = simulate_portfolio(
            self.returns, leverage, transaction_cost=0.001
        )
        self.assertLess(
            result_with_cost['portfolio_value'].iloc[-1],
            result_no_cost['portfolio_value'].iloc[-1]
        )
    
    def test_zero_returns_preserve_value(self):
        """Test portfolio with zero returns stays at initial value."""
        zero_returns = pd.Series(0.0, index=self.returns.index)
        leverage = buy_and_hold_leverage(zero_returns)
        result = simulate_portfolio(
            zero_returns, leverage, initial_value=100.0, transaction_cost=0.0
        )
        np.testing.assert_array_almost_equal(
            result['portfolio_value'].values,
            np.full(len(zero_returns), 100.0)
        )


if __name__ == '__main__':
    unittest.main()
