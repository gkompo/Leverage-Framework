"""Regime comparison experiment.

Fits a 5-component Gaussian Mixture Model (GMM) and a 5-component
Generalized Hyperbolic Mixture Model (GHMM) on S&P 500 returns.
Compares their goodness of fit (log-likelihood, AIC, BIC) and
backtests the leverage strategy under both frameworks.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.stats import genhyperbolic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shiller_data
from src.returns import compute_log_returns
from src.backtest import simulate_portfolio
from src.risk_metrics import compute_all_metrics
from src.utils import setup_logging, ensure_directory, save_dataframe


class GHMixtureModel:
    """1D Generalized Hyperbolic Mixture Model."""
    
    def __init__(self, n_components=5, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.weights_ = None
        self.params_ = None
        self.loglik_ = None
        self.n_params_ = None
        self.aic_ = None
        self.bic_ = None
        
    def _softmax(self, logits):
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)
        
    def fit(self, returns):
        from sklearn.mixture import GaussianMixture
        
        # Initialize component params with Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        gmm.fit(returns.reshape(-1, 1))
        
        # Sort components by mean
        sort_idx = np.argsort(gmm.means_.flatten())
        init_weights = gmm.weights_[sort_idx]
        init_means = gmm.means_.flatten()[sort_idx]
        init_stds = np.sqrt(gmm.covariances_.flatten())[sort_idx]
        
        # Softmax logits
        init_logits = np.log(init_weights + 1e-10)
        init_logits -= init_logits[-1]
        init_logits = init_logits[:-1]
        
        # Setup initial parameters for the 5 components: p, a, b, loc, scale
        init_params = []
        for k in range(self.n_components):
            mu_k = init_means[k]
            sig_k = max(init_stds[k], 1e-4)
            
            p_init = 0.5
            a_init = 1.0 / sig_k
            b_init = 0.0
            loc_init = mu_k
            scale_init = sig_k
            
            init_params.extend([
                p_init,
                np.log(a_init),
                0.0,  # tanh(0) = 0 for b_init
                loc_init,
                np.log(scale_init)
            ])
            
        x0 = np.concatenate([init_logits, init_params])
        
        def nll(params_arr):
            logits = np.append(params_arr[:self.n_components-1], 0.0)
            weights = self._softmax(logits)
            
            p_idx = self.n_components - 1
            log_density = np.zeros(len(returns))
            
            for k in range(self.n_components):
                offset = p_idx + k * 5
                p = params_arr[offset]
                a = np.exp(params_arr[offset+1])
                b_ratio = np.tanh(params_arr[offset+2])
                b = b_ratio * a
                loc = params_arr[offset+3]
                scale = np.exp(params_arr[offset+4])
                
                try:
                    pdf_val = genhyperbolic.pdf(returns, p, a, b, loc=loc, scale=scale)
                    pdf_val = np.nan_to_num(pdf_val, nan=0.0, posinf=0.0, neginf=0.0)
                    pdf_val = np.maximum(pdf_val, 1e-15)
                except Exception:
                    pdf_val = np.full(len(returns), 1e-15)
                
                log_density = np.log(np.exp(log_density) + weights[k] * pdf_val) if k > 0 else np.log(weights[k] * pdf_val)
                
            return -np.sum(log_density)
            
        res = opt.minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 200})
        
        # Extract parameters
        opt_logits = np.append(res.x[:self.n_components-1], 0.0)
        self.weights_ = self._softmax(opt_logits)
        
        self.params_ = []
        p_idx = self.n_components - 1
        for k in range(self.n_components):
            offset = p_idx + k * 5
            p = res.x[offset]
            a = np.exp(res.x[offset+1])
            b = np.tanh(res.x[offset+2]) * a
            loc = res.x[offset+3]
            scale = np.exp(res.x[offset+4])
            
            self.params_.append({
                'p': p,
                'a': a,
                'b': b,
                'loc': loc,
                'scale': scale
            })
            
        self.loglik_ = -res.fun
        self.n_params_ = len(x0)
        self.aic_ = 2 * self.n_params_ - 2 * self.loglik_
        self.bic_ = self.n_params_ * np.log(len(returns)) - 2 * self.loglik_
        
    def predict_proba(self, returns):
        n = len(returns)
        densities = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            p = self.params_[k]
            try:
                densities[:, k] = genhyperbolic.pdf(
                    returns, p['p'], p['a'], p['b'], loc=p['loc'], scale=p['scale']
                )
                densities[:, k] = np.maximum(densities[:, k], 1e-15)
            except Exception:
                densities[:, k] = 1e-15
                
        weighted_densities = densities * self.weights_
        row_sums = weighted_densities.sum(axis=1, keepdims=True)
        return weighted_densities / np.maximum(row_sums, 1e-15)


def run_comparison():
    logger = setup_logging()
    logger.info('=' * 60)
    logger.info('EXPERIMENT: Regime Model Comparison (GMM vs GHMM)')
    logger.info('=' * 60)
    
    data = load_shiller_data()
    prices = data.set_index('Date')['P']
    returns = compute_log_returns(prices)
    
    ret_arr = returns.values
    
    # 1. Fit GMM on 1D returns
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(ret_arr.reshape(-1, 1))
    
    gmm_loglik = gmm.score(ret_arr.reshape(-1, 1)) * len(ret_arr)
    gmm_n_params = 5 * 2 + 4 # 5 means, 5 variances, 4 weights = 14
    gmm_aic = 2 * gmm_n_params - 2 * gmm_loglik
    gmm_bic = gmm_n_params * np.log(len(ret_arr)) - 2 * gmm_loglik
    
    # Sort components by mean
    gmm_sort_idx = np.argsort(gmm.means_.flatten())
    gmm_probs = gmm.predict_proba(ret_arr.reshape(-1, 1))[:, gmm_sort_idx]
    
    # 2. Fit GHMM on 1D returns
    ghmm = GHMixtureModel(n_components=5, random_state=42)
    ghmm.fit(ret_arr)
    
    ghmm_probs = ghmm.predict_proba(ret_arr)
    
    # 3. Model Fit Comparison Table
    fit_comparison = pd.DataFrame({
        'Model': ['GMM (Gaussian)', 'GHMM (Generalized Hyperbolic)'],
        'Log-Likelihood': [gmm_loglik, ghmm.loglik_],
        'Num Parameters': [gmm_n_params, ghmm.n_params_],
        'AIC': [gmm_aic, ghmm.aic_],
        'BIC': [gmm_bic, ghmm.bic_]
    }).set_index('Model')
    
    save_dataframe(fit_comparison, 'results/regime_model_fit.csv')
    logger.info('\nModel Fit Comparison:')
    logger.info(f'\n{fit_comparison.to_string()}')
    
    # 4. Generate Leverage based on posteriors
    # Define leverage mappings for the 5 components (sorted from low to high mean returns / volatility)
    # 0: Crisis, 1: High Vol, 2: Normal, 3: Low Vol, 4: Recovery (mapped by mean returns)
    leverage_map = np.array([0.125, 0.50, 1.00, 1.75, 1.375])
    
    # To avoid look-ahead bias, use lagged posteriors
    gmm_lagged_probs = np.roll(gmm_probs, 1, axis=0)
    gmm_lagged_probs[0, :] = gmm_probs[0, :] # Fallback for first element
    
    ghmm_lagged_probs = np.roll(ghmm_probs, 1, axis=0)
    ghmm_lagged_probs[0, :] = ghmm_probs[0, :]
    
    # Expected leverage as weighted average
    gmm_leverage = pd.Series(np.dot(gmm_lagged_probs, leverage_map), index=returns.index)
    ghmm_leverage = pd.Series(np.dot(ghmm_lagged_probs, leverage_map), index=returns.index)
    
    # Backtests
    gmm_backtest = simulate_portfolio(returns, gmm_leverage)
    ghmm_backtest = simulate_portfolio(returns, ghmm_leverage)
    bh_backtest = simulate_portfolio(returns, pd.Series(1.0, index=returns.index))
    
    # Compute performance metrics
    gmm_perf = compute_all_metrics(gmm_backtest, 'GMM Returns')
    ghmm_perf = compute_all_metrics(ghmm_backtest, 'GHMM Returns')
    bh_perf = compute_all_metrics(bh_backtest, 'Buy & Hold')
    
    performance_comparison = pd.DataFrame([bh_perf, gmm_perf, ghmm_perf]).set_index('strategy')
    
    save_dataframe(performance_comparison, 'results/regime_comparison_performance.csv')
    logger.info('\nStrategy Performance Comparison:')
    display_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio', 
                    'sortino_ratio', 'max_drawdown', 'calmar_ratio']
    logger.info(f'\n{performance_comparison[display_cols].to_string()}')
    
    return fit_comparison, performance_comparison


if __name__ == '__main__':
    run_comparison()
