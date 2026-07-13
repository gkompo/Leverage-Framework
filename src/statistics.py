"""Statistical testing module.

Implements:
- Bootstrap confidence intervals
- Sharpe ratio significance tests
- Diebold-Mariano test for predictive performance
- Regime persistence tests
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats as scipy_stats

logger = logging.getLogger('regime_leverage')


def bootstrap_statistic(data, stat_func, n_bootstrap=10000, 
                        confidence_level=0.95, random_state=42):
    """Bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to bootstrap.
    stat_func : callable
        Function that computes the statistic from a sample.
    n_bootstrap : int
        Number of bootstrap iterations.
    confidence_level : float
        Confidence level for interval.
    random_state : int
        Random seed.
    
    Returns
    -------
    dict
        Statistic, confidence interval, standard error, bootstrap distribution.
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)
    
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        'statistic': stat_func(data),
        'mean': np.mean(bootstrap_stats),
        'std_error': np.std(bootstrap_stats),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'distribution': bootstrap_stats
    }


def bootstrap_sharpe_comparison(returns_a, returns_b, n_bootstrap=10000,
                                 random_state=42):
    """Bootstrap test comparing Sharpe ratios of two strategies.
    
    Parameters
    ----------
    returns_a : pd.Series
        Returns of strategy A.
    returns_b : pd.Series
        Returns of strategy B.
    n_bootstrap : int
        Number of bootstrap iterations.
    random_state : int
        Random seed.
    
    Returns
    -------
    dict
        Sharpe ratio difference, p-value, probability A > B.
    """
    rng = np.random.RandomState(random_state)
    data_a = np.asarray(returns_a)
    data_b = np.asarray(returns_b)
    n = min(len(data_a), len(data_b))
    data_a = data_a[:n]
    data_b = data_b[:n]
    
    def sharpe(r):
        std_val = r.std()
        if std_val == 0 or np.isnan(std_val):
            return 0.0
        return np.sqrt(12) * r.mean() / std_val
    
    observed_diff = sharpe(data_a) - sharpe(data_b)
    
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = sharpe(data_a[idx]) - sharpe(data_b[idx])
    
    # Hypothesis test: H0: Sharpe(A) <= Sharpe(B)
    p_value = (diffs <= 0).mean()
    prob_a_wins = (diffs > 0).mean()
    
    return {
        'sharpe_a': sharpe(data_a),
        'sharpe_b': sharpe(data_b),
        'observed_diff': observed_diff,
        'p_value': p_value,
        'prob_a_wins': prob_a_wins,
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5),
        'n_bootstrap': n_bootstrap
    }


def diebold_mariano_test(errors_a, errors_b, h=1):
    """Diebold-Mariano test for equal predictive accuracy.
    
    Tests H0: E[d_t] = 0 where d_t = e_a_t^2 - e_b_t^2
    
    Parameters
    ----------
    errors_a : np.ndarray
        Forecast errors from model A.
    errors_b : np.ndarray
        Forecast errors from model B.
    h : int
        Forecast horizon.
    
    Returns
    -------
    dict
        DM statistic, p-value.
    """
    d = np.asarray(errors_a) ** 2 - np.asarray(errors_b) ** 2
    n = len(d)
    
    d_mean = d.mean()
    
    # Compute autocovariance-consistent variance estimate (Newey-West type for lag h-1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = gamma_0
    for k in range(1, h):
        if len(d[k:]) > 1:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += 2 * gamma_k
    
    variance = gamma_sum / n
    
    if variance <= 0 or np.isnan(variance):
        return {'dm_statistic': np.nan, 'p_value': np.nan, 'mean_diff': d_mean, 'model_a_better': False}
    
    dm_stat = d_mean / np.sqrt(variance)
    p_value = 2 * scipy_stats.norm.sf(abs(dm_stat))
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_diff': d_mean,
        'model_a_better': dm_stat < 0
    }


def regime_persistence_test(regimes, n_simulations=10000, random_state=42):
    """Test whether regime transitions are significantly different from random.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels.
    n_simulations : int
        Number of random permutations.
    random_state : int
        Random seed.
    
    Returns
    -------
    dict
        Observed persistence, simulated persistence, p-value.
    """
    rng = np.random.RandomState(random_state)
    regimes_arr = np.asarray(regimes)
    
    # Count transitions (regime changes)
    observed_transitions = np.sum(regimes_arr[1:] != regimes_arr[:-1])
    observed_persistence = 1 - observed_transitions / (len(regimes_arr) - 1)
    
    # Simulate under null (random permutation)
    sim_persistence = np.zeros(n_simulations)
    for i in range(n_simulations):
        shuffled = rng.permutation(regimes_arr)
        transitions = np.sum(shuffled[1:] != shuffled[:-1])
        sim_persistence[i] = 1 - transitions / (len(shuffled) - 1)
    
    # H0: observed persistence is less than or equal to random
    p_value = (sim_persistence >= observed_persistence).mean()
    
    return {
        'observed_persistence': observed_persistence,
        'expected_persistence': sim_persistence.mean(),
        'persistence_ratio': observed_persistence / sim_persistence.mean() if sim_persistence.mean() != 0 else np.nan,
        'p_value': p_value
    }


def bootstrap_outperformance(returns_strategy, returns_benchmark,
                              n_bootstrap=10000, random_state=42):
    """Bootstrap probability that strategy beats benchmark.
    
    Parameters
    ----------
    returns_strategy : pd.Series
        Strategy returns.
    returns_benchmark : pd.Series
        Benchmark returns.
    n_bootstrap : int
        Number of bootstrap iterations.
    random_state : int
        Random seed.
    
    Returns
    -------
    dict
        Probability of outperformance, wealth ratio CI.
    """
    rng = np.random.RandomState(random_state)
    strat = np.asarray(returns_strategy)
    bench = np.asarray(returns_benchmark)
    n = min(len(strat), len(bench))
    strat = strat[:n]
    bench = bench[:n]
    
    wealth_ratios = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        w_strat = np.prod(1 + strat[idx])
        w_bench = np.prod(1 + bench[idx])
        wealth_ratios[i] = w_strat / w_bench if w_bench > 0 else 0
    
    prob_outperform = (wealth_ratios > 1.0).mean()
    
    return {
        'prob_outperform': prob_outperform,
        'median_wealth_ratio': np.median(wealth_ratios),
        'ci_lower': np.percentile(wealth_ratios, 2.5),
        'ci_upper': np.percentile(wealth_ratios, 97.5),
        'n_bootstrap': n_bootstrap
    }
