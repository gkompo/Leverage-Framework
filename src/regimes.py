"""Volatility regime detection using Gaussian Mixture Models.

Identifies 5 market regimes based on volatility features:
- Low Volatility
- Normal Market
- High Volatility
- Crisis
- Recovery

Regimes are identified by GMM clustering and then interpreted
based on their statistical characteristics.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('regime_leverage')

REGIME_NAMES = {
    0: 'Low Volatility',
    1: 'Normal Market',
    2: 'High Volatility',
    3: 'Crisis',
    4: 'Recovery'
}


def prepare_regime_features(vol_features):
    """Prepare features for regime detection.
    
    Parameters
    ----------
    vol_features : pd.DataFrame
        DataFrame with volatility features.
    
    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix.
    np.ndarray
        Scaled feature array.
    StandardScaler
        Fitted scaler for inverse transform.
    """
    # Use key features: volatility, volatility change, drawdown
    feature_cols = ['vol_12m', 'vol_change', 'drawdown']
    features = vol_features[feature_cols].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    
    logger.info(f'Prepared {len(features)} observations with '
                f'{len(feature_cols)} features for regime detection')
    
    return features, X_scaled, scaler


def fit_regime_model(X_scaled, n_regimes=5, random_state=42, n_init=10):
    """Fit Gaussian Mixture Model for regime detection.
    
    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_regimes : int
        Number of regimes (default 5).
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of initializations.
    
    Returns
    -------
    GaussianMixture
        Fitted GMM model.
    np.ndarray
        Regime labels (0 to n_regimes-1).
    np.ndarray
        Regime probabilities.
    """
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=random_state,
        n_init=n_init,
        max_iter=500
    )
    
    labels = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    logger.info(f'Fitted GMM with {n_regimes} regimes. '
                f'BIC: {gmm.bic(X_scaled):.2f}, '
                f'AIC: {gmm.aic(X_scaled):.2f}')
    
    return gmm, labels, probabilities


def interpret_regimes(features, labels, returns):
    """Interpret and relabel regimes based on their statistics.
    
    Regime assignment logic:
    - Crisis: highest average volatility
    - Low Volatility: lowest average volatility
    - Recovery: negative drawdown + decreasing volatility + positive returns
    - High Volatility: above-median volatility (remaining)
    - Normal Market: remaining
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    labels : np.ndarray
        Raw GMM labels.
    returns : pd.Series
        Monthly returns aligned with features.
    
    Returns
    -------
    pd.Series
        Interpreted regime labels.
    dict
        Mapping from raw labels to regime names.
    pd.DataFrame
        Regime statistics.
    """
    aligned_returns = returns.reindex(features.index)
    
    # Compute statistics per raw cluster
    stats = pd.DataFrame({
        'avg_vol': features.groupby(labels)['vol_12m'].mean(),
        'avg_return': aligned_returns.groupby(labels).mean(),
        'avg_drawdown': features.groupby(labels)['drawdown'].mean(),
        'avg_vol_change': features.groupby(labels)['vol_change'].mean(),
        'count': pd.Series(labels).value_counts().sort_index()
    })
    
    # Sort clusters by average volatility
    sorted_by_vol = stats['avg_vol'].sort_values()
    cluster_ids = sorted_by_vol.index.tolist()
    
    # Assign regime names based on characteristics
    mapping = {}
    assigned = set()
    
    # Crisis: highest volatility
    crisis_id = cluster_ids[-1]
    mapping[crisis_id] = 'Crisis'
    assigned.add(crisis_id)
    
    # Low Volatility: lowest volatility
    low_vol_id = cluster_ids[0]
    mapping[low_vol_id] = 'Low Volatility'
    assigned.add(low_vol_id)
    
    # Recovery: look for cluster with declining vol and positive returns
    remaining = [c for c in cluster_ids if c not in assigned]
    recovery_scores = {}
    for c in remaining:
        vol_decline = -stats.loc[c, 'avg_vol_change']  # positive = declining vol
        positive_ret = max(0, stats.loc[c, 'avg_return'])
        dd = abs(stats.loc[c, 'avg_drawdown'])  # some drawdown still present
        recovery_scores[c] = vol_decline * 0.4 + positive_ret * 0.4 + dd * 0.2
    
    recovery_id = max(recovery_scores, key=recovery_scores.get)
    mapping[recovery_id] = 'Recovery'
    assigned.add(recovery_id)
    
    # Of the remaining two: higher vol = High Volatility, lower = Normal
    remaining = [c for c in cluster_ids if c not in assigned]
    remaining_vols = {c: stats.loc[c, 'avg_vol'] for c in remaining}
    sorted_remaining = sorted(remaining_vols, key=remaining_vols.get)
    
    mapping[sorted_remaining[0]] = 'Normal Market'
    if len(sorted_remaining) > 1:
        mapping[sorted_remaining[1]] = 'High Volatility'
    else:
        # Fallback if less than 5 clusters for some reason
        for c in cluster_ids:
            if c not in mapping:
                mapping[c] = 'High Volatility'
    
    # Apply mapping
    regime_series = pd.Series(labels, index=features.index).map(mapping)
    
    # Create regime statistics
    regime_stats = pd.DataFrame({
        'avg_volatility': features.groupby(regime_series.values)['vol_12m'].mean(),
        'avg_return_monthly': aligned_returns.groupby(regime_series.values).mean(),
        'avg_drawdown': features.groupby(regime_series.values)['drawdown'].mean(),
        'count': regime_series.value_counts(),
        'pct_of_time': regime_series.value_counts(normalize=True)
    })
    
    # Compute average duration of each regime in months
    durations = []
    if len(regime_series) > 0:
        current_regime = regime_series.iloc[0]
        current_duration = 1
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] == current_regime:
                current_duration += 1
            else:
                durations.append({'regime': current_regime, 'duration': current_duration})
                current_regime = regime_series.iloc[i]
                current_duration = 1
        durations.append({'regime': current_regime, 'duration': current_duration})
        dur_df = pd.DataFrame(durations)
        avg_dur = dur_df.groupby('regime')['duration'].mean()
        regime_stats['avg_duration_months'] = avg_dur
    else:
        regime_stats['avg_duration_months'] = np.nan
    
    logger.info('Regime interpretation:')
    for raw, name in mapping.items():
        logger.info(f'  Cluster {raw} -> {name} '
                    f'(avg vol: {stats.loc[raw, "avg_vol"]:.4f})')
    
    return regime_series, mapping, regime_stats


def detect_regimes(returns, prices, vol_features=None, n_regimes=5):
    """Full regime detection pipeline.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly log returns.
    prices : pd.Series
        Monthly prices.
    vol_features : pd.DataFrame, optional
        Pre-computed volatility features.
    n_regimes : int
        Number of regimes.
    
    Returns
    -------
    pd.Series
        Regime labels.
    pd.DataFrame
        Regime statistics.
    GaussianMixture
        Fitted model.
    StandardScaler
        Fitted scaler.
    """
    from .volatility import compute_volatility_features
    
    if vol_features is None:
        vol_features = compute_volatility_features(returns, prices)
    
    features, X_scaled, scaler = prepare_regime_features(vol_features)
    gmm, labels, probabilities = fit_regime_model(X_scaled, n_regimes)
    regimes, mapping, regime_stats = interpret_regimes(features, labels, returns)
    
    return regimes, regime_stats, gmm, scaler
