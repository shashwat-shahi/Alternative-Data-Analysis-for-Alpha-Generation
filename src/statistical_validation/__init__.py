"""
Statistical Validation Framework for Alternative Data Signals
============================================================

This module implements statistical validation methods and regime detection
for alternative data signals to ensure robust alpha generation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.regime_switching import MarkovRegression
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class AlternativeDataValidator:
    """
    Statistical validation framework for alternative data signals including
    significance testing, correlation analysis, and predictive power assessment.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the validator.
        
        Args:
            confidence_level: Statistical confidence level for tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.scaler = StandardScaler()
        
    def test_signal_significance(self, signal: pd.Series, returns: pd.Series,
                               lag: int = 1) -> Dict[str, float]:
        """
        Test statistical significance of alternative data signal vs returns.
        
        Args:
            signal: Alternative data signal time series
            returns: Asset returns time series  
            lag: Number of periods to lag signal (for predictive testing)
            
        Returns:
            Dictionary with statistical test results
        """
        try:
            # Align data and apply lag
            if lag > 0:
                signal_lagged = signal.shift(lag).dropna()
                returns_aligned = returns.loc[signal_lagged.index]
            else:
                common_index = signal.index.intersection(returns.index)
                signal_lagged = signal.loc[common_index]
                returns_aligned = returns.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = ~(np.isnan(signal_lagged) | np.isnan(returns_aligned))
            signal_clean = signal_lagged[valid_mask]
            returns_clean = returns_aligned[valid_mask]
            
            if len(signal_clean) < 30:
                return {'error': 'Insufficient data for statistical testing'}
            
            # Correlation analysis
            correlation, corr_pvalue = stats.pearsonr(signal_clean, returns_clean)
            spearman_corr, spearman_pvalue = stats.spearmanr(signal_clean, returns_clean)
            
            # Linear regression
            X = sm.add_constant(signal_clean)
            model = sm.OLS(returns_clean, X).fit()
            
            # Information Coefficient (IC) - rank correlation
            signal_ranks = signal_clean.rank()
            returns_ranks = returns_clean.rank()
            ic, ic_pvalue = stats.pearsonr(signal_ranks, returns_ranks)
            
            # Mutual information
            from sklearn.feature_selection import mutual_info_regression
            mi_score = mutual_info_regression(
                signal_clean.values.reshape(-1, 1), 
                returns_clean.values
            )[0]
            
            # Hit rate analysis (directional accuracy)
            signal_direction = np.sign(signal_clean)
            returns_direction = np.sign(returns_clean)
            hit_rate = np.mean(signal_direction == returns_direction)
            
            # Binomial test for hit rate significance
            n_obs = len(signal_direction)
            hit_rate_pvalue = stats.binom_test(
                int(hit_rate * n_obs), n_obs, 0.5, alternative='two-sided'
            )
            
            return {
                'pearson_correlation': correlation,
                'pearson_pvalue': corr_pvalue,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_pvalue,
                'regression_beta': model.params.iloc[1],
                'regression_tstat': model.tvalues.iloc[1],
                'regression_pvalue': model.pvalues.iloc[1],
                'regression_rsquared': model.rsquared,
                'information_coefficient': ic,
                'ic_pvalue': ic_pvalue,
                'mutual_information': mi_score,
                'hit_rate': hit_rate,
                'hit_rate_pvalue': hit_rate_pvalue,
                'n_observations': n_obs,
                'is_significant': corr_pvalue < self.alpha
            }
            
        except Exception as e:
            logger.error(f"Error in significance testing: {e}")
            return {'error': str(e)}
    
    def cross_sectional_validation(self, signals_df: pd.DataFrame, 
                                 returns_df: pd.DataFrame,
                                 min_observations: int = 50) -> Dict[str, float]:
        """
        Perform cross-sectional validation across multiple assets.
        
        Args:
            signals_df: DataFrame with signals for multiple assets (columns)
            returns_df: DataFrame with returns for multiple assets (columns)
            min_observations: Minimum number of observations required
            
        Returns:
            Cross-sectional validation metrics
        """
        try:
            # Align data
            common_index = signals_df.index.intersection(returns_df.index)
            common_columns = signals_df.columns.intersection(returns_df.columns)
            
            if len(common_index) < min_observations or len(common_columns) < 2:
                return {'error': 'Insufficient data for cross-sectional validation'}
            
            signals_aligned = signals_df.loc[common_index, common_columns]
            returns_aligned = returns_df.loc[common_index, common_columns]
            
            # Calculate cross-sectional metrics for each date
            daily_ics = []
            daily_correlations = []
            
            for date in common_index:
                date_signals = signals_aligned.loc[date].dropna()
                date_returns = returns_aligned.loc[date].dropna()
                
                # Get intersection of non-NaN assets
                valid_assets = date_signals.index.intersection(date_returns.index)
                
                if len(valid_assets) >= 5:  # Minimum assets for meaningful correlation
                    date_signals_clean = date_signals.loc[valid_assets]
                    date_returns_clean = date_returns.loc[valid_assets]
                    
                    # Information Coefficient (rank correlation)
                    signal_ranks = date_signals_clean.rank()
                    return_ranks = date_returns_clean.rank()
                    
                    if signal_ranks.std() > 0 and return_ranks.std() > 0:
                        ic, _ = stats.pearsonr(signal_ranks, return_ranks)
                        corr, _ = stats.pearsonr(date_signals_clean, date_returns_clean)
                        
                        daily_ics.append(ic)
                        daily_correlations.append(corr)
            
            if not daily_ics:
                return {'error': 'No valid cross-sectional data found'}
            
            # Calculate summary statistics
            mean_ic = np.mean(daily_ics)
            ic_std = np.std(daily_ics)
            ic_ir = mean_ic / ic_std if ic_std > 0 else 0  # Information Ratio
            
            # IC significance test
            ic_tstat = mean_ic / (ic_std / np.sqrt(len(daily_ics))) if ic_std > 0 else 0
            ic_pvalue = 2 * (1 - stats.t.cdf(abs(ic_tstat), len(daily_ics) - 1))
            
            return {
                'mean_information_coefficient': mean_ic,
                'ic_information_ratio': ic_ir,
                'ic_standard_deviation': ic_std,
                'ic_tstatistic': ic_tstat,
                'ic_pvalue': ic_pvalue,
                'mean_correlation': np.mean(daily_correlations),
                'correlation_std': np.std(daily_correlations),
                'ic_hit_rate': np.mean([ic > 0 for ic in daily_ics]),
                'n_periods': len(daily_ics),
                'n_assets_avg': len(common_columns),
                'is_significant': ic_pvalue < self.alpha
            }
            
        except Exception as e:
            logger.error(f"Error in cross-sectional validation: {e}")
            return {'error': str(e)}
    
    def time_series_validation(self, signal: pd.Series, returns: pd.Series,
                             window_size: int = 252) -> Dict[str, float]:
        """
        Perform rolling time series validation of signal stability.
        
        Args:
            signal: Alternative data signal
            returns: Asset returns
            window_size: Rolling window size for validation
            
        Returns:
            Time series validation metrics
        """
        try:
            # Align data
            common_index = signal.index.intersection(returns.index)
            signal_aligned = signal.loc[common_index]
            returns_aligned = returns.loc[common_index]
            
            if len(common_index) < window_size * 2:
                return {'error': 'Insufficient data for time series validation'}
            
            # Rolling correlations
            rolling_corrs = []
            rolling_ics = []
            
            for i in range(window_size, len(common_index)):
                window_signal = signal_aligned.iloc[i-window_size:i]
                window_returns = returns_aligned.iloc[i-window_size:i]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(window_signal) | np.isnan(window_returns))
                if valid_mask.sum() > 30:  # Minimum observations
                    window_signal_clean = window_signal[valid_mask]
                    window_returns_clean = window_returns[valid_mask]
                    
                    # Correlation
                    if window_signal_clean.std() > 0 and window_returns_clean.std() > 0:
                        corr, _ = stats.pearsonr(window_signal_clean, window_returns_clean)
                        rolling_corrs.append(corr)
                        
                        # Information Coefficient
                        signal_ranks = window_signal_clean.rank()
                        return_ranks = window_returns_clean.rank()
                        ic, _ = stats.pearsonr(signal_ranks, return_ranks)
                        rolling_ics.append(ic)
            
            if not rolling_corrs:
                return {'error': 'No valid rolling windows found'}
            
            # Calculate stability metrics
            correlation_stability = 1 - np.std(rolling_corrs) / (np.abs(np.mean(rolling_corrs)) + 1e-6)
            ic_stability = 1 - np.std(rolling_ics) / (np.abs(np.mean(rolling_ics)) + 1e-6)
            
            # Trend analysis
            periods = np.arange(len(rolling_corrs))
            corr_trend_slope, _, corr_trend_r, corr_trend_p, _ = stats.linregress(periods, rolling_corrs)
            
            return {
                'mean_rolling_correlation': np.mean(rolling_corrs),
                'correlation_stability': correlation_stability,
                'correlation_std': np.std(rolling_corrs),
                'mean_rolling_ic': np.mean(rolling_ics),
                'ic_stability': ic_stability,
                'ic_std': np.std(rolling_ics),
                'correlation_trend_slope': corr_trend_slope,
                'correlation_trend_pvalue': corr_trend_p,
                'n_windows': len(rolling_corrs),
                'window_size': window_size,
                'is_stable': correlation_stability > 0.7 and ic_stability > 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in time series validation: {e}")
            return {'error': str(e)}


class RegimeDetector:
    """
    Market regime detection using Markov Regime Switching models and 
    Gaussian Mixture Models for alternative data validation.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
        """
        self.n_regimes = n_regimes
        self.regime_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        
    def detect_regimes_markov(self, returns: pd.Series, 
                             exog_vars: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect market regimes using Markov Regime Switching model.
        
        Args:
            returns: Asset returns time series
            exog_vars: Optional exogenous variables for regime switching
            
        Returns:
            Regime detection results
        """
        try:
            # Prepare data
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                return {'error': 'Insufficient data for regime detection'}
            
            # Fit Markov Regime Switching model
            if exog_vars is not None:
                exog_aligned = exog_vars.loc[returns_clean.index].dropna()
                common_index = returns_clean.index.intersection(exog_aligned.index)
                returns_model = returns_clean.loc[common_index]
                exog_model = exog_aligned.loc[common_index]
                
                self.regime_model = MarkovRegression(
                    returns_model, 
                    k_regimes=self.n_regimes,
                    exog=exog_model,
                    switching_variance=True
                )
            else:
                self.regime_model = MarkovRegression(
                    returns_clean,
                    k_regimes=self.n_regimes, 
                    switching_variance=True
                )
            
            # Fit model
            regime_results = self.regime_model.fit()
            
            # Extract regime probabilities and states
            regime_probs = regime_results.smoothed_marginal_probabilities
            regime_states = regime_probs.idxmax(axis=1)
            
            # Calculate regime characteristics
            regime_stats = {}
            for regime in range(self.n_regimes):
                regime_mask = regime_states == regime
                regime_returns = returns_clean[regime_mask]
                
                regime_stats[f'regime_{regime}'] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'frequency': regime_mask.mean(),
                    'avg_duration': self._calculate_avg_duration(regime_mask)
                }
            
            return {
                'regime_probabilities': regime_probs,
                'regime_states': regime_states,
                'regime_statistics': regime_stats,
                'model_aic': regime_results.aic,
                'model_bic': regime_results.bic,
                'log_likelihood': regime_results.llf,
                'transition_matrix': regime_results.regime_transition,
                'model_summary': str(regime_results.summary())
            }
            
        except Exception as e:
            logger.error(f"Error in Markov regime detection: {e}")
            return {'error': str(e)}
    
    def detect_regimes_gmm(self, features: pd.DataFrame) -> Dict:
        """
        Detect regimes using Gaussian Mixture Model on multiple features.
        
        Args:
            features: DataFrame with market/economic features for regime detection
            
        Returns:
            GMM regime detection results
        """
        try:
            # Prepare data
            features_clean = features.dropna()
            
            if len(features_clean) < 50:
                return {'error': 'Insufficient data for GMM regime detection'}
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_clean)
            
            # Fit Gaussian Mixture Model
            self.gmm_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42
            )
            
            regime_labels = self.gmm_model.fit_predict(features_scaled)
            regime_probs = self.gmm_model.predict_proba(features_scaled)
            
            # Create results DataFrame
            regime_df = pd.DataFrame(
                regime_probs, 
                index=features_clean.index,
                columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
            )
            regime_df['regime_state'] = regime_labels
            
            # Calculate regime characteristics
            regime_stats = {}
            for regime in range(self.n_regimes):
                regime_mask = regime_labels == regime
                regime_features = features_clean[regime_mask]
                
                regime_stats[f'regime_{regime}'] = {
                    'frequency': regime_mask.mean(),
                    'avg_duration': self._calculate_avg_duration(regime_mask),
                    'feature_means': regime_features.mean().to_dict(),
                    'feature_stds': regime_features.std().to_dict()
                }
            
            return {
                'regime_probabilities': regime_df,
                'regime_labels': regime_labels,
                'regime_statistics': regime_stats,
                'model_aic': self.gmm_model.aic(features_scaled),
                'model_bic': self.gmm_model.bic(features_scaled),
                'log_likelihood': self.gmm_model.score(features_scaled),
                'model_converged': self.gmm_model.converged_
            }
            
        except Exception as e:
            logger.error(f"Error in GMM regime detection: {e}")
            return {'error': str(e)}
    
    def regime_conditional_validation(self, signal: pd.Series, returns: pd.Series,
                                   regime_states: pd.Series) -> Dict[str, Dict]:
        """
        Validate alternative data signal performance across different market regimes.
        
        Args:
            signal: Alternative data signal
            returns: Asset returns
            regime_states: Detected regime states
            
        Returns:
            Regime-conditional validation results
        """
        try:
            # Align all series
            common_index = signal.index.intersection(returns.index).intersection(regime_states.index)
            
            if len(common_index) < 50:
                return {'error': 'Insufficient aligned data for regime validation'}
            
            signal_aligned = signal.loc[common_index]
            returns_aligned = returns.loc[common_index]
            regimes_aligned = regime_states.loc[common_index]
            
            # Validate signal in each regime
            regime_validation = {}
            
            for regime in regimes_aligned.unique():
                regime_mask = regimes_aligned == regime
                regime_signal = signal_aligned[regime_mask]
                regime_returns = returns_aligned[regime_mask]
                
                if len(regime_signal) > 10:  # Minimum observations
                    # Calculate correlation
                    valid_mask = ~(np.isnan(regime_signal) | np.isnan(regime_returns))
                    if valid_mask.sum() > 5:
                        signal_clean = regime_signal[valid_mask]
                        returns_clean = regime_returns[valid_mask]
                        
                        if signal_clean.std() > 0 and returns_clean.std() > 0:
                            correlation, corr_pvalue = stats.pearsonr(signal_clean, returns_clean)
                            
                            # Information Coefficient
                            signal_ranks = signal_clean.rank()
                            return_ranks = returns_clean.rank()
                            ic, ic_pvalue = stats.pearsonr(signal_ranks, return_ranks)
                            
                            # Hit rate
                            signal_direction = np.sign(signal_clean)
                            return_direction = np.sign(returns_clean)
                            hit_rate = np.mean(signal_direction == return_direction)
                            
                            regime_validation[f'regime_{regime}'] = {
                                'correlation': correlation,
                                'correlation_pvalue': corr_pvalue,
                                'information_coefficient': ic,
                                'ic_pvalue': ic_pvalue,
                                'hit_rate': hit_rate,
                                'n_observations': len(signal_clean),
                                'signal_mean': signal_clean.mean(),
                                'signal_std': signal_clean.std(),
                                'returns_mean': returns_clean.mean(),
                                'returns_std': returns_clean.std()
                            }
            
            # Calculate cross-regime stability
            correlations = [v['correlation'] for v in regime_validation.values() if 'correlation' in v]
            ics = [v['information_coefficient'] for v in regime_validation.values() if 'information_coefficient' in v]
            
            stability_metrics = {
                'correlation_stability': 1 - np.std(correlations) / (np.abs(np.mean(correlations)) + 1e-6) if correlations else 0,
                'ic_stability': 1 - np.std(ics) / (np.abs(np.mean(ics)) + 1e-6) if ics else 0,
                'cross_regime_correlation_range': max(correlations) - min(correlations) if correlations else 0,
                'cross_regime_ic_range': max(ics) - min(ics) if ics else 0
            }
            
            return {
                'regime_validation': regime_validation,
                'stability_metrics': stability_metrics,
                'n_regimes_validated': len(regime_validation)
            }
            
        except Exception as e:
            logger.error(f"Error in regime conditional validation: {e}")
            return {'error': str(e)}
    
    def _calculate_avg_duration(self, regime_mask: pd.Series) -> float:
        """Calculate average duration of regime periods."""
        if not isinstance(regime_mask, pd.Series):
            regime_mask = pd.Series(regime_mask)
            
        # Find regime change points
        regime_changes = regime_mask != regime_mask.shift(1)
        regime_periods = regime_changes.cumsum()
        
        # Calculate duration of each regime period
        durations = []
        for period in regime_periods[regime_mask].unique():
            period_length = (regime_periods == period).sum()
            durations.append(period_length)
        
        return np.mean(durations) if durations else 0


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Sample signal and returns
    signal = pd.Series(np.random.randn(len(dates)), index=dates)
    returns = pd.Series(0.1 * signal.shift(1) + 0.02 * np.random.randn(len(dates)), index=dates)
    
    # Test validation framework
    validator = AlternativeDataValidator()
    significance_results = validator.test_signal_significance(signal, returns)
    print("Signal Significance Results:")
    print(significance_results)
    
    # Test regime detection
    regime_detector = RegimeDetector()
    regime_results = regime_detector.detect_regimes_markov(returns)
    print("\nRegime Detection Results:")
    print(f"Number of regimes detected: {len(regime_results.get('regime_statistics', {}))}")