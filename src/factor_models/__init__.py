"""
Factor Models for ESG and Satellite Data
========================================

This module implements factor models that incorporate ESG scores and satellite data
for long-term alpha generation in quantitative investment strategies.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESGFactorModel:
    """
    ESG (Environmental, Social, Governance) factor model for alpha generation.
    Incorporates ESG scores and ratings into quantitative investment models.
    """
    
    def __init__(self, standardize: bool = True):
        """
        Initialize ESG factor model.
        
        Args:
            standardize: Whether to standardize ESG scores
        """
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.factor_loadings = None
        self.esg_factors = None
        self.model = None
        
    def create_esg_factors(self, esg_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ESG factor exposures from raw ESG data.
        
        Args:
            esg_data: DataFrame with ESG scores and metrics
                     Expected columns: E_score, S_score, G_score, overall_score, etc.
        
        Returns:
            DataFrame with derived ESG factors
        """
        try:
            esg_factors = esg_data.copy()
            
            # Basic ESG factors
            if all(col in esg_data.columns for col in ['E_score', 'S_score', 'G_score']):
                # ESG momentum (change over time)
                esg_factors['E_momentum'] = esg_data['E_score'].pct_change(periods=252)  # 1-year momentum
                esg_factors['S_momentum'] = esg_data['S_score'].pct_change(periods=252)
                esg_factors['G_momentum'] = esg_data['G_score'].pct_change(periods=252)
                
                # ESG quality (relative to median)
                esg_factors['E_quality'] = esg_data['E_score'] - esg_data['E_score'].rolling(252).median()
                esg_factors['S_quality'] = esg_data['S_score'] - esg_data['S_score'].rolling(252).median()
                esg_factors['G_quality'] = esg_data['G_score'] - esg_data['G_score'].rolling(252).median()
                
                # ESG consistency (inverse of volatility)
                esg_factors['E_consistency'] = 1 / (esg_data['E_score'].rolling(252).std() + 1e-6)
                esg_factors['S_consistency'] = 1 / (esg_data['S_score'].rolling(252).std() + 1e-6)
                esg_factors['G_consistency'] = 1 / (esg_data['G_score'].rolling(252).std() + 1e-6)
                
                # Composite factors
                esg_factors['esg_composite'] = (
                    esg_data['E_score'] * 0.4 + 
                    esg_data['S_score'] * 0.3 + 
                    esg_data['G_score'] * 0.3
                )
                
                esg_factors['esg_improvement'] = (
                    esg_factors['E_momentum'] * 0.4 +
                    esg_factors['S_momentum'] * 0.3 +
                    esg_factors['G_momentum'] * 0.3
                )
            
            # Additional ESG-derived factors
            if 'overall_score' in esg_data.columns:
                esg_factors['esg_percentile'] = esg_data['overall_score'].rolling(252).rank(pct=True)
                esg_factors['esg_zscore'] = (
                    esg_data['overall_score'] - esg_data['overall_score'].rolling(252).mean()
                ) / esg_data['overall_score'].rolling(252).std()
            
            # Industry-relative ESG factors (if industry data available)
            if 'industry' in esg_data.columns:
                for score_col in ['E_score', 'S_score', 'G_score', 'overall_score']:
                    if score_col in esg_data.columns:
                        esg_factors[f'{score_col}_industry_relative'] = (
                            esg_data.groupby('industry')[score_col].transform(
                                lambda x: x - x.median()
                            )
                        )
            
            # Remove infinite and NaN values
            esg_factors = esg_factors.replace([np.inf, -np.inf], np.nan)
            esg_factors = esg_factors.fillna(method='ffill').fillna(0)
            
            self.esg_factors = esg_factors
            
            logger.info(f"Created {len(esg_factors.columns)} ESG factors")
            return esg_factors
            
        except Exception as e:
            logger.error(f"Error creating ESG factors: {e}")
            raise
    
    def fit_factor_model(self, returns: pd.Series, esg_factors: pd.DataFrame,
                        method: str = 'ridge', alpha: float = 1.0) -> Dict:
        """
        Fit ESG factor model to predict returns.
        
        Args:
            returns: Asset returns time series
            esg_factors: ESG factor exposures
            method: Regression method ('ridge', 'lasso', 'elastic_net', 'ols')
            alpha: Regularization strength
            
        Returns:
            Model fitting results
        """
        try:
            # Align data
            common_index = returns.index.intersection(esg_factors.index)
            if len(common_index) < 100:
                raise ValueError("Insufficient aligned data for model fitting")
            
            returns_aligned = returns.loc[common_index]
            factors_aligned = esg_factors.loc[common_index]
            
            # Remove NaN values
            valid_mask = ~(returns_aligned.isna() | factors_aligned.isna().any(axis=1))
            returns_clean = returns_aligned[valid_mask]
            factors_clean = factors_aligned[valid_mask]
            
            if len(returns_clean) < 50:
                raise ValueError("Insufficient clean data for model fitting")
            
            # Standardize factors if required
            if self.standardize:
                factors_scaled = self.scaler.fit_transform(factors_clean)
                factors_scaled = pd.DataFrame(
                    factors_scaled, 
                    index=factors_clean.index, 
                    columns=factors_clean.columns
                )
            else:
                factors_scaled = factors_clean
            
            # Fit model based on method
            if method == 'ridge':
                self.model = Ridge(alpha=alpha, random_state=42)
            elif method == 'lasso':
                self.model = Lasso(alpha=alpha, random_state=42)
            elif method == 'elastic_net':
                self.model = ElasticNet(alpha=alpha, random_state=42)
            elif method == 'ols':
                # Use statsmodels for OLS with statistics
                X = sm.add_constant(factors_scaled)
                self.model = sm.OLS(returns_clean, X).fit()
                
                return {
                    'method': method,
                    'coefficients': self.model.params.to_dict(),
                    'pvalues': self.model.pvalues.to_dict(),
                    'tvalues': self.model.tvalues.to_dict(),
                    'rsquared': self.model.rsquared,
                    'rsquared_adj': self.model.rsquared_adj,
                    'fstatistic': self.model.fvalue,
                    'fstatistic_pvalue': self.model.f_pvalue,
                    'aic': self.model.aic,
                    'bic': self.model.bic,
                    'n_observations': len(returns_clean)
                }
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Fit sklearn model
            self.model.fit(factors_scaled, returns_clean)
            
            # Make predictions for model evaluation
            predictions = self.model.predict(factors_scaled)
            
            # Calculate performance metrics
            mse = mean_squared_error(returns_clean, predictions)
            r2 = r2_score(returns_clean, predictions)
            
            # Store factor loadings
            self.factor_loadings = pd.Series(
                self.model.coef_, 
                index=factors_scaled.columns
            )
            
            return {
                'method': method,
                'factor_loadings': self.factor_loadings.to_dict(),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r_squared': r2,
                'n_factors': len(factors_scaled.columns),
                'n_observations': len(returns_clean),
                'alpha': alpha
            }
            
        except Exception as e:
            logger.error(f"Error fitting ESG factor model: {e}")
            raise
    
    def generate_esg_scores(self, esg_factors: pd.DataFrame) -> pd.Series:
        """
        Generate composite ESG-based alpha scores.
        
        Args:
            esg_factors: ESG factor exposures
            
        Returns:
            ESG-based alpha scores
        """
        try:
            if self.factor_loadings is None:
                raise ValueError("Model must be fitted before generating scores")
            
            # Align factors with model factors
            model_factors = esg_factors[self.factor_loadings.index]
            
            # Standardize if required
            if self.standardize:
                factors_scaled = self.scaler.transform(model_factors.fillna(0))
                factors_scaled = pd.DataFrame(
                    factors_scaled,
                    index=model_factors.index,
                    columns=model_factors.columns
                )
            else:
                factors_scaled = model_factors.fillna(0)
            
            # Generate scores
            if hasattr(self.model, 'predict'):
                scores = self.model.predict(factors_scaled)
            else:  # statsmodels OLS
                X = sm.add_constant(factors_scaled)
                scores = self.model.predict(X)
            
            return pd.Series(scores, index=factors_scaled.index, name='esg_alpha_score')
            
        except Exception as e:
            logger.error(f"Error generating ESG scores: {e}")
            raise


class SatelliteDataModel:
    """
    Satellite data factor model for alternative alpha generation.
    Incorporates satellite imagery and geospatial data for investment insights.
    """
    
    def __init__(self, standardize: bool = True):
        """
        Initialize satellite data model.
        
        Args:
            standardize: Whether to standardize satellite data features
        """
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca_model = None
        self.factor_model = None
        self.satellite_factors = None
        
    def create_satellite_factors(self, satellite_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create factor exposures from satellite data.
        
        Args:
            satellite_data: DataFrame with satellite-derived metrics
                          Expected columns: economic_activity, vegetation_index, 
                          nighttime_lights, construction_activity, etc.
        
        Returns:
            DataFrame with satellite-based factors
        """
        try:
            satellite_factors = satellite_data.copy()
            
            # Economic activity factors
            if 'nighttime_lights' in satellite_data.columns:
                # Economic activity momentum
                satellite_factors['lights_momentum'] = satellite_data['nighttime_lights'].pct_change(periods=90)
                satellite_factors['lights_trend'] = satellite_data['nighttime_lights'].rolling(180).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0
                )
                
                # Economic activity seasonal adjustment
                satellite_factors['lights_seasonal'] = (
                    satellite_data['nighttime_lights'] - 
                    satellite_data['nighttime_lights'].rolling(365).mean()
                )
            
            # Agricultural and environmental factors
            if 'vegetation_index' in satellite_data.columns:
                # Vegetation health and trends
                satellite_factors['vegetation_momentum'] = satellite_data['vegetation_index'].pct_change(periods=30)
                satellite_factors['vegetation_volatility'] = satellite_data['vegetation_index'].rolling(90).std()
                satellite_factors['vegetation_percentile'] = satellite_data['vegetation_index'].rolling(365).rank(pct=True)
            
            # Construction and infrastructure activity
            if 'construction_activity' in satellite_data.columns:
                satellite_factors['construction_momentum'] = satellite_data['construction_activity'].pct_change(periods=60)
                satellite_factors['construction_acceleration'] = satellite_factors['construction_momentum'].pct_change(periods=30)
            
            # Traffic and logistics factors
            if 'traffic_density' in satellite_data.columns:
                satellite_factors['traffic_momentum'] = satellite_data['traffic_density'].pct_change(periods=7)
                satellite_factors['traffic_weekly_change'] = (
                    satellite_data['traffic_density'] - 
                    satellite_data['traffic_density'].shift(7)
                ) / satellite_data['traffic_density'].shift(7)
            
            # Port and shipping activity
            if 'port_activity' in satellite_data.columns:
                satellite_factors['port_momentum'] = satellite_data['port_activity'].pct_change(periods=30)
                satellite_factors['port_seasonal'] = (
                    satellite_data['port_activity'] - 
                    satellite_data['port_activity'].rolling(365).mean()
                )
            
            # Cross-factor interactions
            if all(col in satellite_factors.columns for col in ['lights_momentum', 'construction_momentum']):
                satellite_factors['economic_composite'] = (
                    satellite_factors['lights_momentum'] * 0.6 +
                    satellite_factors['construction_momentum'] * 0.4
                )
            
            # Regional relative factors (if region data available)
            if 'region' in satellite_data.columns:
                for base_col in ['nighttime_lights', 'vegetation_index', 'construction_activity']:
                    if base_col in satellite_data.columns:
                        satellite_factors[f'{base_col}_regional_relative'] = (
                            satellite_data.groupby('region')[base_col].transform(
                                lambda x: (x - x.median()) / (x.std() + 1e-6)
                            )
                        )
            
            # Clean data
            satellite_factors = satellite_factors.replace([np.inf, -np.inf], np.nan)
            satellite_factors = satellite_factors.fillna(method='ffill').fillna(0)
            
            self.satellite_factors = satellite_factors
            
            logger.info(f"Created {len(satellite_factors.columns)} satellite factors")
            return satellite_factors
            
        except Exception as e:
            logger.error(f"Error creating satellite factors: {e}")
            raise
    
    def apply_pca_reduction(self, satellite_factors: pd.DataFrame, 
                          n_components: Optional[int] = None,
                          variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction of satellite factors.
        
        Args:
            satellite_factors: Satellite factor DataFrame
            n_components: Number of PCA components (if None, use variance threshold)
            variance_threshold: Minimum variance to explain
            
        Returns:
            PCA-transformed factors
        """
        try:
            # Prepare data
            factors_clean = satellite_factors.fillna(0)
            
            if self.standardize:
                factors_scaled = self.scaler.fit_transform(factors_clean)
            else:
                factors_scaled = factors_clean.values
            
            # Determine number of components
            if n_components is None:
                # Use variance threshold
                pca_temp = PCA()
                pca_temp.fit(factors_scaled)
                cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumvar >= variance_threshold) + 1
            
            # Apply PCA
            self.pca_model = PCA(n_components=n_components, random_state=42)
            factors_pca = self.pca_model.fit_transform(factors_scaled)
            
            # Create DataFrame with PCA factors
            pca_df = pd.DataFrame(
                factors_pca,
                index=satellite_factors.index,
                columns=[f'satellite_pc_{i+1}' for i in range(n_components)]
            )
            
            logger.info(f"PCA reduced {satellite_factors.shape[1]} factors to {n_components} components")
            logger.info(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.3f}")
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error in PCA reduction: {e}")
            raise
    
    def fit_satellite_model(self, returns: pd.Series, satellite_factors: pd.DataFrame,
                          use_pca: bool = True, method: str = 'random_forest') -> Dict:
        """
        Fit satellite data model to predict returns.
        
        Args:
            returns: Asset returns time series
            satellite_factors: Satellite factor exposures
            use_pca: Whether to apply PCA reduction
            method: Model method ('random_forest', 'ridge', 'elastic_net')
            
        Returns:
            Model fitting results
        """
        try:
            # Align data
            common_index = returns.index.intersection(satellite_factors.index)
            if len(common_index) < 100:
                raise ValueError("Insufficient aligned data for model fitting")
            
            returns_aligned = returns.loc[common_index]
            factors_aligned = satellite_factors.loc[common_index]
            
            # Apply PCA if requested
            if use_pca:
                factors_processed = self.apply_pca_reduction(factors_aligned)
            else:
                factors_processed = factors_aligned
            
            # Remove NaN values
            valid_mask = ~(returns_aligned.isna() | factors_processed.isna().any(axis=1))
            returns_clean = returns_aligned[valid_mask]
            factors_clean = factors_processed[valid_mask]
            
            if len(returns_clean) < 50:
                raise ValueError("Insufficient clean data for model fitting")
            
            # Fit model
            if method == 'random_forest':
                self.factor_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif method == 'ridge':
                self.factor_model = Ridge(alpha=1.0, random_state=42)
            elif method == 'elastic_net':
                self.factor_model = ElasticNet(alpha=1.0, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Fit model
            self.factor_model.fit(factors_clean, returns_clean)
            
            # Make predictions
            predictions = self.factor_model.predict(factors_clean)
            
            # Calculate metrics
            mse = mean_squared_error(returns_clean, predictions)
            r2 = r2_score(returns_clean, predictions)
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(self.factor_model, 'feature_importances_'):
                feature_importance = pd.Series(
                    self.factor_model.feature_importances_,
                    index=factors_clean.columns
                ).sort_values(ascending=False)
            
            return {
                'method': method,
                'use_pca': use_pca,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r_squared': r2,
                'feature_importance': feature_importance.to_dict() if feature_importance is not None else None,
                'n_factors': len(factors_clean.columns),
                'n_observations': len(returns_clean),
                'pca_explained_variance': self.pca_model.explained_variance_ratio_.sum() if use_pca else None
            }
            
        except Exception as e:
            logger.error(f"Error fitting satellite model: {e}")
            raise
    
    def generate_satellite_scores(self, satellite_factors: pd.DataFrame) -> pd.Series:
        """
        Generate satellite data-based alpha scores.
        
        Args:
            satellite_factors: Satellite factor exposures
            
        Returns:
            Satellite-based alpha scores
        """
        try:
            if self.factor_model is None:
                raise ValueError("Model must be fitted before generating scores")
            
            # Process factors (apply PCA if used during fitting)
            if self.pca_model is not None:
                factors_clean = satellite_factors.fillna(0)
                if self.standardize:
                    factors_scaled = self.scaler.transform(factors_clean)
                else:
                    factors_scaled = factors_clean.values
                factors_processed = self.pca_model.transform(factors_scaled)
                factors_processed = pd.DataFrame(
                    factors_processed,
                    index=satellite_factors.index,
                    columns=[f'satellite_pc_{i+1}' for i in range(factors_processed.shape[1])]
                )
            else:
                factors_processed = satellite_factors.fillna(0)
            
            # Generate scores
            scores = self.factor_model.predict(factors_processed)
            
            return pd.Series(scores, index=factors_processed.index, name='satellite_alpha_score')
            
        except Exception as e:
            logger.error(f"Error generating satellite scores: {e}")
            raise


class MultiFactorModel:
    """
    Combined multi-factor model incorporating ESG, satellite data, and traditional factors.
    """
    
    def __init__(self, esg_model: ESGFactorModel, satellite_model: SatelliteDataModel):
        """
        Initialize multi-factor model.
        
        Args:
            esg_model: Fitted ESG factor model
            satellite_model: Fitted satellite data model
        """
        self.esg_model = esg_model
        self.satellite_model = satellite_model
        self.combined_model = None
        self.factor_weights = None
        
    def combine_models(self, returns: pd.Series, esg_factors: pd.DataFrame,
                      satellite_factors: pd.DataFrame, traditional_factors: Optional[pd.DataFrame] = None,
                      method: str = 'ridge') -> Dict:
        """
        Combine ESG, satellite, and traditional factors into unified model.
        
        Args:
            returns: Asset returns
            esg_factors: ESG factor exposures
            satellite_factors: Satellite factor exposures
            traditional_factors: Optional traditional factors (e.g., Fama-French)
            method: Combination method
            
        Returns:
            Combined model results
        """
        try:
            # Generate individual alpha scores
            esg_scores = self.esg_model.generate_esg_scores(esg_factors)
            satellite_scores = self.satellite_model.generate_satellite_scores(satellite_factors)
            
            # Combine scores into factor matrix
            combined_factors = pd.DataFrame({
                'esg_alpha': esg_scores,
                'satellite_alpha': satellite_scores
            })
            
            # Add traditional factors if provided
            if traditional_factors is not None:
                common_index = combined_factors.index.intersection(traditional_factors.index)
                combined_factors = combined_factors.loc[common_index]
                traditional_aligned = traditional_factors.loc[common_index]
                combined_factors = pd.concat([combined_factors, traditional_aligned], axis=1)
            
            # Align with returns
            common_index = returns.index.intersection(combined_factors.index)
            returns_aligned = returns.loc[common_index]
            factors_aligned = combined_factors.loc[common_index]
            
            # Clean data
            valid_mask = ~(returns_aligned.isna() | factors_aligned.isna().any(axis=1))
            returns_clean = returns_aligned[valid_mask]
            factors_clean = factors_aligned[valid_mask]
            
            # Fit combined model
            if method == 'ridge':
                self.combined_model = Ridge(alpha=1.0, random_state=42)
            else:
                raise ValueError(f"Method {method} not implemented for combined model")
            
            self.combined_model.fit(factors_clean, returns_clean)
            
            # Store factor weights
            self.factor_weights = pd.Series(
                self.combined_model.coef_,
                index=factors_clean.columns
            )
            
            # Model evaluation
            predictions = self.combined_model.predict(factors_clean)
            mse = mean_squared_error(returns_clean, predictions)
            r2 = r2_score(returns_clean, predictions)
            
            return {
                'method': method,
                'factor_weights': self.factor_weights.to_dict(),
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r_squared': r2,
                'n_factors': len(factors_clean.columns),
                'n_observations': len(returns_clean),
                'esg_weight': self.factor_weights.get('esg_alpha', 0),
                'satellite_weight': self.factor_weights.get('satellite_alpha', 0)
            }
            
        except Exception as e:
            logger.error(f"Error combining models: {e}")
            raise
    
    def generate_combined_scores(self, esg_factors: pd.DataFrame,
                               satellite_factors: pd.DataFrame,
                               traditional_factors: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate combined alpha scores from all factor models.
        
        Args:
            esg_factors: ESG factor exposures
            satellite_factors: Satellite factor exposures  
            traditional_factors: Optional traditional factors
            
        Returns:
            Combined alpha scores
        """
        try:
            if self.combined_model is None:
                raise ValueError("Combined model must be fitted first")
            
            # Generate individual scores
            esg_scores = self.esg_model.generate_esg_scores(esg_factors)
            satellite_scores = self.satellite_model.generate_satellite_scores(satellite_factors)
            
            # Combine into factor matrix
            combined_factors = pd.DataFrame({
                'esg_alpha': esg_scores,
                'satellite_alpha': satellite_scores
            })
            
            # Add traditional factors if provided
            if traditional_factors is not None:
                common_index = combined_factors.index.intersection(traditional_factors.index)
                combined_factors = combined_factors.loc[common_index]
                traditional_aligned = traditional_factors.loc[common_index]
                combined_factors = pd.concat([combined_factors, traditional_aligned], axis=1)
            
            # Ensure same columns as training
            factors_aligned = combined_factors[self.factor_weights.index].fillna(0)
            
            # Generate combined scores
            scores = self.combined_model.predict(factors_aligned)
            
            return pd.Series(scores, index=factors_aligned.index, name='combined_alpha_score')
            
        except Exception as e:
            logger.error(f"Error generating combined scores: {e}")
            raise


# Utility functions for creating sample data
def create_sample_esg_data(start_date: str = '2020-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    """Create sample ESG data for testing."""
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(42)
    
    esg_data = pd.DataFrame({
        'E_score': 50 + 20 * np.random.randn(len(dates)).cumsum() * 0.01,
        'S_score': 60 + 15 * np.random.randn(len(dates)).cumsum() * 0.01,
        'G_score': 70 + 10 * np.random.randn(len(dates)).cumsum() * 0.01,
    }, index=dates)
    
    esg_data['overall_score'] = (esg_data['E_score'] + esg_data['S_score'] + esg_data['G_score']) / 3
    esg_data = esg_data.clip(0, 100)  # ESG scores typically 0-100
    
    return esg_data

def create_sample_satellite_data(start_date: str = '2020-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    """Create sample satellite data for testing."""
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(43)
    
    satellite_data = pd.DataFrame({
        'nighttime_lights': 1000 + 100 * np.random.randn(len(dates)).cumsum() * 0.01,
        'vegetation_index': 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + 0.05 * np.random.randn(len(dates)),
        'construction_activity': 500 + 50 * np.random.randn(len(dates)).cumsum() * 0.02,
        'traffic_density': 800 + 80 * np.random.randn(len(dates)).cumsum() * 0.015,
        'port_activity': 300 + 30 * np.random.randn(len(dates)).cumsum() * 0.02,
    }, index=dates)
    
    satellite_data = satellite_data.clip(lower=0)  # Non-negative values
    
    return satellite_data


if __name__ == "__main__":
    # Example usage
    print("Testing Factor Models...")
    
    # Create sample data
    esg_data = create_sample_esg_data()
    satellite_data = create_sample_satellite_data()
    
    # Create sample returns (correlated with factors)
    np.random.seed(44)
    returns = 0.001 * (esg_data['overall_score'].pct_change() + 
                      satellite_data['nighttime_lights'].pct_change() +
                      0.02 * np.random.randn(len(esg_data)))
    
    # Test ESG model
    esg_model = ESGFactorModel()
    esg_factors = esg_model.create_esg_factors(esg_data)
    esg_results = esg_model.fit_factor_model(returns, esg_factors)
    print(f"ESG Model R-squared: {esg_results['r_squared']:.3f}")
    
    # Test Satellite model
    satellite_model = SatelliteDataModel()
    sat_factors = satellite_model.create_satellite_factors(satellite_data)
    sat_results = satellite_model.fit_satellite_model(returns, sat_factors)
    print(f"Satellite Model R-squared: {sat_results['r_squared']:.3f}")