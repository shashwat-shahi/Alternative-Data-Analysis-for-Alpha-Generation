"""
Utility Functions for Alternative Data Analysis
==============================================

Common utilities for data processing, visualization, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataProcessor:
    """
    Utility class for data processing and cleaning.
    """
    
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   remove_outliers: bool = True,
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean dataset by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Forward fill then backward fill
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers if requested
        if remove_outliers:
            for col in numeric_cols:
                if outlier_method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    
                elif outlier_method == 'zscore':
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < outlier_threshold]
        
        logger.info(f"Data cleaning removed {len(df) - len(df_clean)} rows")
        return df_clean
    
    @staticmethod
    def winsorize_data(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      lower_percentile: float = 0.01,
                      upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Winsorize data to handle extreme values.
        
        Args:
            df: Input DataFrame
            columns: Columns to winsorize (if None, all numeric columns)
            lower_percentile: Lower percentile for winsorization
            upper_percentile: Upper percentile for winsorization
            
        Returns:
            Winsorized DataFrame
        """
        df_winsorized = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df.columns:
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                df_winsorized[col] = df_winsorized[col].clip(lower_bound, upper_bound)
        
        return df_winsorized
    
    @staticmethod
    def create_lagged_features(df: pd.DataFrame, 
                             columns: List[str],
                             lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            df: Input DataFrame with datetime index
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df_lagged = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                              columns: List[str],
                              windows: List[int],
                              functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of rolling window sizes
            functions: List of functions to apply ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            df_rolling[f'{col}_rolling_{window}_mean'] = df[col].rolling(window).mean()
                        elif func == 'std':
                            df_rolling[f'{col}_rolling_{window}_std'] = df[col].rolling(window).std()
                        elif func == 'min':
                            df_rolling[f'{col}_rolling_{window}_min'] = df[col].rolling(window).min()
                        elif func == 'max':
                            df_rolling[f'{col}_rolling_{window}_max'] = df[col].rolling(window).max()
                        elif func == 'median':
                            df_rolling[f'{col}_rolling_{window}_median'] = df[col].rolling(window).median()
        
        return df_rolling


class Visualizer:
    """
    Visualization utilities for alternative data analysis.
    """
    
    @staticmethod
    def plot_time_series(data: Union[pd.Series, pd.DataFrame], 
                        title: str = "Time Series Plot",
                        figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot time series data.
        
        Args:
            data: Time series data
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if isinstance(data, pd.Series):
            plt.plot(data.index, data.values, linewidth=2)
            plt.ylabel(data.name or 'Value')
        else:
            for col in data.columns:
                plt.plot(data.index, data[col], label=col, linewidth=2)
            plt.legend()
            plt.ylabel('Value')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, 
                              title: str = "Correlation Matrix",
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame with numeric columns
            title: Plot title
            figsize: Figure size
        """
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_factor_performance(factor_returns: pd.DataFrame,
                              benchmark_returns: pd.Series,
                              title: str = "Factor Performance") -> None:
        """
        Plot factor performance vs benchmark.
        
        Args:
            factor_returns: DataFrame with factor returns
            benchmark_returns: Benchmark returns series
            title: Plot title
        """
        # Calculate cumulative returns
        factor_cumulative = (1 + factor_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        # Add factor lines
        for factor in factor_returns.columns:
            fig.add_trace(go.Scatter(
                x=factor_cumulative.index,
                y=factor_cumulative[factor],
                mode='lines',
                name=factor,
                line=dict(width=2)
            ))
        
        # Add benchmark
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative,
            mode='lines',
            name='Benchmark',
            line=dict(width=3, color='black', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    @staticmethod
    def plot_signal_analysis(signal: pd.Series, 
                           returns: pd.Series,
                           title: str = "Signal Analysis") -> None:
        """
        Plot signal vs returns analysis.
        
        Args:
            signal: Signal time series
            returns: Returns time series
            title: Plot title
        """
        # Align data
        common_index = signal.index.intersection(returns.index)
        signal_aligned = signal.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Signal Over Time', 'Returns Over Time', 'Signal vs Returns Scatter'],
            vertical_spacing=0.08
        )
        
        # Signal time series
        fig.add_trace(go.Scatter(
            x=signal_aligned.index,
            y=signal_aligned.values,
            mode='lines',
            name='Signal',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Returns time series
        fig.add_trace(go.Scatter(
            x=returns_aligned.index,
            y=returns_aligned.values,
            mode='lines',
            name='Returns',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=signal_aligned.values,
            y=returns_aligned.values,
            mode='markers',
            name='Signal vs Returns',
            marker=dict(color='green', opacity=0.6)
        ), row=3, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        fig.show()
    
    @staticmethod
    def plot_regime_analysis(returns: pd.Series, 
                           regime_probs: pd.DataFrame,
                           title: str = "Regime Analysis") -> None:
        """
        Plot regime detection results.
        
        Args:
            returns: Returns time series
            regime_probs: DataFrame with regime probabilities
            title: Plot title
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Returns with Regime Overlay', 'Regime Probabilities'],
            shared_xaxis=True,
            vertical_spacing=0.1
        )
        
        # Returns with regime coloring
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns.values,
            mode='lines',
            name='Returns',
            line=dict(color='black')
        ), row=1, col=1)
        
        # Regime probabilities
        colors = ['blue', 'red', 'green', 'orange']
        for i, col in enumerate(regime_probs.columns):
            fig.add_trace(go.Scatter(
                x=regime_probs.index,
                y=regime_probs[col],
                mode='lines',
                name=f'Regime {i}',
                line=dict(color=colors[i % len(colors)])
            ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=600,
            template='plotly_white'
        )
        
        fig.show()


class PerformanceAnalyzer:
    """
    Performance analysis utilities.
    """
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, 
                                    benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Returns time series
            benchmark: Optional benchmark returns
            
        Returns:
            Dictionary with performance metrics
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'error': 'No valid returns data'}
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        periods_per_year = 252 if len(returns_clean) > 252 else len(returns_clean)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns_clean)) - 1
        
        volatility = returns_clean.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        var_95 = returns_clean.quantile(0.05)
        var_99 = returns_clean.quantile(0.01)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'win_rate': (returns_clean > 0).mean(),
            'n_observations': len(returns_clean)
        }
        
        # Benchmark comparison
        if benchmark is not None:
            common_index = returns.index.intersection(benchmark.index)
            if len(common_index) > 10:
                returns_aligned = returns.loc[common_index]
                benchmark_aligned = benchmark.loc[common_index]
                
                excess_returns = returns_aligned - benchmark_aligned
                tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
                
                metrics.update({
                    'excess_return': excess_returns.mean() * periods_per_year,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'beta': returns_aligned.cov(benchmark_aligned) / benchmark_aligned.var() if benchmark_aligned.var() > 0 else 0
                })
        
        return metrics
    
    @staticmethod
    def calculate_factor_attribution(returns: pd.Series,
                                   factor_exposures: pd.DataFrame,
                                   factor_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate factor attribution for returns.
        
        Args:
            returns: Portfolio returns
            factor_exposures: Factor exposures over time
            factor_returns: Factor returns over time
            
        Returns:
            Factor attribution results
        """
        try:
            # Align all data
            common_index = returns.index.intersection(factor_exposures.index).intersection(factor_returns.index)
            
            if len(common_index) < 10:
                return {'error': 'Insufficient aligned data for attribution'}
            
            returns_aligned = returns.loc[common_index]
            exposures_aligned = factor_exposures.loc[common_index]
            factor_rets_aligned = factor_returns.loc[common_index]
            
            # Calculate factor contributions
            attribution = {}
            total_attribution = 0
            
            for factor in exposures_aligned.columns:
                if factor in factor_rets_aligned.columns:
                    # Factor contribution = exposure * factor return
                    factor_contribution = (exposures_aligned[factor] * factor_rets_aligned[factor]).mean()
                    attribution[f'{factor}_contribution'] = factor_contribution
                    total_attribution += factor_contribution
            
            # Residual (unexplained) return
            total_return = returns_aligned.mean()
            residual = total_return - total_attribution
            attribution['residual'] = residual
            attribution['total_return'] = total_return
            attribution['explained_return'] = total_attribution
            attribution['r_squared'] = 1 - (residual / total_return) ** 2 if total_return != 0 else 0
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error in factor attribution: {e}")
            return {'error': str(e)}


# Utility functions
def load_config(config_path: str) -> Dict:
    """Load configuration from file."""
    import json
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_results(results: Dict, filepath: str) -> None:
    """Save results to file."""
    import pickle
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def load_results(filepath: str) -> Dict:
    """Load results from file."""
    import pickle
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = pd.Series(0.001 + 0.02 * np.random.randn(len(dates)), index=dates)
    
    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_performance_metrics(returns)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")