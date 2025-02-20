#!/usr/bin/env python3
"""
Alternative Data Analysis for Alpha Generation - Main Example
============================================================

This script demonstrates the complete workflow for generating alpha using
alternative data sources including sentiment analysis, ESG factors, and satellite data.
"""

import sys
import os
import warnings
import json
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import logging

# Import our modules
from sentiment_analysis import SentimentAnalyzer, TradingSignalGenerator
from statistical_validation import AlternativeDataValidator, RegimeDetector
from factor_models import ESGFactorModel, SatelliteDataModel, MultiFactorModel
from data_sources import DataIntegrator
from utils import Visualizer, PerformanceAnalyzer, DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def load_config():
    """Load configuration from config file."""
    try:
        with open('config/config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        return {
            "validation": {"confidence_level": 0.95},
            "models": {
                "sentiment_analysis": {"model_name": "ProsusAI/finbert"},
                "esg_factor_model": {"standardize": True, "method": "ridge"},
                "satellite_model": {"use_pca": True, "method": "random_forest"}
            }
        }

def main():
    """Main execution function."""
    logger.info("Starting Alternative Data Analysis for Alpha Generation")
    
    # Load configuration
    config = load_config()
    
    # Parameters
    SYMBOL = 'AAPL'
    START_DATE = '2022-01-01'
    END_DATE = '2023-12-31'
    REGION = 'US_WEST'
    
    logger.info(f"Analyzing {SYMBOL} from {START_DATE} to {END_DATE}")
    
    try:
        # Step 1: Data Integration
        logger.info("Step 1: Integrating data sources...")
        integrator = DataIntegrator()
        dataset = integrator.create_integrated_dataset(
            symbol=SYMBOL,
            start_date=START_DATE,
            end_date=END_DATE,
            region=REGION
        )
        
        if not dataset:
            logger.error("Failed to create integrated dataset")
            return
        
        logger.info("✓ Data integration completed")
        logger.info(f"  - Stock data: {len(dataset['stock_data'])} observations")
        logger.info(f"  - News articles: {len(dataset['news_data'])} articles")
        logger.info(f"  - ESG data: {len(dataset['esg_data'])} observations")
        logger.info(f"  - Satellite data: {len(dataset['satellite_data'])} observations")
        
        # Step 2: Sentiment Analysis
        logger.info("\nStep 2: Performing sentiment analysis...")
        
        # Initialize sentiment analyzer
        sentiment_config = config.get('models', {}).get('sentiment_analysis', {})
        sentiment_analyzer = SentimentAnalyzer(
            model_name=sentiment_config.get('model_name', 'ProsusAI/finbert')
        )
        
        # Analyze news sentiment
        news_texts = dataset['news_data']['text'].tolist()
        sentiment_df = sentiment_analyzer.batch_analyze(news_texts)
        
        # Generate trading signals
        signal_generator = TradingSignalGenerator(sentiment_analyzer)
        sentiment_metrics = signal_generator.aggregate_sentiment_scores(sentiment_df)
        trading_signals = signal_generator.generate_signals(sentiment_metrics, dataset['stock_data'])
        
        logger.info("✓ Sentiment analysis completed")
        logger.info(f"  - Average sentiment: {sentiment_metrics.get('avg_finbert_sentiment', 0):.3f}")
        logger.info(f"  - Signal strength: {trading_signals.get('signal_strength', 'Unknown')}")
        logger.info(f"  - Position size: {trading_signals.get('position_size', 0):.3f}")
        
        # Step 3: ESG Factor Model
        logger.info("\nStep 3: Building ESG factor model...")
        
        esg_config = config.get('models', {}).get('esg_factor_model', {})
        esg_model = ESGFactorModel(standardize=esg_config.get('standardize', True))
        
        # Create ESG factors
        esg_factors = esg_model.create_esg_factors(dataset['esg_data'])
        
        # Fit ESG model
        esg_results = esg_model.fit_factor_model(
            dataset['returns'], 
            esg_factors,
            method=esg_config.get('method', 'ridge')
        )
        
        logger.info("✓ ESG factor model completed")
        logger.info(f"  - R-squared: {esg_results.get('r_squared', 0):.3f}")
        logger.info(f"  - Number of factors: {esg_results.get('n_factors', 0)}")
        
        # Step 4: Satellite Data Model
        logger.info("\nStep 4: Building satellite data model...")
        
        sat_config = config.get('models', {}).get('satellite_model', {})
        satellite_model = SatelliteDataModel(standardize=True)
        
        # Create satellite factors
        satellite_factors = satellite_model.create_satellite_factors(dataset['satellite_data'])
        
        # Fit satellite model
        satellite_results = satellite_model.fit_satellite_model(
            dataset['returns'],
            satellite_factors,
            use_pca=sat_config.get('use_pca', True),
            method=sat_config.get('method', 'random_forest')
        )
        
        logger.info("✓ Satellite data model completed")
        logger.info(f"  - R-squared: {satellite_results.get('r_squared', 0):.3f}")
        logger.info(f"  - PCA explained variance: {satellite_results.get('pca_explained_variance', 0):.3f}")
        
        # Step 5: Multi-Factor Model
        logger.info("\nStep 5: Combining models...")
        
        multi_factor_model = MultiFactorModel(esg_model, satellite_model)
        combined_results = multi_factor_model.combine_models(
            dataset['returns'],
            esg_factors,
            satellite_factors
        )
        
        logger.info("✓ Multi-factor model completed")
        logger.info(f"  - Combined R-squared: {combined_results.get('r_squared', 0):.3f}")
        logger.info(f"  - ESG weight: {combined_results.get('esg_weight', 0):.3f}")
        logger.info(f"  - Satellite weight: {combined_results.get('satellite_weight', 0):.3f}")
        
        # Step 6: Statistical Validation
        logger.info("\nStep 6: Statistical validation...")
        
        validation_config = config.get('validation', {})
        validator = AlternativeDataValidator(
            confidence_level=validation_config.get('confidence_level', 0.95)
        )
        
        # Generate combined alpha scores
        combined_scores = multi_factor_model.generate_combined_scores(
            esg_factors, satellite_factors
        )
        
        # Validate combined signal
        validation_results = validator.test_signal_significance(
            combined_scores, dataset['returns'], lag=1
        )
        
        logger.info("✓ Statistical validation completed")
        logger.info(f"  - Signal is significant: {validation_results.get('is_significant', False)}")
        logger.info(f"  - Information Coefficient: {validation_results.get('information_coefficient', 0):.3f}")
        logger.info(f"  - Hit rate: {validation_results.get('hit_rate', 0):.3f}")
        
        # Step 7: Regime Detection
        logger.info("\nStep 7: Regime detection...")
        
        regime_detector = RegimeDetector(n_regimes=2)
        regime_results = regime_detector.detect_regimes_markov(dataset['returns'])
        
        if 'error' not in regime_results:
            regime_states = regime_results['regime_states']
            
            # Regime-conditional validation
            regime_validation = regime_detector.regime_conditional_validation(
                combined_scores, dataset['returns'], regime_states
            )
            
            logger.info("✓ Regime detection completed")
            logger.info(f"  - Number of regimes detected: {len(regime_results.get('regime_statistics', {}))}")
            logger.info(f"  - Model AIC: {regime_results.get('model_aic', 0):.2f}")
        else:
            logger.warning(f"Regime detection failed: {regime_results['error']}")
        
        # Step 8: Performance Analysis
        logger.info("\nStep 8: Performance analysis...")
        
        analyzer = PerformanceAnalyzer()
        
        # Simulate strategy returns (simplified)
        strategy_returns = combined_scores.shift(1) * dataset['returns']
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) > 0:
            performance_metrics = analyzer.calculate_performance_metrics(
                strategy_returns, dataset['returns']
            )
            
            logger.info("✓ Performance analysis completed")
            logger.info(f"  - Total return: {performance_metrics.get('total_return', 0):.3f}")
            logger.info(f"  - Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  - Max drawdown: {performance_metrics.get('max_drawdown', 0):.3f}")
            logger.info(f"  - Information ratio: {performance_metrics.get('information_ratio', 0):.3f}")
        
        # Step 9: Summary Results
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        summary = {
            'symbol': SYMBOL,
            'period': f"{START_DATE} to {END_DATE}",
            'sentiment_signal_strength': trading_signals.get('signal_strength'),
            'esg_model_rsquared': esg_results.get('r_squared', 0),
            'satellite_model_rsquared': satellite_results.get('r_squared', 0),
            'combined_model_rsquared': combined_results.get('r_squared', 0),
            'signal_significant': validation_results.get('is_significant', False),
            'information_coefficient': validation_results.get('information_coefficient', 0),
            'strategy_sharpe_ratio': performance_metrics.get('sharpe_ratio', 0) if 'performance_metrics' in locals() else 0
        }
        
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        logger.info("\n✓ Alternative Data Analysis completed successfully!")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    try:
        results = main()
        print("\nExecution completed successfully!")
        print("Check the logs for detailed analysis results.")
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nExecution failed: {e}")
        sys.exit(1)