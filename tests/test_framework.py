"""
Test Suite for Alternative Data Analysis Framework
================================================

Basic tests to ensure core functionality works correctly.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analysis import SentimentAnalyzer, TradingSignalGenerator
from statistical_validation import AlternativeDataValidator, RegimeDetector
from factor_models import ESGFactorModel, SatelliteDataModel, create_sample_esg_data, create_sample_satellite_data
from data_sources import DataIntegrator
from utils import DataProcessor, PerformanceAnalyzer

class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.signal_generator = TradingSignalGenerator(self.analyzer)
        
    def test_sentiment_analysis(self):
        """Test basic sentiment analysis."""
        text = "Company reports strong quarterly earnings beating expectations"
        result = self.analyzer.analyze_financial_text(text)
        
        self.assertIn('finbert_label', result)
        self.assertIn('finbert_score', result)
        self.assertIn('vader_compound', result)
        
    def test_batch_analysis(self):
        """Test batch sentiment analysis."""
        texts = [
            "Strong earnings beat expectations",
            "Market volatility concerns investors",
            "Positive outlook for the sector"
        ]
        
        results = self.analyzer.batch_analyze(texts)
        self.assertEqual(len(results), len(texts))
        self.assertIn('finbert_label', results.columns)

class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AlternativeDataValidator()
        self.regime_detector = RegimeDetector()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        self.signal = pd.Series(np.random.randn(len(dates)), index=dates)
        self.returns = pd.Series(0.1 * self.signal.shift(1) + 0.02 * np.random.randn(len(dates)), index=dates)
        
    def test_signal_significance(self):
        """Test signal significance testing."""
        results = self.validator.test_signal_significance(self.signal, self.returns)
        
        self.assertIn('pearson_correlation', results)
        self.assertIn('information_coefficient', results)
        self.assertIn('is_significant', results)
        
    def test_regime_detection(self):
        """Test regime detection."""
        results = self.regime_detector.detect_regimes_markov(self.returns)
        
        if 'error' not in results:
            self.assertIn('regime_probabilities', results)
            self.assertIn('regime_states', results)

class TestFactorModels(unittest.TestCase):
    """Test factor model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.esg_data = create_sample_esg_data()
        self.satellite_data = create_sample_satellite_data()
        
        # Create sample returns
        np.random.seed(42)
        returns_data = 0.001 * (
            self.esg_data['overall_score'].pct_change() + 
            self.satellite_data['nighttime_lights'].pct_change() +
            0.02 * np.random.randn(len(self.esg_data))
        )
        self.returns = returns_data.dropna()
        
    def test_esg_factor_model(self):
        """Test ESG factor model."""
        esg_model = ESGFactorModel()
        esg_factors = esg_model.create_esg_factors(self.esg_data)
        
        self.assertGreater(len(esg_factors.columns), len(self.esg_data.columns))
        
        # Test model fitting
        results = esg_model.fit_factor_model(self.returns, esg_factors)
        self.assertIn('r_squared', results)
        
    def test_satellite_model(self):
        """Test satellite data model."""
        satellite_model = SatelliteDataModel()
        satellite_factors = satellite_model.create_satellite_factors(self.satellite_data)
        
        self.assertGreater(len(satellite_factors.columns), len(self.satellite_data.columns))
        
        # Test model fitting
        results = satellite_model.fit_satellite_model(self.returns, satellite_factors)
        self.assertIn('r_squared', results)

class TestDataSources(unittest.TestCase):
    """Test data source functionality."""
    
    def test_data_integrator(self):
        """Test data integration."""
        integrator = DataIntegrator()
        dataset = integrator.create_integrated_dataset(
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-06-30'
        )
        
        self.assertIn('stock_data', dataset)
        self.assertIn('returns', dataset)
        self.assertIn('news_data', dataset)
        self.assertIn('esg_data', dataset)

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        self.data = pd.DataFrame({
            'value1': np.random.randn(len(dates)),
            'value2': np.random.randn(len(dates)) * 0.5,
            'value3': np.random.randn(len(dates)) * 2
        }, index=dates)
        
        self.returns = pd.Series(0.001 + 0.02 * np.random.randn(len(dates)), index=dates)
        
    def test_data_processor(self):
        """Test data processing utilities."""
        processor = DataProcessor()
        
        # Test data cleaning
        cleaned_data = processor.clean_data(self.data)
        self.assertLessEqual(len(cleaned_data), len(self.data))
        
        # Test winsorization
        winsorized_data = processor.winsorize_data(self.data)
        self.assertEqual(len(winsorized_data), len(self.data))
        
    def test_performance_analyzer(self):
        """Test performance analysis."""
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_performance_metrics(self.returns)
        
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSentimentAnalysis,
        TestStatisticalValidation,
        TestFactorModels,
        TestDataSources,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Alternative Data Analysis Framework Tests...")
    print("="*60)
    
    success = run_tests()
    
    if success:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        sys.exit(1)