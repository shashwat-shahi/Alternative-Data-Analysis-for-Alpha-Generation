"""
Alternative Data Analysis for Alpha Generation
============================================

A comprehensive framework for generating alpha using alternative data sources including:
- Sentiment analysis from news and social media
- Statistical validation with regime detection
- Factor models incorporating ESG and satellite data
"""

__version__ = "1.0.0"
__author__ = "Alternative Data Analysis Team"

from .sentiment_analysis import SentimentAnalyzer, TradingSignalGenerator
from .statistical_validation import AlternativeDataValidator, RegimeDetector
from .factor_models import ESGFactorModel, SatelliteDataModel, MultiFactorModel

__all__ = [
    'SentimentAnalyzer',
    'TradingSignalGenerator', 
    'AlternativeDataValidator',
    'RegimeDetector',
    'ESGFactorModel',
    'SatelliteDataModel',
    'MultiFactorModel'
]