"""
Sentiment Analysis Module for Trading Signals
=============================================

This module implements sentiment-based trading signals from news and social media data
using transformer models for financial market prediction.
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Advanced sentiment analyzer using multiple transformer models and traditional methods
    for financial text analysis.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name: HuggingFace model name for financial sentiment analysis
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
        
    def _load_models(self):
        """Load transformer and traditional sentiment models."""
        try:
            # Load FinBERT for financial sentiment
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load VADER for social media sentiment
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            logger.info(f"Sentiment models loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            raise
    
    def analyze_financial_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of financial text using FinBERT.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary with sentiment scores and classification
        """
        try:
            # FinBERT analysis
            finbert_result = self.finbert_analyzer(text)[0]
            
            # TextBlob analysis for comparison
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # VADER analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            return {
                'finbert_label': finbert_result['label'],
                'finbert_score': finbert_result['score'],
                'textblob_polarity': textblob_polarity,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {'error': str(e)}
    
    def batch_analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment scores for each text
        """
        results = []
        
        for i, text in enumerate(texts):
            sentiment_scores = self.analyze_financial_text(text)
            sentiment_scores['text_id'] = i
            sentiment_scores['text'] = text[:100] + "..." if len(text) > 100 else text
            results.append(sentiment_scores)
            
        return pd.DataFrame(results)


class TradingSignalGenerator:
    """
    Generate trading signals based on sentiment analysis of news and social media data.
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize trading signal generator.
        
        Args:
            sentiment_analyzer: Initialized SentimentAnalyzer instance
        """
        self.sentiment_analyzer = sentiment_analyzer
        
    def aggregate_sentiment_scores(self, sentiment_df: pd.DataFrame, 
                                 timestamp_col: str = 'timestamp') -> Dict[str, float]:
        """
        Aggregate sentiment scores over a time period.
        
        Args:
            sentiment_df: DataFrame with sentiment scores and timestamps
            timestamp_col: Name of timestamp column
            
        Returns:
            Aggregated sentiment metrics
        """
        if sentiment_df.empty:
            return {'error': 'Empty sentiment data'}
            
        # Calculate weighted sentiment scores
        finbert_scores = sentiment_df['finbert_score'].values
        finbert_labels = sentiment_df['finbert_label'].values
        
        # Convert labels to numeric scores
        label_weights = {'positive': 1, 'negative': -1, 'neutral': 0}
        weighted_finbert = [
            score * label_weights.get(label.lower(), 0) 
            for score, label in zip(finbert_scores, finbert_labels)
        ]
        
        return {
            'avg_finbert_sentiment': np.mean(weighted_finbert),
            'avg_vader_compound': sentiment_df['vader_compound'].mean(),
            'avg_textblob_polarity': sentiment_df['textblob_polarity'].mean(),
            'sentiment_volatility': np.std(weighted_finbert),
            'positive_ratio': sum(1 for label in finbert_labels if label.lower() == 'positive') / len(finbert_labels),
            'negative_ratio': sum(1 for label in finbert_labels if label.lower() == 'negative') / len(finbert_labels),
            'total_articles': len(sentiment_df)
        }
    
    def generate_signals(self, sentiment_metrics: Dict[str, float], 
                        price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on sentiment metrics and price data.
        
        Args:
            sentiment_metrics: Aggregated sentiment scores
            price_data: Historical price data with OHLCV columns
            
        Returns:
            Trading signal recommendations
        """
        if 'error' in sentiment_metrics:
            return sentiment_metrics
            
        # Extract key sentiment indicators
        finbert_sentiment = sentiment_metrics['avg_finbert_sentiment']
        vader_sentiment = sentiment_metrics['avg_vader_compound']
        sentiment_vol = sentiment_metrics['sentiment_volatility']
        positive_ratio = sentiment_metrics['positive_ratio']
        
        # Calculate price momentum
        if len(price_data) >= 5:
            recent_returns = price_data['Close'].pct_change().tail(5).mean()
            price_volatility = price_data['Close'].pct_change().std()
        else:
            recent_returns = 0
            price_volatility = 0
        
        # Generate composite signal
        sentiment_signal = (finbert_sentiment * 0.6 + vader_sentiment * 0.4)
        
        # Adjust for sentiment confidence and consistency
        confidence_multiplier = min(1.0, positive_ratio + (1 - sentiment_vol))
        
        # Combine sentiment with momentum
        momentum_weight = 0.3
        sentiment_weight = 0.7
        
        composite_signal = (
            sentiment_signal * sentiment_weight + 
            recent_returns * momentum_weight
        ) * confidence_multiplier
        
        # Generate position sizing recommendation
        if abs(composite_signal) < 0.1:
            position_size = 0.0  # No position
            signal_strength = "NEUTRAL"
        elif composite_signal > 0.3:
            position_size = min(1.0, composite_signal * 2)  # Long position
            signal_strength = "STRONG_BUY" if composite_signal > 0.6 else "BUY"
        elif composite_signal < -0.3:
            position_size = max(-1.0, composite_signal * 2)  # Short position  
            signal_strength = "STRONG_SELL" if composite_signal < -0.6 else "SELL"
        else:
            position_size = composite_signal * 0.5  # Weak position
            signal_strength = "WEAK_BUY" if composite_signal > 0 else "WEAK_SELL"
        
        return {
            'composite_signal': composite_signal,
            'position_size': position_size,
            'signal_strength': signal_strength,
            'sentiment_contribution': sentiment_signal * sentiment_weight,
            'momentum_contribution': recent_returns * momentum_weight,
            'confidence_multiplier': confidence_multiplier,
            'risk_adjusted_signal': composite_signal / max(price_volatility, 0.01)
        }
    
    def backtest_signals(self, symbol: str, start_date: str, end_date: str,
                        news_data: pd.DataFrame) -> Dict[str, float]:
        """
        Backtest sentiment-based trading signals.
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            news_data: DataFrame with news data and timestamps
            
        Returns:
            Backtest performance metrics
        """
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            price_data = ticker.history(start=start_date, end=end_date)
            
            if price_data.empty:
                return {'error': f'No price data found for {symbol}'}
            
            # Analyze sentiment for news data
            sentiment_df = self.sentiment_analyzer.batch_analyze(news_data['text'].tolist())
            
            # Merge with timestamps
            sentiment_df['date'] = pd.to_datetime(news_data['timestamp'].values)
            sentiment_df = sentiment_df.set_index('date')
            
            # Generate daily signals
            daily_signals = []
            daily_returns = []
            
            for date in price_data.index:
                # Get sentiment data for the day
                day_sentiment = sentiment_df[sentiment_df.index.date == date.date()]
                
                if not day_sentiment.empty:
                    sentiment_metrics = self.aggregate_sentiment_scores(day_sentiment)
                    
                    # Get price data up to current date
                    historical_prices = price_data[price_data.index <= date]
                    
                    if len(historical_prices) >= 5:
                        signals = self.generate_signals(sentiment_metrics, historical_prices)
                        
                        # Calculate next day return
                        next_day_idx = price_data.index.get_loc(date) + 1
                        if next_day_idx < len(price_data):
                            next_day_return = (
                                price_data.iloc[next_day_idx]['Close'] / 
                                price_data.iloc[price_data.index.get_loc(date)]['Close'] - 1
                            )
                            
                            daily_signals.append(signals['composite_signal'])
                            daily_returns.append(next_day_return)
            
            # Calculate performance metrics
            if daily_signals and daily_returns:
                signals_array = np.array(daily_signals)
                returns_array = np.array(daily_returns)
                
                # Strategy returns (signal * next day return)
                strategy_returns = signals_array * returns_array
                
                # Performance metrics
                total_return = np.prod(1 + strategy_returns) - 1
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
                max_drawdown = self._calculate_max_drawdown(strategy_returns)
                win_rate = sum(1 for r in strategy_returns if r > 0) / len(strategy_returns)
                
                return {
                    'total_return': total_return,
                    'annualized_return': (1 + total_return) ** (252 / len(strategy_returns)) - 1,
                    'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(strategy_returns),
                    'avg_signal_strength': np.mean(np.abs(signals_array))
                }
            else:
                return {'error': 'Insufficient data for backtesting'}
                
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)


# Example usage and utility functions
def create_sample_news_data() -> pd.DataFrame:
    """Create sample news data for testing purposes."""
    sample_news = [
        {
            'timestamp': '2024-01-15 09:30:00',
            'text': 'Company XYZ reports strong quarterly earnings beating analyst expectations by 15%',
            'source': 'Financial News'
        },
        {
            'timestamp': '2024-01-15 10:15:00', 
            'text': 'Market volatility increases as investors worry about inflation data',
            'source': 'Market Watch'
        },
        {
            'timestamp': '2024-01-15 14:20:00',
            'text': 'Tech stocks rally on positive AI development news and strong sector outlook',
            'source': 'Tech Times'
        }
    ]
    
    return pd.DataFrame(sample_news)


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    signal_generator = TradingSignalGenerator(analyzer)
    
    # Test with sample data
    sample_news = create_sample_news_data()
    sentiment_results = analyzer.batch_analyze(sample_news['text'].tolist())
    print("Sentiment Analysis Results:")
    print(sentiment_results)