"""
Data Sources Module
==================

Utilities for fetching and processing alternative data from various sources
including news APIs, social media, financial data, and ESG data providers.
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataProvider:
    """
    Provider for news data from various sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news data provider.
        
        Args:
            api_key: API key for news services (NewsAPI, etc.)
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def fetch_company_news(self, company: str, from_date: str, to_date: str,
                          language: str = 'en', page_size: int = 100) -> pd.DataFrame:
        """
        Fetch news articles for a specific company.
        
        Args:
            company: Company name or ticker
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            page_size: Number of articles per page
            
        Returns:
            DataFrame with news articles
        """
        try:
            if not self.api_key:
                logger.warning("No API key provided, returning sample data")
                return self._create_sample_news_data(company, from_date, to_date)
            
            # Search for company news
            params = {
                'q': company,
                'from': from_date,
                'to': to_date,
                'language': language,
                'pageSize': page_size,
                'apiKey': self.api_key,
                'sortBy': 'publishedAt'
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                news_data = []
                for article in articles:
                    news_data.append({
                        'timestamp': article.get('publishedAt'),
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'author': article.get('author', ''),
                        'text': f"{article.get('title', '')} {article.get('description', '')}"
                    })
                
                df = pd.DataFrame(news_data)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Fetched {len(df)} news articles for {company}")
                return df
            else:
                logger.error(f"Error fetching news: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching company news: {e}")
            return self._create_sample_news_data(company, from_date, to_date)
    
    def _create_sample_news_data(self, company: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Create sample news data for testing."""
        dates = pd.date_range(from_date, to_date, freq='D')
        np.random.seed(42)
        
        sample_articles = [
            f"{company} reports strong quarterly earnings with revenue growth",
            f"Analysts upgrade {company} stock on positive outlook", 
            f"{company} announces new product launch and market expansion",
            f"Market volatility affects {company} stock performance",
            f"{company} CEO discusses company strategy in investor call",
            f"Industry trends benefit {company} business model",
            f"{company} faces regulatory challenges in key markets",
            f"{company} partnership announced with major technology firm",
            f"ESG initiatives at {company} gain investor attention",
            f"{company} stock reaches new highs on strong fundamentals"
        ]
        
        news_data = []
        for date in dates[::2]:  # Every other day
            if np.random.random() > 0.3:  # 70% chance of news
                article = np.random.choice(sample_articles)
                news_data.append({
                    'timestamp': date,
                    'title': article,
                    'description': f"Detailed analysis of {article.lower()}",
                    'content': f"Full article content about {article.lower()}...",
                    'source': np.random.choice(['Financial Times', 'Reuters', 'Bloomberg', 'WSJ']),
                    'url': f"https://example.com/news/{len(news_data)}",
                    'author': f"Reporter {np.random.randint(1, 10)}",
                    'text': article
                })
        
        return pd.DataFrame(news_data)


class ESGDataProvider:
    """
    Provider for ESG (Environmental, Social, Governance) data.
    """
    
    def __init__(self, provider: str = 'sample'):
        """
        Initialize ESG data provider.
        
        Args:
            provider: ESG data provider ('refinitiv', 'msci', 'sample')
        """
        self.provider = provider
        
    def fetch_esg_scores(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch ESG scores for a company.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with ESG scores
        """
        try:
            if self.provider == 'sample':
                return self._create_sample_esg_data(symbol, start_date, end_date)
            else:
                logger.warning(f"Provider {self.provider} not implemented, using sample data")
                return self._create_sample_esg_data(symbol, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching ESG data: {e}")
            return pd.DataFrame()
    
    def _create_sample_esg_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create sample ESG data."""
        dates = pd.date_range(start_date, end_date, freq='QS')  # Quarterly data
        np.random.seed(hash(symbol) % 1000)
        
        # Base scores with some persistence and trend
        base_e = 60 + 10 * np.random.randn()
        base_s = 65 + 8 * np.random.randn()
        base_g = 70 + 6 * np.random.randn()
        
        esg_data = []
        for i, date in enumerate(dates):
            # Add some trend and noise
            trend = i * 0.5
            noise_e = 3 * np.random.randn()
            noise_s = 2.5 * np.random.randn()
            noise_g = 2 * np.random.randn()
            
            e_score = np.clip(base_e + trend + noise_e, 0, 100)
            s_score = np.clip(base_s + trend + noise_s, 0, 100)
            g_score = np.clip(base_g + trend + noise_g, 0, 100)
            
            esg_data.append({
                'timestamp': date,
                'symbol': symbol,
                'E_score': e_score,
                'S_score': s_score,
                'G_score': g_score,
                'overall_score': (e_score + s_score + g_score) / 3,
                'industry': np.random.choice(['Technology', 'Financial', 'Healthcare', 'Energy']),
                'market_cap': np.random.choice(['Large', 'Mid', 'Small'])
            })
        
        df = pd.DataFrame(esg_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')


class SatelliteDataProvider:
    """
    Provider for satellite and geospatial data.
    """
    
    def __init__(self, provider: str = 'sample'):
        """
        Initialize satellite data provider.
        
        Args:
            provider: Data provider ('planet', 'maxar', 'sample')
        """
        self.provider = provider
        
    def fetch_economic_activity_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch economic activity data from satellite imagery.
        
        Args:
            region: Geographic region identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with satellite-derived economic indicators
        """
        try:
            if self.provider == 'sample':
                return self._create_sample_satellite_data(region, start_date, end_date)
            else:
                logger.warning(f"Provider {self.provider} not implemented, using sample data")
                return self._create_sample_satellite_data(region, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching satellite data: {e}")
            return pd.DataFrame()
    
    def _create_sample_satellite_data(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create sample satellite data."""
        dates = pd.date_range(start_date, end_date, freq='W')  # Weekly data
        np.random.seed(hash(region) % 1000)
        
        # Base values for different metrics
        base_lights = 1000 + 200 * np.random.randn()
        base_vegetation = 0.6 + 0.1 * np.random.randn()
        base_construction = 500 + 100 * np.random.randn()
        
        satellite_data = []
        for i, date in enumerate(dates):
            # Seasonal patterns
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 52)  # Annual cycle
            
            # Trend and noise
            trend = i * 0.002
            
            nighttime_lights = max(0, base_lights * seasonal_factor * (1 + trend) + 50 * np.random.randn())
            vegetation_idx = np.clip(base_vegetation * seasonal_factor + 0.05 * np.random.randn(), 0, 1)
            construction = max(0, base_construction * (1 + trend) + 25 * np.random.randn())
            traffic = max(0, 800 * seasonal_factor + 40 * np.random.randn())
            port_activity = max(0, 300 * seasonal_factor * (1 + trend) + 20 * np.random.randn())
            
            satellite_data.append({
                'timestamp': date,
                'region': region,
                'nighttime_lights': nighttime_lights,
                'vegetation_index': vegetation_idx,
                'construction_activity': construction,
                'traffic_density': traffic,
                'port_activity': port_activity,
                'economic_activity': nighttime_lights * 0.0008 + construction * 0.001  # Composite
            })
        
        df = pd.DataFrame(satellite_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')


class FinancialDataProvider:
    """
    Provider for financial market data.
    """
    
    def __init__(self):
        """Initialize financial data provider."""
        pass
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock price data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}, creating sample data")
                return self._create_sample_stock_data(symbol, start_date, end_date)
            
            logger.info(f"Fetched {len(data)} days of stock data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return self._create_sample_stock_data(symbol, start_date, end_date)
    
    def _create_sample_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create sample stock data with realistic price movements."""
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        # Generate realistic stock price using geometric Brownian motion
        n_days = len(dates)
        dt = 1/252  # Daily time step
        mu = 0.1  # Annual drift
        sigma = 0.2  # Annual volatility
        S0 = 100  # Initial price
        
        # Generate price path
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
        price_path = S0 * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        stock_data = []
        for i, (date, close) in enumerate(zip(dates, price_path)):
            # Intraday volatility
            daily_vol = 0.02 * np.random.random()
            high = close * (1 + daily_vol)
            low = close * (1 - daily_vol)
            open_price = close * (1 + 0.005 * np.random.randn())
            
            volume = int(1000000 * (1 + 0.5 * np.random.random()))
            
            stock_data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(stock_data, index=dates)
        df.index.name = 'Date'
        
        return df
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        if 'Close' in price_data.columns:
            return price_data['Close'].pct_change().dropna()
        else:
            logger.error("No 'Close' column found in price data")
            return pd.Series()


class DataIntegrator:
    """
    Utility class to integrate data from multiple sources.
    """
    
    def __init__(self):
        """Initialize data integrator."""
        self.news_provider = NewsDataProvider()
        self.esg_provider = ESGDataProvider()
        self.satellite_provider = SatelliteDataProvider()
        self.financial_provider = FinancialDataProvider()
        
    def create_integrated_dataset(self, symbol: str, start_date: str, end_date: str,
                                region: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create integrated dataset with all data sources.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            region: Geographic region for satellite data
            
        Returns:
            Dictionary with integrated datasets
        """
        try:
            logger.info(f"Creating integrated dataset for {symbol} from {start_date} to {end_date}")
            
            # Fetch all data sources
            stock_data = self.financial_provider.fetch_stock_data(symbol, start_date, end_date)
            news_data = self.news_provider.fetch_company_news(symbol, start_date, end_date)
            esg_data = self.esg_provider.fetch_esg_scores(symbol, start_date, end_date)
            
            if region:
                satellite_data = self.satellite_provider.fetch_economic_activity_data(region, start_date, end_date)
            else:
                satellite_data = pd.DataFrame()
            
            # Calculate returns
            returns = self.financial_provider.calculate_returns(stock_data)
            
            integrated_data = {
                'stock_data': stock_data,
                'returns': returns,
                'news_data': news_data,
                'esg_data': esg_data,
                'satellite_data': satellite_data,
                'symbol': symbol,
                'date_range': (start_date, end_date)
            }
            
            logger.info("Successfully created integrated dataset")
            return integrated_data
            
        except Exception as e:
            logger.error(f"Error creating integrated dataset: {e}")
            return {}
    
    def align_data_frequencies(self, datasets: Dict[str, pd.DataFrame], 
                             target_frequency: str = 'D') -> Dict[str, pd.DataFrame]:
        """
        Align different data sources to common frequency.
        
        Args:
            datasets: Dictionary of datasets with different frequencies
            target_frequency: Target frequency ('D', 'W', 'M')
            
        Returns:
            Dictionary with aligned datasets
        """
        try:
            aligned_data = {}
            
            for name, data in datasets.items():
                if data.empty:
                    aligned_data[name] = data
                    continue
                    
                if hasattr(data.index, 'freq') or isinstance(data.index, pd.DatetimeIndex):
                    # Resample to target frequency
                    if target_frequency == 'D':
                        aligned_data[name] = data.resample('D').last().fillna(method='ffill')
                    elif target_frequency == 'W':
                        aligned_data[name] = data.resample('W').last()
                    elif target_frequency == 'M':
                        aligned_data[name] = data.resample('M').last()
                    else:
                        aligned_data[name] = data
                else:
                    aligned_data[name] = data
            
            logger.info(f"Aligned {len(datasets)} datasets to {target_frequency} frequency")
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning data frequencies: {e}")
            return datasets


# Example usage
if __name__ == "__main__":
    # Test data providers
    integrator = DataIntegrator()
    
    # Create sample integrated dataset
    dataset = integrator.create_integrated_dataset(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31',
        region='US_WEST'
    )
    
    print("Integrated Dataset Summary:")
    for name, data in dataset.items():
        if isinstance(data, pd.DataFrame):
            print(f"{name}: {len(data)} records")
        elif isinstance(data, pd.Series):
            print(f"{name}: {len(data)} observations")
        else:
            print(f"{name}: {data}")