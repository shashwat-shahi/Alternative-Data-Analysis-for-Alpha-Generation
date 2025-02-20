# Alternative Data Analysis for Alpha Generation

A comprehensive framework for generating alpha using alternative data sources including sentiment analysis from news and social media, statistical validation with regime detection, and factor models incorporating ESG scores and satellite data.

## Features

### 🔍 Sentiment-Based Trading Signals
- **Transformer Models**: Uses FinBERT and other state-of-the-art models for financial sentiment analysis
- **Multi-Source Data**: Processes news articles, social media posts, and financial text
- **Advanced NLP**: Combines VADER, TextBlob, and transformer models for robust sentiment scoring
- **Signal Generation**: Creates actionable trading signals with confidence measures

### 📊 Statistical Validation Framework
- **Significance Testing**: Comprehensive statistical tests for signal validation
- **Cross-Sectional Analysis**: Validates signals across multiple assets and time periods
- **Information Coefficient**: Calculates rank correlations and predictive power
- **Rolling Validation**: Time series stability analysis with rolling windows

### 🌍 Regime Detection
- **Markov Regime Switching**: Detects market regime changes using statistical models
- **Gaussian Mixture Models**: Alternative regime detection using multiple features
- **Regime-Conditional Validation**: Tests signal performance across different market conditions

### 🌱 ESG Factor Models
- **ESG Score Integration**: Incorporates Environmental, Social, and Governance scores
- **Factor Engineering**: Creates momentum, quality, and consistency factors from ESG data
- **Industry-Relative Analysis**: Generates sector-adjusted ESG factors
- **Predictive Modeling**: Uses regularized regression for alpha generation

### 🛰️ Satellite Data Models
- **Economic Activity**: Nighttime lights and construction activity indicators
- **Agricultural Data**: Vegetation indices and crop monitoring
- **Infrastructure**: Traffic density and port activity analysis
- **PCA Reduction**: Dimensionality reduction for high-dimensional satellite features

## Installation

```bash
# Clone the repository
git clone https://github.com/shashwat-shahi/Alternative-Data-Analysis-for-Alpha-Generation.git
cd Alternative-Data-Analysis-for-Alpha-Generation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.sentiment_analysis import SentimentAnalyzer, TradingSignalGenerator
from src.statistical_validation import AlternativeDataValidator, RegimeDetector
from src.factor_models import ESGFactorModel, SatelliteDataModel, MultiFactorModel
from src.data_sources import DataIntegrator

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
signal_generator = TradingSignalGenerator(sentiment_analyzer)
validator = AlternativeDataValidator()
regime_detector = RegimeDetector()

# Create integrated dataset
integrator = DataIntegrator()
dataset = integrator.create_integrated_dataset(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    region='US_WEST'
)

# Analyze sentiment and generate signals
news_sentiment = sentiment_analyzer.batch_analyze(dataset['news_data']['text'].tolist())
sentiment_metrics = signal_generator.aggregate_sentiment_scores(news_sentiment)
trading_signals = signal_generator.generate_signals(sentiment_metrics, dataset['stock_data'])

# Validate signals
validation_results = validator.test_signal_significance(
    signal=trading_signals['composite_signal'], 
    returns=dataset['returns']
)

print(f"Signal significance: {validation_results['is_significant']}")
print(f"Information Coefficient: {validation_results['information_coefficient']:.3f}")
```

## Project Structure

```
├── src/
│   ├── sentiment_analysis/     # Sentiment analysis and trading signals
│   ├── statistical_validation/ # Statistical validation and regime detection
│   ├── factor_models/         # ESG and satellite data factor models
│   ├── data_sources/          # Data fetching and integration utilities
│   └── utils/                 # Common utilities and visualization
├── config/                    # Configuration files
├── notebooks/                 # Example Jupyter notebooks
├── tests/                     # Unit tests
└── requirements.txt           # Python dependencies
```

## Core Components

### Sentiment Analysis Module

The sentiment analysis module provides comprehensive sentiment scoring from financial text:

- **FinBERT Integration**: Uses ProsusAI/finbert for financial sentiment classification
- **Multi-Model Ensemble**: Combines transformer models with traditional methods
- **Trading Signal Generation**: Converts sentiment scores into actionable trading signals
- **Backtesting Framework**: Historical performance evaluation with risk metrics

### Statistical Validation Framework

Robust statistical validation ensures signal reliability:

- **Significance Testing**: Pearson/Spearman correlations, regression analysis
- **Information Coefficient**: Rank correlation analysis for predictive power
- **Cross-Sectional Validation**: Multi-asset validation across time periods
- **Time Series Stability**: Rolling window analysis for signal consistency

### Factor Models

Advanced factor models for long-term alpha generation:

#### ESG Factor Model
- Creates factors from Environmental, Social, and Governance scores
- Generates momentum, quality, and consistency metrics
- Industry-relative analysis for sector-neutral strategies
- Regularized regression for factor selection

#### Satellite Data Model
- Processes economic activity indicators from satellite imagery
- Incorporates vegetation indices, construction activity, and traffic data
- PCA dimensionality reduction for efficient modeling
- Random forest and linear models for prediction

### Regime Detection

Market regime detection for adaptive strategies:

- **Markov Regime Switching**: Statistical regime identification
- **Gaussian Mixture Models**: Feature-based regime detection
- **Regime-Conditional Validation**: Tests signal performance across regimes
- **Transition Analysis**: Regime duration and transition probability analysis

## Configuration

The framework uses JSON configuration files for easy customization:

```json
{
  "models": {
    "sentiment_analysis": {
      "model_name": "ProsusAI/finbert",
      "use_cuda": true,
      "batch_size": 32
    },
    "esg_factor_model": {
      "standardize": true,
      "method": "ridge",
      "alpha": 1.0
    }
  },
  "validation": {
    "confidence_level": 0.95,
    "min_observations": 100
  }
}
```

## Data Sources

The framework supports multiple alternative data sources:

- **News Data**: Financial news articles and press releases
- **Social Media**: Twitter sentiment and social signals  
- **ESG Data**: Environmental, Social, and Governance scores
- **Satellite Data**: Economic activity from satellite imagery
- **Financial Data**: Price and volume data via yfinance

## Performance Metrics

Comprehensive performance evaluation includes:

- **Return Metrics**: Total return, Sharpe ratio, maximum drawdown
- **Risk Metrics**: Volatility, VaR, tracking error
- **Signal Quality**: Information coefficient, hit rate, signal-to-noise ratio
- **Factor Attribution**: Contribution analysis across factor exposures

## Examples

See the `notebooks/` directory for detailed examples:

- **Sentiment Analysis Tutorial**: Complete walkthrough of sentiment-based signals
- **ESG Factor Analysis**: ESG factor construction and validation
- **Satellite Data Integration**: Processing and modeling satellite data
- **Multi-Factor Model**: Combining all data sources for alpha generation

## Requirements

- Python 3.8+
- PyTorch for transformer models
- Scikit-learn for machine learning
- Pandas/NumPy for data processing
- Statsmodels for statistical analysis
- Plotly/Matplotlib for visualization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This framework is for educational and research purposes. Past performance does not guarantee future results. Always conduct proper due diligence before making investment decisions.