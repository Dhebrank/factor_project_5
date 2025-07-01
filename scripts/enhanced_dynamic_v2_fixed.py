"""
Enhanced Dynamic v2 Fixed: Multi-Signal Regime Detection with Real Data
Fixes the issues identified in debug analysis:
1. Uses real economic data (10-year Treasury from Sharadar)
2. Fixes technical indicators using correct S&P 500 data
3. Adjusts signal weighting to reduce neutral bias
4. Improves signal logic to avoid over-conservative defaults
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDynamicV2Fixed:
    """Fixed Enhanced Dynamic v2 with real data and improved signal logic"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/enhanced_dynamic_v2"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Rebalanced signal weights (reduce economic weight, increase VIX)
        self.signal_weights = {
            'vix_regime': 0.45,        # Increased from 35% - primary signal
            'economic_regime': 0.20,   # Reduced from 30% - supporting signal
            'technical_regime': 0.20,  # Same - technical validation
            'factor_momentum': 0.15    # Same - factor timing
        }
        
        # Load data
        self.load_all_data()
        
        # Enhanced allocation matrices (5 regimes)
        self.allocation_matrices = {
            'crisis':     {'Value': 0.10, 'Quality': 0.45, 'MinVol': 0.40, 'Momentum': 0.05},
            'defensive':  {'Value': 0.15, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.15},
            'neutral':    {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275},  # Base optimized
            'growth':     {'Value': 0.20, 'Quality': 0.25, 'MinVol': 0.20, 'Momentum': 0.35},
            'momentum':   {'Value': 0.15, 'Quality': 0.20, 'MinVol': 0.15, 'Momentum': 0.50}
        }
        
        logger.info("Enhanced Dynamic v2 Fixed Framework initialized")
        
    def load_all_data(self):
        """Load MSCI, market, and real economic data"""
        logger.info("Loading comprehensive dataset with real economic data...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data (VIX, S&P 500)
        self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                     index_col=0, parse_dates=True)
        
        # Load real economic data from Sharadar
        self.load_real_economic_data()
        
        # Combine all datasets
        self.data = pd.concat([
            self.factor_returns, 
            self.market_data, 
            self.economic_data
        ], axis=1)
        
        # Calculate technical indicators using correct S&P 500 data
        self.calculate_fixed_technical_indicators()
        
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"Economic data: {list(self.economic_data.columns)}")
        logger.info(f"Technical indicators calculated with real S&P 500 data")
        
    def load_real_economic_data(self):
        """Load real economic indicators from Sharadar database"""
        logger.info("Loading real economic indicators from Sharadar...")
        
        try:
            # Connect to Sharadar database
            connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
            engine = create_engine(connection_string)
            
            # Query 10-year Treasury data (daily)
            treasury_query = """
            SELECT 
                date,
                value as treasury_10y
            FROM economic_data 
            WHERE series_id = 'DGS10' 
            AND date >= '1998-01-01'
            ORDER BY date
            """
            
            treasury_data = pd.read_sql(treasury_query, engine, parse_dates=['date'])
            treasury_data.set_index('date', inplace=True)
            
            # Convert daily to monthly (end-of-month values)
            treasury_monthly = treasury_data.resample('M').last()
            
            # Create economic indicators dataframe
            self.economic_data = pd.DataFrame(index=self.factor_returns.index)
            
            # Align treasury data with MSCI data dates
            aligned_treasury = treasury_monthly.reindex(self.economic_data.index, method='ffill')
            self.economic_data['treasury_10y'] = aligned_treasury['treasury_10y']
            
            # Calculate derived economic indicators
            self.economic_data['treasury_10y_change'] = self.economic_data['treasury_10y'].pct_change(12)  # YoY change
            self.economic_data['treasury_trend'] = self.economic_data['treasury_10y'].rolling(6).mean()    # 6-month trend
            self.economic_data['treasury_level_regime'] = self.classify_treasury_level(self.economic_data['treasury_10y'])
            
            logger.info(f"Loaded real FRED economic data: {len(self.economic_data.columns)} indicators")
            logger.info(f"Treasury 10Y range: {self.economic_data['treasury_10y'].min():.2f}% to {self.economic_data['treasury_10y'].max():.2f}%")
            
        except Exception as e:
            logger.warning(f"Could not load real economic data: {e}")
            # Create improved dummy data if database unavailable
            self.economic_data = pd.DataFrame(index=self.factor_returns.index)
            # Use VIX as proxy for economic stress (inverted relationship)
            vix_data = self.market_data['VIX'] if 'VIX' in self.market_data.columns else pd.Series(20, index=self.factor_returns.index)
            self.economic_data['treasury_10y'] = 5.0 + (vix_data - 20) * 0.1  # Rough VIX-Treasury relationship
            self.economic_data['treasury_10y_change'] = self.economic_data['treasury_10y'].pct_change(12)
            self.economic_data['treasury_trend'] = self.economic_data['treasury_10y'].rolling(6).mean()
            self.economic_data['treasury_level_regime'] = self.classify_treasury_level(self.economic_data['treasury_10y'])
            logger.info("Using improved dummy economic data based on VIX")
    
    def classify_treasury_level(self, treasury_series):
        """Classify treasury yield levels into regimes"""
        # Historical Treasury 10Y ranges
        conditions = [
            treasury_series < 2.0,    # Very low rates
            treasury_series < 3.5,    # Low rates
            treasury_series < 5.0,    # Normal rates
            treasury_series < 6.5,    # High rates
        ]
        choices = ['very_low', 'low', 'normal', 'high']
        return pd.Series(np.select(conditions, choices, default='very_high'), index=treasury_series.index)
    
    def calculate_fixed_technical_indicators(self):
        """Calculate technical indicators using correct S&P 500 data"""
        logger.info("Calculating technical indicators with fixed S&P 500 data...")
        
        # Use SP500_Adj (adjusted close price) as the primary price series
        if 'SP500_Adj' in self.data.columns:
            sp500_price = self.data['SP500_Adj']
            price_column = 'SP500_Adj'
        elif 'SP500' in self.data.columns:
            sp500_price = self.data['SP500']
            price_column = 'SP500'
        else:
            logger.warning("No S&P 500 price data found")
            return
        
        logger.info(f"Using {price_column} for technical analysis")
        
        # Moving averages
        self.data['sp500_ma_6'] = sp500_price.rolling(6).mean()
        self.data['sp500_ma_12'] = sp500_price.rolling(12).mean()
        self.data['sp500_ma_24'] = sp500_price.rolling(24).mean()
        
        # Trend indicators (price above moving average)
        self.data['sp500_trend_6'] = (sp500_price > self.data['sp500_ma_6']).astype(int)
        self.data['sp500_trend_12'] = (sp500_price > self.data['sp500_ma_12']).astype(int)
        self.data['sp500_trend_24'] = (sp500_price > self.data['sp500_ma_24']).astype(int)
        
        # Momentum indicators
        self.data['sp500_momentum_3'] = sp500_price.pct_change(3)
        self.data['sp500_momentum_6'] = sp500_price.pct_change(6)
        self.data['sp500_momentum_12'] = sp500_price.pct_change(12)
        
        # Volatility using monthly returns
        if 'SP500_Monthly_Return' in self.data.columns:
            returns = self.data['SP500_Monthly_Return']
            self.data['sp500_volatility_6'] = returns.rolling(6).std()
            self.data['sp500_volatility_12'] = returns.rolling(12).std()
        
        logger.info("Technical indicators calculated successfully")
        logger.info(f"Sample trend scores: 6M={self.data['sp500_trend_6'].mean():.2f}, 12M={self.data['sp500_trend_12'].mean():.2f}, 24M={self.data['sp500_trend_24'].mean():.2f}")
    
    def generate_improved_vix_signal(self, date):
        """Generate improved VIX signal with more granular classification"""
        vix = self.data.loc[date, 'VIX']
        
        if vix >= 50:
            signal = {'crisis': 1.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 40:
            signal = {'crisis': 0.7, 'defensive': 0.3, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 30:
            signal = {'crisis': 0.0, 'defensive': 0.8, 'neutral': 0.2, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 25:
            signal = {'crisis': 0.0, 'defensive': 0.4, 'neutral': 0.6, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 20:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.6, 'growth': 0.4, 'momentum': 0.0}
        elif vix >= 15:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.5, 'momentum': 0.2}
        else:  # VIX < 15 (low volatility, potential momentum environment)
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.1, 'growth': 0.4, 'momentum': 0.5}
        
        return signal
    
    def generate_improved_economic_signal(self, date):
        """Generate improved economic signal with real Treasury data"""
        try:
            treasury_level = self.data.loc[date, 'treasury_10y']
            treasury_change = self.data.loc[date, 'treasury_10y_change']
            treasury_regime = self.data.loc[date, 'treasury_level_regime']
            
            if pd.isna(treasury_level) or pd.isna(treasury_change):
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            
            # Economic regime classification based on yield level and change
            if treasury_regime == 'very_high' or (treasury_level > 6.0):
                # Very high rates = crisis/defensive
                signal = {'crisis': 0.3, 'defensive': 0.7, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            elif treasury_regime == 'high' or (treasury_level > 4.5 and treasury_change > 0.5):
                # High rates + rising = defensive
                signal = {'crisis': 0.0, 'defensive': 0.8, 'neutral': 0.2, 'growth': 0.0, 'momentum': 0.0}
            elif treasury_change > 1.0:  # Rapidly rising rates
                signal = {'crisis': 0.1, 'defensive': 0.6, 'neutral': 0.3, 'growth': 0.0, 'momentum': 0.0}
            elif treasury_regime == 'very_low' or treasury_level < 2.0:
                # Very low rates = growth/momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.2, 'growth': 0.4, 'momentum': 0.4}
            elif treasury_change < -0.5:  # Falling rates
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.7, 'momentum': 0.0}
            elif treasury_regime == 'low':
                # Low rates = growth
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.4, 'growth': 0.6, 'momentum': 0.0}
            else:
                # Normal conditions
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.7, 'growth': 0.3, 'momentum': 0.0}
                
            return signal
                
        except Exception as e:
            logger.warning(f"Economic signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_improved_technical_signal(self, date):
        """Generate improved technical signal with fixed S&P 500 data"""
        try:
            # Get technical indicators (now properly calculated)
            trend_6 = self.data.loc[date, 'sp500_trend_6'] if 'sp500_trend_6' in self.data.columns else 0.5
            trend_12 = self.data.loc[date, 'sp500_trend_12'] if 'sp500_trend_12' in self.data.columns else 0.5
            trend_24 = self.data.loc[date, 'sp500_trend_24'] if 'sp500_trend_24' in self.data.columns else 0.5
            
            momentum_3 = self.data.loc[date, 'sp500_momentum_3'] if 'sp500_momentum_3' in self.data.columns else 0
            momentum_6 = self.data.loc[date, 'sp500_momentum_6'] if 'sp500_momentum_6' in self.data.columns else 0
            momentum_12 = self.data.loc[date, 'sp500_momentum_12'] if 'sp500_momentum_12' in self.data.columns else 0
            
            # Handle NaN values with more sophisticated defaults
            if pd.isna(trend_6) or pd.isna(trend_12) or pd.isna(trend_24):
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            
            if pd.isna(momentum_3): momentum_3 = 0.0
            if pd.isna(momentum_6): momentum_6 = 0.0
            if pd.isna(momentum_12): momentum_12 = 0.0
            
            # Technical classification
            trend_score = trend_6 + trend_12 + trend_24  # 0-3 scale
            momentum_score = momentum_3 + momentum_6 + momentum_12
            
            # Improved technical regime classification
            if trend_score >= 3 and momentum_score > 0.15:
                # Strong uptrend + strong momentum = momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.2, 'momentum': 0.8}
            elif trend_score >= 3 and momentum_score > 0.05:
                # Strong uptrend + positive momentum = growth
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.8, 'momentum': 0.2}
            elif trend_score >= 2 and momentum_score > 0.0:
                # Moderate uptrend = growth
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.7, 'momentum': 0.0}
            elif trend_score >= 2:
                # Moderate uptrend but weak momentum = neutral/growth
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.6, 'growth': 0.4, 'momentum': 0.0}
            elif trend_score >= 1 and momentum_score > -0.05:
                # Mixed trends = neutral
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            elif momentum_score < -0.15:
                # Strong negative momentum = crisis/defensive
                signal = {'crisis': 0.4, 'defensive': 0.6, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            elif momentum_score < -0.05:
                # Negative momentum = defensive
                signal = {'crisis': 0.0, 'defensive': 0.8, 'neutral': 0.2, 'growth': 0.0, 'momentum': 0.0}
            else:
                # Weak/unclear conditions = defensive
                signal = {'crisis': 0.0, 'defensive': 0.6, 'neutral': 0.4, 'growth': 0.0, 'momentum': 0.0}
                
            return signal
                
        except Exception as e:
            logger.warning(f"Technical signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_improved_factor_momentum_signal(self, date):
        """Generate improved factor momentum signal"""
        try:
            # Calculate factor momentum scores
            factor_momentum_6 = self.factor_returns.rolling(6).sum().loc[date]
            factor_momentum_12 = self.factor_returns.rolling(12).sum().loc[date]
            
            # Z-score factor momentum vs historical (more stable calculation)
            momentum_zscore = {}
            for factor in self.factor_returns.columns:
                hist_data = self.factor_returns[factor].rolling(36).sum()
                hist_subset = hist_data.loc[:date].dropna()
                
                if len(hist_subset) >= 24:  # Need more history for stability
                    mean_val = hist_subset.mean()
                    std_val = hist_subset.std()
                    if std_val > 0:
                        z_score = (factor_momentum_12[factor] - mean_val) / std_val
                        momentum_zscore[factor] = z_score if not pd.isna(z_score) else 0
                    else:
                        momentum_zscore[factor] = 0
                else:
                    momentum_zscore[factor] = 0
            
            # More sophisticated momentum classification
            strong_momentum = sum(1 for z in momentum_zscore.values() if z > 1.5)
            moderate_momentum = sum(1 for z in momentum_zscore.values() if z > 0.5)
            weak_momentum = sum(1 for z in momentum_zscore.values() if z < -1.5)
            moderate_weak = sum(1 for z in momentum_zscore.values() if z < -0.5)
            
            # Factor momentum regime classification
            if strong_momentum >= 3:  # Very strong factor momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.2, 'momentum': 0.8}
            elif strong_momentum >= 2:  # Strong momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.2, 'growth': 0.3, 'momentum': 0.5}
            elif moderate_momentum >= 3:  # Broad positive momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.7, 'momentum': 0.0}
            elif weak_momentum >= 3:  # Very weak factor momentum
                signal = {'crisis': 0.3, 'defensive': 0.7, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            elif weak_momentum >= 2:  # Moderate weakness
                signal = {'crisis': 0.0, 'defensive': 0.7, 'neutral': 0.3, 'growth': 0.0, 'momentum': 0.0}
            elif moderate_weak >= 3:  # Broad negative momentum
                signal = {'crisis': 0.0, 'defensive': 0.5, 'neutral': 0.5, 'growth': 0.0, 'momentum': 0.0}
            else:  # Mixed or neutral momentum
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.8, 'growth': 0.2, 'momentum': 0.0}
                
            return signal
                
        except Exception as e:
            logger.warning(f"Factor momentum signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_multi_signal_regime_fixed(self, date):
        """Generate final regime using improved weighted multi-signal voting"""
        
        # Generate individual signals (now improved)
        vix_signal = self.generate_improved_vix_signal(date)
        economic_signal = self.generate_improved_economic_signal(date)
        technical_signal = self.generate_improved_technical_signal(date)
        factor_momentum_signal = self.generate_improved_factor_momentum_signal(date)
        
        # Initialize regime votes
        regime_votes = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        
        # Apply rebalanced weighted voting
        for regime in regime_votes.keys():
            regime_votes[regime] += vix_signal[regime] * self.signal_weights['vix_regime']
            regime_votes[regime] += economic_signal[regime] * self.signal_weights['economic_regime']
            regime_votes[regime] += technical_signal[regime] * self.signal_weights['technical_regime']
            regime_votes[regime] += factor_momentum_signal[regime] * self.signal_weights['factor_momentum']
        
        # Select regime with highest vote
        final_regime = max(regime_votes, key=regime_votes.get)
        
        return final_regime, regime_votes
    
    def calculate_enhanced_dynamic_v2_fixed_returns(self):
        """Calculate Enhanced Dynamic v2 Fixed returns using improved multi-signal framework"""
        logger.info("Calculating Enhanced Dynamic v2 Fixed returns...")
        
        portfolio_returns = []
        regime_history = []
        
        for i, date in enumerate(self.factor_returns.index):
            if i < 36:  # Need history for momentum calculations
                allocation = self.allocation_matrices['neutral']
                regime = 'neutral'
            else:
                # Generate improved multi-signal regime
                regime, regime_votes = self.generate_multi_signal_regime_fixed(date)
                allocation = self.allocation_matrices[regime]
            
            # Calculate portfolio return
            month_return = (self.factor_returns.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
            regime_history.append(regime)
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.factor_returns)} periods")
        
        returns_series = pd.Series(portfolio_returns, index=self.factor_returns.index)
        regime_series = pd.Series(regime_history, index=self.factor_returns.index)
        
        # Calculate regime distribution
        regime_distribution = regime_series.value_counts(normalize=True)
        
        logger.info("Fixed multi-signal regime distribution:")
        for regime, percentage in regime_distribution.items():
            logger.info(f"  {regime.capitalize()}: {percentage:.1%}")
        
        return returns_series, regime_series, regime_distribution
    
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """Calculate comprehensive performance metrics"""
        
        # Basic performance metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(12)) if len(returns[returns < 0]) > 0 else sharpe_ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def validate_enhanced_dynamic_v2_fixed(self):
        """Validate Enhanced Dynamic v2 Fixed against baselines"""
        logger.info("🚀 Starting Enhanced Dynamic v2 Fixed Validation...")
        
        # Calculate Enhanced Dynamic v2 Fixed returns
        v2_fixed_returns, regime_series, regime_distribution = self.calculate_enhanced_dynamic_v2_fixed_returns()
        
        # Calculate baseline Enhanced Dynamic (original)
        baseline_returns = self.calculate_baseline_enhanced_dynamic()
        
        # Calculate performance metrics
        v2_fixed_performance = self.calculate_performance_metrics(v2_fixed_returns)
        baseline_performance = self.calculate_performance_metrics(baseline_returns)
        
        # Calculate improvement
        return_improvement = v2_fixed_performance['annual_return'] - baseline_performance['annual_return']
        sharpe_improvement = v2_fixed_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio']
        
        results = {
            'enhanced_dynamic_v2_fixed': {
                'performance': v2_fixed_performance,
                'returns': v2_fixed_returns,
                'regime_distribution': regime_distribution.to_dict()
            },
            'baseline_enhanced_dynamic': {
                'performance': baseline_performance,
                'returns': baseline_returns
            },
            'improvement': {
                'annual_return': return_improvement,
                'sharpe_ratio': sharpe_improvement
            }
        }
        
        logger.info("\\n" + "="*70)
        logger.info("🎯 ENHANCED DYNAMIC V2 FIXED VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"Baseline Enhanced Dynamic: {baseline_performance['annual_return']:.2%} return, {baseline_performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Enhanced Dynamic v2 Fixed: {v2_fixed_performance['annual_return']:.2%} return, {v2_fixed_performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Improvement: {return_improvement:+.2%} return, {sharpe_improvement:+.3f} Sharpe")
        logger.info(f"Regime distribution: {regime_distribution.to_dict()}")
        
        return results
    
    def calculate_baseline_enhanced_dynamic(self):
        """Calculate baseline Enhanced Dynamic returns for comparison"""
        # This replicates the original Enhanced Dynamic strategy
        base_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        # VIX regimes
        vix = self.data['VIX']
        regimes = pd.Series(0, index=vix.index)
        regimes[vix >= 25] = 1
        regimes[vix >= 35] = 2
        regimes[vix >= 50] = 3
        
        # Regime allocations
        regime_allocations = {
            0: base_allocation,  # Normal
            1: base_allocation,  # Elevated 
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis
        }
        
        # Calculate factor momentum
        factor_momentum = self.factor_returns.rolling(12).sum()
        momentum_zscore = factor_momentum.rolling(36).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x) >= 12 else 0
        )
        
        portfolio_returns = []
        for i, date in enumerate(self.factor_returns.index):
            if i < 12:
                allocation = base_allocation
            else:
                regime = regimes.loc[date]
                
                if regime <= 1:  # Normal/Elevated
                    allocation = base_allocation.copy()
                    
                    if i >= 36:  # Apply momentum tilts
                        momentum_scores = momentum_zscore.loc[date]
                        tilt_strength = 0.05
                        for factor in allocation.keys():
                            momentum_tilt = np.clip(momentum_scores[factor] * 0.02, -tilt_strength, tilt_strength)
                            allocation[factor] += momentum_tilt
                        
                        total_weight = sum(allocation.values())
                        allocation = {k: v/total_weight for k, v in allocation.items()}
                        
                elif regime == 2:  # Stress
                    allocation = {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10}
                else:  # Crisis
                    allocation = {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}
            
            month_return = (self.factor_returns.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        return pd.Series(portfolio_returns, index=self.factor_returns.index)

def main():
    """Main execution"""
    validator = EnhancedDynamicV2Fixed()
    results = validator.validate_enhanced_dynamic_v2_fixed()

if __name__ == "__main__":
    main()