"""
Enhanced Dynamic v2: Multi-Signal Regime Detection Framework
Implements sophisticated 4-signal regime detection system with optimized signal weighting
Target: Beat current Enhanced Dynamic (9.88% return, 0.719 Sharpe) with multi-signal approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
import psycopg2
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDynamicV2MultiSignal:
    """Multi-signal regime detection for Enhanced Dynamic v2 strategy"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/enhanced_dynamic_v2"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Signal weights (optimized from factor_project_4 learnings)
        self.signal_weights = {
            'vix_regime': 0.35,        # Primary volatility signal
            'economic_regime': 0.30,   # FRED economic indicators  
            'technical_regime': 0.20,  # S&P 500 technical analysis
            'factor_momentum': 0.15    # Factor momentum signals
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
        
        logger.info("Enhanced Dynamic v2 Multi-Signal Framework initialized")
        
    def load_all_data(self):
        """Load MSCI, market, and enhanced economic data"""
        logger.info("Loading comprehensive dataset for Enhanced Dynamic v2...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data (VIX, S&P 500)
        self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                     index_col=0, parse_dates=True)
        
        # Load additional economic data from Sharadar/FRED
        self.load_fred_economic_data()
        
        # Combine all datasets
        self.data = pd.concat([
            self.factor_returns, 
            self.market_data, 
            self.economic_data
        ], axis=1)
        
        # Calculate additional technical indicators
        self.calculate_technical_indicators()
        
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"Total signals: VIX, S&P 500, {len(self.economic_data.columns)} economic indicators")
        
    def load_fred_economic_data(self):
        """Load FRED economic indicators from Sharadar database"""
        logger.info("Loading FRED economic indicators from Sharadar...")
        
        try:
            # Connect to Sharadar database
            connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
            engine = create_engine(connection_string)
            
            # Query FRED economic data (monthly frequency)
            fred_query = """
            SELECT 
                date,
                value as treasury_10y
            FROM fred 
            WHERE ticker = 'DGS10' 
            AND frequency = 'M'
            AND date >= '1998-01-01'
            ORDER BY date
            """
            
            # Load 10-year Treasury (primary economic indicator)
            treasury_data = pd.read_sql(fred_query, engine, parse_dates=['date'])
            treasury_data.set_index('date', inplace=True)
            treasury_data.index = treasury_data.index.to_period('M').to_timestamp('M')
            
            # Create economic indicators dataframe
            self.economic_data = pd.DataFrame(index=self.factor_returns.index)
            
            # Align treasury data with MSCI data dates
            aligned_treasury = treasury_data.reindex(self.economic_data.index, method='ffill')
            self.economic_data['treasury_10y'] = aligned_treasury['treasury_10y']
            
            # Calculate derived economic indicators
            self.economic_data['treasury_10y_change'] = self.economic_data['treasury_10y'].pct_change(12)  # YoY change
            self.economic_data['treasury_trend'] = self.economic_data['treasury_10y'].rolling(6).mean()    # 6-month trend
            
            logger.info(f"Loaded FRED economic data: {len(self.economic_data.columns)} indicators")
            
        except Exception as e:
            logger.warning(f"Could not load FRED data: {e}")
            # Create dummy economic data if database unavailable
            self.economic_data = pd.DataFrame(index=self.factor_returns.index)
            self.economic_data['treasury_10y'] = 3.0  # Constant dummy value
            self.economic_data['treasury_10y_change'] = 0.0
            self.economic_data['treasury_trend'] = 3.0
            logger.info("Using dummy economic data for testing")
    
    def calculate_technical_indicators(self):
        """Calculate S&P 500 technical indicators"""
        logger.info("Calculating technical indicators...")
        
        if 'SP500_Price' in self.data.columns:
            sp500_price = self.data['SP500_Price']
            
            # Moving averages
            self.data['sp500_ma_6'] = sp500_price.rolling(6).mean()
            self.data['sp500_ma_12'] = sp500_price.rolling(12).mean()
            self.data['sp500_ma_24'] = sp500_price.rolling(24).mean()
            
            # Trend indicators
            self.data['sp500_trend_6'] = (sp500_price > self.data['sp500_ma_6']).astype(int)
            self.data['sp500_trend_12'] = (sp500_price > self.data['sp500_ma_12']).astype(int)
            self.data['sp500_trend_24'] = (sp500_price > self.data['sp500_ma_24']).astype(int)
            
            # Momentum indicators
            self.data['sp500_momentum_3'] = sp500_price.pct_change(3)
            self.data['sp500_momentum_6'] = sp500_price.pct_change(6)
            self.data['sp500_momentum_12'] = sp500_price.pct_change(12)
            
            # Volatility (using monthly returns)
            if 'SP500_Monthly_Return' in self.data.columns:
                returns = self.data['SP500_Monthly_Return']
                self.data['sp500_volatility_6'] = returns.rolling(6).std()
                self.data['sp500_volatility_12'] = returns.rolling(12).std()
        
        logger.info("Technical indicators calculated")
    
    def generate_vix_regime_signal(self, date):
        """Generate VIX-based regime signal (35% weight)"""
        vix = self.data.loc[date, 'VIX']
        
        if vix >= 50:
            return {'crisis': 1.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 35:
            return {'crisis': 0.0, 'defensive': 1.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 25:
            return {'crisis': 0.0, 'defensive': 0.3, 'neutral': 0.7, 'growth': 0.0, 'momentum': 0.0}
        elif vix >= 15:
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.7, 'growth': 0.3, 'momentum': 0.0}
        else:
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.4, 'momentum': 0.3}
    
    def generate_economic_regime_signal(self, date):
        """Generate economic regime signal (30% weight)"""
        try:
            # Treasury yield indicators
            treasury_level = self.data.loc[date, 'treasury_10y']
            treasury_change = self.data.loc[date, 'treasury_10y_change']
            
            # Economic regime classification
            if pd.isna(treasury_level) or pd.isna(treasury_change):
                # Default to neutral if data unavailable
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            
            # Rising rates + high level = defensive/crisis
            if treasury_level > 4.5 and treasury_change > 0.5:
                return {'crisis': 0.3, 'defensive': 0.7, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            # High rates but stable = defensive
            elif treasury_level > 4.0:
                return {'crisis': 0.0, 'defensive': 0.7, 'neutral': 0.3, 'growth': 0.0, 'momentum': 0.0}
            # Rising rates moderate = neutral to defensive
            elif treasury_change > 0.3:
                return {'crisis': 0.0, 'defensive': 0.4, 'neutral': 0.6, 'growth': 0.0, 'momentum': 0.0}
            # Low rates = growth/momentum
            elif treasury_level < 2.0:
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.2, 'growth': 0.5, 'momentum': 0.3}
            # Falling rates = growth
            elif treasury_change < -0.3:
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.7, 'momentum': 0.0}
            else:
                # Neutral conditions
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
                
        except Exception as e:
            logger.warning(f"Economic signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_technical_regime_signal(self, date):
        """Generate technical regime signal (20% weight)"""
        try:
            # Technical trend indicators
            trend_6 = self.data.loc[date, 'sp500_trend_6'] if 'sp500_trend_6' in self.data.columns else 1
            trend_12 = self.data.loc[date, 'sp500_trend_12'] if 'sp500_trend_12' in self.data.columns else 1
            trend_24 = self.data.loc[date, 'sp500_trend_24'] if 'sp500_trend_24' in self.data.columns else 1
            
            # Momentum indicators
            momentum_3 = self.data.loc[date, 'sp500_momentum_3'] if 'sp500_momentum_3' in self.data.columns else 0
            momentum_6 = self.data.loc[date, 'sp500_momentum_6'] if 'sp500_momentum_6' in self.data.columns else 0
            
            # Technical regime classification
            trend_score = trend_6 + trend_12 + trend_24  # 0-3 scale
            momentum_score = momentum_3 + momentum_6      # Can be negative
            
            # Strong uptrend + positive momentum = momentum/growth
            if trend_score >= 3 and momentum_score > 0.10:
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.3, 'momentum': 0.7}
            # Uptrend but weak momentum = growth
            elif trend_score >= 2 and momentum_score > 0.0:
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.2, 'growth': 0.8, 'momentum': 0.0}
            # Mixed trends = neutral
            elif trend_score >= 1:
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            # Downtrend + negative momentum = defensive/crisis
            elif momentum_score < -0.10:
                return {'crisis': 0.3, 'defensive': 0.7, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            # Weak downtrend = defensive
            else:
                return {'crisis': 0.0, 'defensive': 0.8, 'neutral': 0.2, 'growth': 0.0, 'momentum': 0.0}
                
        except Exception as e:
            logger.warning(f"Technical signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_factor_momentum_signal(self, date):
        """Generate factor momentum signal (15% weight)"""
        try:
            # Calculate factor momentum scores
            factor_momentum_6 = self.factor_returns.rolling(6).sum().loc[date]
            factor_momentum_12 = self.factor_returns.rolling(12).sum().loc[date]
            
            # Z-score factor momentum vs historical
            momentum_zscore = {}
            for factor in self.factor_returns.columns:
                hist_data = self.factor_returns[factor].rolling(36).sum()
                if len(hist_data.dropna()) > 12:
                    z_score = (factor_momentum_12[factor] - hist_data.mean()) / hist_data.std()
                    momentum_zscore[factor] = z_score if not pd.isna(z_score) else 0
                else:
                    momentum_zscore[factor] = 0
            
            # Strong momentum factors suggest momentum regime
            strong_momentum = sum(1 for z in momentum_zscore.values() if z > 1.0)
            weak_momentum = sum(1 for z in momentum_zscore.values() if z < -1.0)
            
            # Factor momentum regime classification
            if strong_momentum >= 3:  # Most factors showing strong momentum
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.2, 'momentum': 0.8}
            elif strong_momentum >= 2:  # Some strong momentum
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.4, 'momentum': 0.3}
            elif weak_momentum >= 3:  # Most factors weak
                return {'crisis': 0.2, 'defensive': 0.8, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            elif weak_momentum >= 2:  # Some weak momentum
                return {'crisis': 0.0, 'defensive': 0.6, 'neutral': 0.4, 'growth': 0.0, 'momentum': 0.0}
            else:  # Mixed or neutral momentum
                return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
                
        except Exception as e:
            logger.warning(f"Factor momentum signal error at {date}: {e}")
            return {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
    
    def generate_multi_signal_regime(self, date):
        """Generate final regime using weighted multi-signal voting"""
        
        # Generate individual signals
        vix_signal = self.generate_vix_regime_signal(date)
        economic_signal = self.generate_economic_regime_signal(date)
        technical_signal = self.generate_technical_regime_signal(date)
        factor_momentum_signal = self.generate_factor_momentum_signal(date)
        
        # Initialize regime votes
        regime_votes = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
        
        # Apply weighted voting
        for regime in regime_votes.keys():
            regime_votes[regime] += vix_signal[regime] * self.signal_weights['vix_regime']
            regime_votes[regime] += economic_signal[regime] * self.signal_weights['economic_regime']
            regime_votes[regime] += technical_signal[regime] * self.signal_weights['technical_regime']
            regime_votes[regime] += factor_momentum_signal[regime] * self.signal_weights['factor_momentum']
        
        # Select regime with highest vote
        final_regime = max(regime_votes, key=regime_votes.get)
        
        return final_regime, regime_votes
    
    def calculate_enhanced_dynamic_v2_returns(self):
        """Calculate Enhanced Dynamic v2 returns using multi-signal regime detection"""
        logger.info("Calculating Enhanced Dynamic v2 returns with multi-signal framework...")
        
        portfolio_returns = []
        regime_history = []
        
        for i, date in enumerate(self.factor_returns.index):
            if i < 36:  # Need history for momentum calculations
                # Use neutral allocation for initial periods
                allocation = self.allocation_matrices['neutral']
                regime = 'neutral'
            else:
                # Generate multi-signal regime
                regime, regime_votes = self.generate_multi_signal_regime(date)
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
        
        logger.info("Multi-signal regime distribution:")
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
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            beta = np.cov(returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
            alpha = annual_return - beta * (benchmark_returns.mean() * 12)
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(12)
            information_ratio = (annual_return - benchmark_returns.mean() * 12) / tracking_error if tracking_error > 0 else 0
        else:
            beta = alpha = tracking_error = information_ratio = np.nan
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def validate_enhanced_dynamic_v2(self):
        """Validate Enhanced Dynamic v2 against baseline strategies"""
        logger.info("🚀 Starting Enhanced Dynamic v2 Multi-Signal Validation...")
        
        # Calculate Enhanced Dynamic v2 returns
        v2_returns, regime_series, regime_distribution = self.calculate_enhanced_dynamic_v2_returns()
        
        # Calculate baseline Enhanced Dynamic (simple VIX-only)
        baseline_returns = self.calculate_baseline_enhanced_dynamic()
        
        # Calculate performance metrics
        v2_performance = self.calculate_performance_metrics(v2_returns)
        baseline_performance = self.calculate_performance_metrics(baseline_returns)
        
        # Calculate improvement
        return_improvement = v2_performance['annual_return'] - baseline_performance['annual_return']
        sharpe_improvement = v2_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio']
        
        results = {
            'enhanced_dynamic_v2': {
                'performance': v2_performance,
                'returns': v2_returns,
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
        
        logger.info("\n" + "="*70)
        logger.info("🎯 ENHANCED DYNAMIC V2 MULTI-SIGNAL VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"Baseline Enhanced Dynamic: {baseline_performance['annual_return']:.2%} return, {baseline_performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Enhanced Dynamic v2: {v2_performance['annual_return']:.2%} return, {v2_performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Improvement: {return_improvement:+.2%} return, {sharpe_improvement:+.3f} Sharpe")
        logger.info(f"Multi-signal regime framework with {len(regime_distribution)} regime types")
        
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
    
    def save_results(self, results):
        """Save Enhanced Dynamic v2 validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to serializable format
        serializable_results = self.make_serializable(results)
        
        results_file = self.results_dir / f"enhanced_dynamic_v2_validation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Create summary report
        v2_perf = results['enhanced_dynamic_v2']['performance']
        baseline_perf = results['baseline_enhanced_dynamic']['performance']
        improvement = results['improvement']
        regime_dist = results['enhanced_dynamic_v2']['regime_distribution']
        
        summary_lines = [
            "# ENHANCED DYNAMIC V2: MULTI-SIGNAL VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Methodology: Multi-signal regime detection with 4-signal framework",
            "",
            "## MULTI-SIGNAL FRAMEWORK",
            f"**VIX Regime Signal**: 35% weight - Primary volatility detection",
            f"**Economic Regime Signal**: 30% weight - FRED economic indicators",
            f"**Technical Regime Signal**: 20% weight - S&P 500 technical analysis",
            f"**Factor Momentum Signal**: 15% weight - Enhanced factor momentum",
            "",
            "## PERFORMANCE COMPARISON",
            f"**Baseline Enhanced Dynamic**: {baseline_perf['annual_return']:.2%} return, {baseline_perf['sharpe_ratio']:.3f} Sharpe",
            f"**Enhanced Dynamic v2**: {v2_perf['annual_return']:.2%} return, {v2_perf['sharpe_ratio']:.3f} Sharpe",
            f"**Improvement**: {improvement['annual_return']:+.2%} return, {improvement['sharpe_ratio']:+.3f} Sharpe",
            "",
            "## MULTI-SIGNAL REGIME DISTRIBUTION",
        ]
        
        for regime, percentage in regime_dist.items():
            summary_lines.append(f"**{regime.capitalize()} Regime**: {percentage:.1%} of periods")
        
        summary_file = self.results_dir / f"enhanced_dynamic_v2_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write('\\n'.join(summary_lines))
        
        logger.info(f"Results saved: {results_file}")
        logger.info(f"Summary saved: {summary_file}")
        
        return results_file, summary_file
    
    def make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(k): self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return {str(k): self.make_serializable(v) for k, v in obj.to_dict().items()}
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            try:
                return float(obj) if hasattr(obj, '__float__') else str(obj)
            except:
                return str(obj)

def main():
    """Main execution"""
    validator = EnhancedDynamicV2MultiSignal()
    results = validator.validate_enhanced_dynamic_v2()
    validator.save_results(results)

if __name__ == "__main__":
    main()