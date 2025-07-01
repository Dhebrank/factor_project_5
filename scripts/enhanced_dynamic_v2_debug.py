"""
Enhanced Dynamic v2 Debug: Signal Analysis and Diagnostics
Investigates why multi-signal framework resulted in 100% neutral allocation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDynamicV2Debug:
    """Debug version of Enhanced Dynamic v2 to analyze signal behavior"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/enhanced_dynamic_v2"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Signal weights
        self.signal_weights = {
            'vix_regime': 0.35,
            'economic_regime': 0.30,
            'technical_regime': 0.20,
            'factor_momentum': 0.15
        }
        
        # Load data
        self.load_data()
        
        logger.info("Enhanced Dynamic v2 Debug Framework initialized")
        
    def load_data(self):
        """Load data with diagnostics"""
        logger.info("Loading data for signal diagnostics...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data (VIX, S&P 500)
        self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                     index_col=0, parse_dates=True)
        
        # Combine datasets
        self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
        
        # Add dummy economic data (since FRED failed)
        self.data['treasury_10y'] = 3.0  # Constant dummy
        self.data['treasury_10y_change'] = 0.0
        self.data['treasury_trend'] = 3.0
        
        # Calculate technical indicators
        self.calculate_technical_indicators()
        
        logger.info(f"Data loaded: {len(self.data)} observations")
        logger.info(f"VIX range: {self.data['VIX'].min():.1f} to {self.data['VIX'].max():.1f}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
    def calculate_technical_indicators(self):
        """Calculate technical indicators with diagnostics"""
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
            
            # Volatility
            if 'SP500_Monthly_Return' in self.data.columns:
                returns = self.data['SP500_Monthly_Return']
                self.data['sp500_volatility_6'] = returns.rolling(6).std()
                self.data['sp500_volatility_12'] = returns.rolling(12).std()
                
            logger.info("Technical indicators calculated successfully")
        else:
            logger.warning("SP500_Price not found - technical indicators will be limited")
    
    def debug_vix_signal(self, date):
        """Debug VIX signal with detailed output"""
        vix = self.data.loc[date, 'VIX']
        
        if vix >= 50:
            signal = {'crisis': 1.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            reason = f"VIX {vix:.1f} >= 50 (Crisis)"
        elif vix >= 35:
            signal = {'crisis': 0.0, 'defensive': 1.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            reason = f"VIX {vix:.1f} >= 35 (Defensive)"
        elif vix >= 25:
            signal = {'crisis': 0.0, 'defensive': 0.3, 'neutral': 0.7, 'growth': 0.0, 'momentum': 0.0}
            reason = f"VIX {vix:.1f} >= 25 (Elevated - Mixed)"
        elif vix >= 15:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.7, 'growth': 0.3, 'momentum': 0.0}
            reason = f"VIX {vix:.1f} >= 15 (Normal - Mixed)"
        else:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.4, 'momentum': 0.3}
            reason = f"VIX {vix:.1f} < 15 (Low volatility)"
            
        return signal, reason
    
    def debug_economic_signal(self, date):
        """Debug economic signal with detailed output"""
        treasury_level = self.data.loc[date, 'treasury_10y']
        treasury_change = self.data.loc[date, 'treasury_10y_change']
        
        # Since we're using dummy data (constant 3.0), this will always be neutral
        if pd.isna(treasury_level) or pd.isna(treasury_change):
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            reason = "Economic data unavailable (dummy data)"
        elif treasury_level == 3.0 and treasury_change == 0.0:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            reason = "Dummy economic data - neutral"
        else:
            # Regular logic (won't trigger with dummy data)
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            reason = "Normal economic conditions"
            
        return signal, reason
    
    def debug_technical_signal(self, date):
        """Debug technical signal with detailed output"""
        try:
            # Get technical indicators
            trend_6 = self.data.loc[date, 'sp500_trend_6'] if 'sp500_trend_6' in self.data.columns else 1
            trend_12 = self.data.loc[date, 'sp500_trend_12'] if 'sp500_trend_12' in self.data.columns else 1
            trend_24 = self.data.loc[date, 'sp500_trend_24'] if 'sp500_trend_24' in self.data.columns else 1
            
            momentum_3 = self.data.loc[date, 'sp500_momentum_3'] if 'sp500_momentum_3' in self.data.columns else 0
            momentum_6 = self.data.loc[date, 'sp500_momentum_6'] if 'sp500_momentum_6' in self.data.columns else 0
            
            # Check for NaN values
            if pd.isna(trend_6) or pd.isna(trend_12) or pd.isna(trend_24):
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
                reason = "Technical indicators NaN - default neutral"
                return signal, reason
            
            if pd.isna(momentum_3) or pd.isna(momentum_6):
                momentum_3 = momentum_6 = 0.0
            
            trend_score = trend_6 + trend_12 + trend_24
            momentum_score = momentum_3 + momentum_6
            
            # Technical regime classification
            if trend_score >= 3 and momentum_score > 0.10:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.3, 'momentum': 0.7}
                reason = f"Strong uptrend (score={trend_score}) + momentum ({momentum_score:.3f})"
            elif trend_score >= 2 and momentum_score > 0.0:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.2, 'growth': 0.8, 'momentum': 0.0}
                reason = f"Uptrend (score={trend_score}) + positive momentum ({momentum_score:.3f})"
            elif trend_score >= 1:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Mixed trends (score={trend_score})"
            elif momentum_score < -0.10:
                signal = {'crisis': 0.3, 'defensive': 0.7, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Downtrend + negative momentum ({momentum_score:.3f})"
            else:
                signal = {'crisis': 0.0, 'defensive': 0.8, 'neutral': 0.2, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Weak conditions (trend={trend_score}, momentum={momentum_score:.3f})"
                
        except Exception as e:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            reason = f"Technical signal error: {e}"
            
        return signal, reason
    
    def debug_factor_momentum_signal(self, date):
        """Debug factor momentum signal with detailed output"""
        try:
            # Calculate factor momentum
            factor_momentum_12 = self.factor_returns.rolling(12).sum().loc[date]
            
            # Calculate z-scores
            momentum_zscore = {}
            for factor in self.factor_returns.columns:
                hist_data = self.factor_returns[factor].rolling(36).sum()
                hist_subset = hist_data.loc[:date].dropna()
                
                if len(hist_subset) >= 12:
                    z_score = (factor_momentum_12[factor] - hist_subset.mean()) / hist_subset.std()
                    momentum_zscore[factor] = z_score if not pd.isna(z_score) else 0
                else:
                    momentum_zscore[factor] = 0
            
            # Count strong/weak momentum factors
            strong_momentum = sum(1 for z in momentum_zscore.values() if z > 1.0)
            weak_momentum = sum(1 for z in momentum_zscore.values() if z < -1.0)
            
            # Factor momentum classification
            if strong_momentum >= 3:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.2, 'momentum': 0.8}
                reason = f"Strong momentum: {strong_momentum} factors (z-scores: {momentum_zscore})"
            elif strong_momentum >= 2:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.3, 'growth': 0.4, 'momentum': 0.3}
                reason = f"Moderate momentum: {strong_momentum} strong factors"
            elif weak_momentum >= 3:
                signal = {'crisis': 0.2, 'defensive': 0.8, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Weak momentum: {weak_momentum} weak factors"
            elif weak_momentum >= 2:
                signal = {'crisis': 0.0, 'defensive': 0.6, 'neutral': 0.4, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Some weak momentum: {weak_momentum} weak factors"
            else:
                signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
                reason = f"Mixed momentum: {strong_momentum} strong, {weak_momentum} weak"
                
        except Exception as e:
            signal = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 1.0, 'growth': 0.0, 'momentum': 0.0}
            reason = f"Factor momentum error: {e}"
            
        return signal, reason
    
    def analyze_signal_behavior(self, sample_dates=10):
        """Analyze signal behavior on sample dates"""
        logger.info(f"🔍 Analyzing signal behavior on {sample_dates} sample dates...")
        
        # Select sample dates across the dataset
        total_dates = len(self.data.index)
        sample_indices = np.linspace(50, total_dates-1, sample_dates, dtype=int)  # Skip first 50 for momentum
        sample_dates_list = [self.data.index[i] for i in sample_indices]
        
        analysis_results = []
        
        for date in sample_dates_list:
            logger.info(f"\n--- Signal Analysis for {date.strftime('%Y-%m-%d')} ---")
            
            # Debug each signal
            vix_signal, vix_reason = self.debug_vix_signal(date)
            economic_signal, economic_reason = self.debug_economic_signal(date)
            technical_signal, technical_reason = self.debug_technical_signal(date)
            factor_signal, factor_reason = self.debug_factor_momentum_signal(date)
            
            # Calculate final weighted vote
            regime_votes = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            
            for regime in regime_votes.keys():
                regime_votes[regime] += vix_signal[regime] * self.signal_weights['vix_regime']
                regime_votes[regime] += economic_signal[regime] * self.signal_weights['economic_regime']
                regime_votes[regime] += technical_signal[regime] * self.signal_weights['technical_regime']
                regime_votes[regime] += factor_signal[regime] * self.signal_weights['factor_momentum']
            
            final_regime = max(regime_votes, key=regime_votes.get)
            max_vote = regime_votes[final_regime]
            
            # Log detailed results
            logger.info(f"VIX Signal (35%): {vix_reason}")
            logger.info(f"  Vote: {vix_signal}")
            logger.info(f"Economic Signal (30%): {economic_reason}")
            logger.info(f"  Vote: {economic_signal}")
            logger.info(f"Technical Signal (20%): {technical_reason}")
            logger.info(f"  Vote: {technical_signal}")
            logger.info(f"Factor Signal (15%): {factor_reason}")
            logger.info(f"  Vote: {factor_signal}")
            logger.info(f"Final Regime Votes: {regime_votes}")
            logger.info(f"SELECTED REGIME: {final_regime.upper()} (score: {max_vote:.3f})")
            
            analysis_results.append({
                'date': date,
                'vix_signal': vix_signal,
                'vix_reason': vix_reason,
                'economic_signal': economic_signal,
                'economic_reason': economic_reason,
                'technical_signal': technical_signal,
                'technical_reason': technical_reason,
                'factor_signal': factor_signal,
                'factor_reason': factor_reason,
                'final_votes': regime_votes,
                'selected_regime': final_regime,
                'max_vote': max_vote
            })
        
        return analysis_results
    
    def analyze_regime_distribution(self):
        """Analyze full regime distribution with diagnostics"""
        logger.info("🔍 Analyzing full regime distribution...")
        
        regime_history = []
        
        for i, date in enumerate(self.data.index):
            if i < 36:  # Skip initial periods
                regime_history.append('neutral')
                continue
                
            # Generate signals (without detailed logging)
            vix_signal, _ = self.debug_vix_signal(date)
            economic_signal, _ = self.debug_economic_signal(date)
            technical_signal, _ = self.debug_technical_signal(date)
            factor_signal, _ = self.debug_factor_momentum_signal(date)
            
            # Calculate weighted votes
            regime_votes = {'crisis': 0.0, 'defensive': 0.0, 'neutral': 0.0, 'growth': 0.0, 'momentum': 0.0}
            
            for regime in regime_votes.keys():
                regime_votes[regime] += vix_signal[regime] * self.signal_weights['vix_regime']
                regime_votes[regime] += economic_signal[regime] * self.signal_weights['economic_regime']
                regime_votes[regime] += technical_signal[regime] * self.signal_weights['technical_regime']
                regime_votes[regime] += factor_signal[regime] * self.signal_weights['factor_momentum']
            
            final_regime = max(regime_votes, key=regime_votes.get)
            regime_history.append(final_regime)
        
        # Calculate distribution
        regime_series = pd.Series(regime_history, index=self.data.index)
        regime_distribution = regime_series.value_counts(normalize=True)
        
        logger.info("📊 Full Dataset Regime Distribution:")
        for regime, percentage in regime_distribution.items():
            logger.info(f"  {regime.capitalize()}: {percentage:.1%}")
        
        return regime_series, regime_distribution
    
    def identify_neutral_bias_causes(self):
        """Identify why signals are biased toward neutral"""
        logger.info("🔍 Investigating neutral bias causes...")
        
        # Check VIX distribution
        vix_stats = self.data['VIX'].describe()
        logger.info(f"VIX Statistics: {vix_stats}")
        
        vix_regime_dist = pd.cut(self.data['VIX'], 
                                bins=[0, 15, 25, 35, 50, 100], 
                                labels=['Very Low', 'Low', 'Elevated', 'Stress', 'Crisis']).value_counts()
        logger.info(f"VIX Regime Distribution: {vix_regime_dist}")
        
        # Check economic data
        logger.info(f"Economic data sample: treasury_10y={self.data['treasury_10y'].iloc[100]}, change={self.data['treasury_10y_change'].iloc[100]}")
        
        # Check technical indicators
        if 'sp500_trend_6' in self.data.columns:
            tech_stats = {
                'trend_6_mean': self.data['sp500_trend_6'].mean(),
                'trend_12_mean': self.data['sp500_trend_12'].mean(),
                'trend_24_mean': self.data['sp500_trend_24'].mean(),
                'momentum_3_mean': self.data['sp500_momentum_3'].mean(),
                'momentum_6_mean': self.data['sp500_momentum_6'].mean()
            }
            logger.info(f"Technical indicators: {tech_stats}")
        
        # Check factor momentum
        if len(self.factor_returns) > 36:
            recent_momentum = self.factor_returns.tail(12).sum()
            logger.info(f"Recent factor momentum: {recent_momentum}")

def main():
    """Main debug execution"""
    debugger = EnhancedDynamicV2Debug()
    
    # Analyze signal behavior on sample dates
    sample_analysis = debugger.analyze_signal_behavior(sample_dates=5)
    
    # Analyze full regime distribution
    regime_series, regime_distribution = debugger.analyze_regime_distribution()
    
    # Identify neutral bias causes
    debugger.identify_neutral_bias_causes()
    
    print("\n" + "="*70)
    print("🎯 ENHANCED DYNAMIC V2 DEBUG SUMMARY")
    print("="*70)
    print(f"Final regime distribution: {regime_distribution.to_dict()}")
    print(f"Neutral bias: {regime_distribution.get('neutral', 0):.1%}")

if __name__ == "__main__":
    main()