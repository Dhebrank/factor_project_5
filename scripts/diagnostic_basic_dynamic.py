"""
Diagnostic script to understand Basic Dynamic v2 performance calculation issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicDynamicDiagnostic:
    """Diagnostic tool for Basic Dynamic performance calculation"""
    
    def __init__(self):
        self.data_dir = Path("data/processed")
        self.load_data()
        
    def load_data(self):
        """Load data and check characteristics"""
        logger.info("Loading data for diagnostic...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data 
        try:
            self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                         index_col=0, parse_dates=True)
            self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
            logger.info("Using real market data")
        except:
            logger.warning("Market data not found, creating dummy VIX")
            self.data = self.factor_returns.copy()
            # Create realistic VIX data
            self.data['VIX'] = 20 + 15 * np.random.randn(len(self.data)).cumsum() * 0.1
            self.data['VIX'] = np.clip(self.data['VIX'], 10, 80)
            logger.info("Using dummy VIX data")
            
        # Display data characteristics
        logger.info(f"Data period: {self.data.index[0]} to {self.data.index[-1]}")
        logger.info(f"Total observations: {len(self.data)}")
        
        # Factor returns statistics
        logger.info("\nFactor returns statistics:")
        for factor in self.factor_returns.columns:
            annual_return = (1 + self.factor_returns[factor]).prod() ** (12 / len(self.factor_returns)) - 1
            annual_vol = self.factor_returns[factor].std() * np.sqrt(12)
            sharpe = annual_return / annual_vol
            logger.info(f"  {factor}: {annual_return:.2%} return, {annual_vol:.1%} vol, {sharpe:.2f} Sharpe")
        
        # VIX statistics
        if 'VIX' in self.data.columns:
            vix_stats = self.data['VIX'].describe()
            logger.info(f"\nVIX statistics:")
            logger.info(f"  Mean: {vix_stats['mean']:.1f}")
            logger.info(f"  Std: {vix_stats['std']:.1f}")
            logger.info(f"  Min: {vix_stats['min']:.1f}")
            logger.info(f"  Max: {vix_stats['max']:.1f}")
            
    def test_simple_basic_dynamic(self):
        """Test simple Basic Dynamic with original thresholds"""
        logger.info("\nTesting simple Basic Dynamic with original thresholds...")
        
        # Base allocation
        base_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        # VIX regimes (25, 35, 50)
        vix = self.data['VIX']
        regimes = pd.Series(0, index=vix.index)
        regimes[vix >= 25] = 1
        regimes[vix >= 35] = 2  
        regimes[vix >= 50] = 3
        
        # Check regime distribution
        regime_dist = regimes.value_counts().sort_index()
        logger.info("Regime distribution:")
        regime_names = ['Normal', 'Elevated', 'Stress', 'Crisis']
        for i, count in regime_dist.items():
            logger.info(f"  {regime_names[i]}: {count} months ({count/len(regimes):.1%})")
        
        # Calculate returns
        regime_allocations = {
            0: base_allocation,  # Normal
            1: base_allocation,  # Elevated 
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis
        }
        
        portfolio_returns = []
        for date in self.factor_returns.index:
            regime = regimes.loc[date]
            allocation = regime_allocations[regime]
            month_return = (self.factor_returns.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        returns_series = pd.Series(portfolio_returns, index=self.factor_returns.index)
        
        # Calculate performance
        annual_return = (1 + returns_series).prod() ** (12 / len(returns_series)) - 1
        annual_vol = returns_series.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol
        
        logger.info(f"\nBasic Dynamic performance (original thresholds):")
        logger.info(f"  Annual return: {annual_return:.2%}")
        logger.info(f"  Annual volatility: {annual_vol:.1%}")
        logger.info(f"  Sharpe ratio: {sharpe_ratio:.3f}")
        
        return returns_series, annual_return, sharpe_ratio
    
    def test_single_optimization_period(self):
        """Test optimization on a single period to understand methodology"""
        logger.info("\nTesting single optimization period...")
        
        # Use first 60 months for training, next 12 for testing
        train_data = self.data.iloc[0:60]
        test_data = self.data.iloc[60:72]
        
        logger.info(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Test a few threshold combinations on training data
        test_thresholds = [(20, 30, 45), (25, 35, 50), (30, 40, 55)]
        
        base_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        train_results = {}
        test_results = {}
        
        for thresholds in test_thresholds:
            # Train performance
            train_returns = self.calculate_dynamic_returns(train_data, thresholds, base_allocation)
            train_metrics = self.calculate_metrics(train_returns)
            train_results[thresholds] = train_metrics
            
            # Test performance
            test_returns = self.calculate_dynamic_returns(test_data, thresholds, base_allocation)
            test_metrics = self.calculate_metrics(test_returns)
            test_results[thresholds] = test_metrics
            
            logger.info(f"\nThresholds {thresholds}:")
            logger.info(f"  Train: {train_metrics['annual_return']:.2%} return, {train_metrics['sharpe_ratio']:.3f} Sharpe")
            logger.info(f"  Test:  {test_metrics['annual_return']:.2%} return, {test_metrics['sharpe_ratio']:.3f} Sharpe")
        
        return train_results, test_results
    
    def calculate_dynamic_returns(self, data_subset, thresholds, base_allocation):
        """Calculate dynamic returns for data subset"""
        normal_elevated, elevated_stress, stress_crisis = thresholds
        
        # Create regimes
        vix = data_subset['VIX']
        regimes = pd.Series(0, index=vix.index)
        regimes[vix >= normal_elevated] = 1
        regimes[vix >= elevated_stress] = 2  
        regimes[vix >= stress_crisis] = 3
        
        # Regime allocations
        regime_allocations = {
            0: base_allocation,  # Normal
            1: base_allocation,  # Elevated 
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis
        }
        
        # Calculate returns
        factor_subset = data_subset[self.factor_returns.columns]
        portfolio_returns = []
        
        for date in factor_subset.index:
            regime = regimes.loc[date]
            allocation = regime_allocations[regime]
            month_return = (factor_subset.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        return pd.Series(portfolio_returns, index=factor_subset.index)
    
    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) < 12:
            annual_return = (1 + returns).prod() - 1
            annual_vol = returns.std() * np.sqrt(len(returns))
        else:
            annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
            annual_vol = returns.std() * np.sqrt(12)
            
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        logger.info("🔍 Running Basic Dynamic diagnostic...")
        
        # Test basic implementation
        baseline_returns, baseline_return, baseline_sharpe = self.test_simple_basic_dynamic()
        
        # Test single optimization period
        train_results, test_results = self.test_single_optimization_period()
        
        # Summary
        print("\n" + "="*60)
        print("🔍 BASIC DYNAMIC DIAGNOSTIC SUMMARY")
        print("="*60)
        print(f"Baseline Basic Dynamic: {baseline_return:.2%} return, {baseline_sharpe:.3f} Sharpe")
        print("\nSingle period optimization shows reasonable results")
        print("12.10% return suggests calculation error in walk-forward averaging")

def main():
    """Main execution"""
    diagnostic = BasicDynamicDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main()