"""
Long-Term MSCI Factor Validation Framework
Tests three strategic approaches over 26.5 years of MSCI factor data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MSCILongTermValidator:
    """Validate factor strategies using 26.5 years of MSCI data"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/long_term_performance"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Strategy allocations to test
        self.allocations = {
            'original_static': {'Value': 0.25, 'Quality': 0.30, 'MinVol': 0.20, 'Momentum': 0.25},
            'optimized_static': {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275},
            'equal_weight': {'Value': 0.25, 'Quality': 0.25, 'MinVol': 0.25, 'Momentum': 0.25}
        }
        
    def load_data(self):
        """Load processed MSCI data"""
        logger.info("Loading MSCI factor data...")
        
        try:
            # Load returns data
            returns_file = self.data_dir / "msci_factor_returns.csv"
            self.returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            
            # Load metadata
            metadata_file = self.data_dir / "msci_data_metadata.json"
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded data: {len(self.returns)} months from {self.returns.index.min()} to {self.returns.index.max()}")
            logger.info(f"Factors: {list(self.returns.columns)}")
            
            # Basic statistics
            annual_returns = self.returns.mean() * 12
            annual_vol = self.returns.std() * np.sqrt(12)
            
            logger.info("MSCI Factor Performance (Annualized):")
            for factor in self.returns.columns:
                logger.info(f"  {factor}: {annual_returns[factor]:.2%} return, {annual_vol[factor]:.2%} volatility")
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_vix_proxy(self):
        """Create VIX proxy for regime detection (simplified for MSCI data)"""
        # For MSCI monthly data, we'll use MinVol factor volatility as a market stress proxy
        # Higher MinVol relative performance suggests lower market stress
        # Lower MinVol relative performance suggests higher market stress
        
        minvol_rolling_vol = self.returns['MinVol'].rolling(12).std()
        market_stress = minvol_rolling_vol / minvol_rolling_vol.rolling(36).mean() 
        
        # Create regime thresholds based on historical distribution
        stress_75 = market_stress.quantile(0.75)
        stress_90 = market_stress.quantile(0.90)
        stress_95 = market_stress.quantile(0.95)
        
        # Define regimes
        regimes = pd.Series(0, index=market_stress.index)  # Normal
        regimes[market_stress > stress_75] = 1  # Elevated
        regimes[market_stress > stress_90] = 2  # Stress  
        regimes[market_stress > stress_95] = 3  # Crisis
        
        self.market_regimes = regimes
        self.market_stress = market_stress
        
        logger.info("Market regime proxy created:")
        regime_counts = regimes.value_counts().sort_index()
        regime_names = ['Normal', 'Elevated', 'Stress', 'Crisis']
        for i, count in regime_counts.items():
            logger.info(f"  {regime_names[i]}: {count} months ({count/len(regimes):.1%})")
    
    def run_static_strategy(self, allocation_name, weights):
        """Run static allocation strategy"""
        logger.info(f"Running static strategy: {allocation_name}")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        annual_return = (1 + portfolio_returns).prod() ** (12/len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        positive_months = (portfolio_returns > 0).sum()
        win_rate = positive_months / len(portfolio_returns)
        
        results = {
            'strategy': f'Static_{allocation_name}',
            'allocation': weights,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_months': len(portfolio_returns),
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative
        }
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return results
    
    def run_basic_dynamic_strategy(self, base_allocation):
        """Run basic dynamic strategy with regime-based allocation"""
        logger.info("Running basic dynamic strategy...")
        
        # Define regime-based allocations (defensive during stress)
        regime_allocations = {
            0: base_allocation,  # Normal - base allocation
            1: base_allocation,  # Elevated - base allocation
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress - defensive
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis - very defensive
        }
        
        portfolio_returns = []
        allocations_used = []
        
        for date, regime in self.market_regimes.items():
            if date in self.returns.index:
                # Get allocation for current regime
                current_allocation = regime_allocations.get(regime, base_allocation)
                allocations_used.append((date, regime, current_allocation))
                
                # Calculate portfolio return for this month
                month_return = (self.returns.loc[date] * pd.Series(current_allocation)).sum()
                portfolio_returns.append(month_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=self.returns.index)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        annual_return = (1 + portfolio_returns).prod() ** (12/len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        results = {
            'strategy': 'Basic_Dynamic',
            'base_allocation': base_allocation,
            'regime_allocations': regime_allocations,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_months': len(portfolio_returns),
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative,
            'allocations_used': allocations_used
        }
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return results
    
    def run_enhanced_dynamic_strategy(self, base_allocation):
        """Run enhanced dynamic strategy with momentum-based adjustments"""
        logger.info("Running enhanced dynamic strategy...")
        
        # Calculate factor momentum (12-month return)
        factor_momentum = self.returns.rolling(12).sum()
        
        portfolio_returns = []
        allocations_used = []
        
        for i, date in enumerate(self.returns.index):
            if i < 12:  # Need 12 months of history
                current_allocation = base_allocation
            else:
                # Get current regime
                regime = self.market_regimes.loc[date]
                
                # Base allocation from regime
                if regime <= 1:  # Normal/Elevated
                    regime_allocation = base_allocation
                elif regime == 2:  # Stress
                    regime_allocation = {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10}
                else:  # Crisis
                    regime_allocation = {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}
                
                # Momentum-based adjustments (only in normal/elevated regimes)
                if regime <= 1:
                    momentum_scores = factor_momentum.loc[date]
                    momentum_rank = momentum_scores.rank(ascending=False)
                    
                    # Tilt toward top momentum factors (5% max tilt)
                    tilt_amount = 0.025  # 2.5% tilt per factor
                    current_allocation = regime_allocation.copy()
                    
                    for factor in momentum_rank.index:
                        if momentum_rank[factor] <= 2:  # Top 2 factors
                            current_allocation[factor] += tilt_amount
                        elif momentum_rank[factor] >= 3:  # Bottom 2 factors  
                            current_allocation[factor] -= tilt_amount
                    
                    # Normalize to sum to 1
                    total_weight = sum(current_allocation.values())
                    current_allocation = {k: v/total_weight for k, v in current_allocation.items()}
                else:
                    current_allocation = regime_allocation
            
            allocations_used.append((date, current_allocation))
            
            # Calculate portfolio return
            month_return = (self.returns.loc[date] * pd.Series(current_allocation)).sum()
            portfolio_returns.append(month_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=self.returns.index)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        annual_return = (1 + portfolio_returns).prod() ** (12/len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
        
        results = {
            'strategy': 'Enhanced_Dynamic',
            'base_allocation': base_allocation,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_months': len(portfolio_returns),
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative,
            'allocations_used': allocations_used
        }
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return results
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        logger.info("🚀 Starting comprehensive MSCI validation...")
        
        if not self.load_data():
            return None
        
        # Create market regime proxy
        self.create_vix_proxy()
        
        all_results = {}
        
        # Test all static strategies
        for name, weights in self.allocations.items():
            results = self.run_static_strategy(name, weights)
            all_results[f'static_{name}'] = results
        
        # Test dynamic strategies (using optimized allocation as base)
        base_allocation = self.allocations['optimized_static']
        
        basic_dynamic = self.run_basic_dynamic_strategy(base_allocation)
        all_results['basic_dynamic'] = basic_dynamic
        
        enhanced_dynamic = self.run_enhanced_dynamic_strategy(base_allocation)
        all_results['enhanced_dynamic'] = enhanced_dynamic
        
        # Summary comparison
        self.create_performance_summary(all_results)
        
        # Save results
        self.save_results(all_results)
        
        logger.info("✅ Comprehensive validation completed")
        return all_results
    
    def create_performance_summary(self, results):
        """Create performance comparison summary"""
        logger.info("\n📊 PERFORMANCE SUMMARY (26.5 Years):")
        logger.info("="*60)
        
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Annual Return': f"{result['annual_return']:.2%}",
                'Volatility': f"{result['annual_volatility']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Win Rate': f"{result['win_rate']:.1%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by Sharpe ratio
        summary_df['Sharpe_Numeric'] = [results[name]['sharpe_ratio'] for name in results.keys()]
        summary_df = summary_df.sort_values('Sharpe_Numeric', ascending=False)
        summary_df = summary_df.drop('Sharpe_Numeric', axis=1)
        
        print(summary_df.to_string(index=False))
        
        # Find best performing strategy
        best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
        logger.info(f"\n🏆 BEST STRATEGY: {best_strategy}")
        logger.info(f"Sharpe Ratio: {results[best_strategy]['sharpe_ratio']:.3f}")
        
        return summary_df
    
    def save_results(self, results):
        """Save validation results"""
        logger.info("Saving validation results...")
        
        # Save performance summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for strategy_name, result in results.items():
            serializable_result = result.copy()
            # Convert pandas objects to serializable formats
            serializable_result['portfolio_returns'] = {str(k): v for k, v in result['portfolio_returns'].to_dict().items()}
            serializable_result['cumulative_returns'] = {str(k): v for k, v in result['cumulative_returns'].to_dict().items()}
            
            if 'allocations_used' in result:
                # Convert allocation history to serializable format
                serializable_result['allocations_used'] = [
                    (str(date), allocation) for date, *allocation in result['allocations_used']
                ]
            
            serializable_results[strategy_name] = serializable_result
        
        # Save JSON results
        results_file = self.results_dir / f"msci_validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Saved results: {results_file}")
        
        # Save CSV summary
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                'strategy': strategy_name,
                'annual_return': result['annual_return'],
                'annual_volatility': result['annual_volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'total_months': result['total_months']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"msci_performance_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary: {summary_file}")
        
        return results_file, summary_file

def main():
    """Main execution"""
    validator = MSCILongTermValidator()
    results = validator.run_comprehensive_validation()
    
    if results:
        print("\n🎯 MSCI Long-Term Validation Complete!")
        print("Results saved to results/long_term_performance/")

if __name__ == "__main__":
    main()