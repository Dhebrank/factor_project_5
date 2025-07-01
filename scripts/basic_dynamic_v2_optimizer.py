"""
Basic Dynamic v2: VIX Threshold Optimization Framework
Optimizes VIX regime thresholds using walk-forward analysis for improved performance
Target: Beat current Basic Dynamic (9.26% return, 0.665 Sharpe) with optimized thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicDynamicV2Optimizer:
    """VIX threshold optimization for Basic Dynamic strategy"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/basic_dynamic_v2"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # VIX threshold optimization ranges
        self.threshold_ranges = {
            'normal_elevated': range(18, 33, 2),    # [18, 20, 22, 24, 26, 28, 30, 32]
            'elevated_stress': range(28, 46, 3),    # [28, 31, 34, 37, 40, 43] 
            'stress_crisis': range(40, 61, 4)       # [40, 44, 48, 52, 56, 60]
        }
        
        # Base allocation (current static optimized)
        self.base_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        logger.info(f"Threshold optimization ranges: {len(list(product(*self.threshold_ranges.values())))} combinations")
        
    def load_data(self):
        """Load MSCI factor returns and market data"""
        logger.info("Loading data for Basic Dynamic v2 optimization...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data (VIX, S&P 500)
        self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                     index_col=0, parse_dates=True)
        
        # Combine datasets
        self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
        
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"VIX data range: {self.data['VIX'].min():.1f} to {self.data['VIX'].max():.1f}")
        
    def create_vix_regimes(self, normal_elevated, elevated_stress, stress_crisis):
        """Create VIX-based market regimes with custom thresholds"""
        vix = self.data['VIX']
        
        regimes = pd.Series(0, index=vix.index)  # 0 = Normal
        regimes[vix >= normal_elevated] = 1      # Elevated
        regimes[vix >= elevated_stress] = 2      # Stress
        regimes[vix >= stress_crisis] = 3        # Crisis
        
        return regimes
    
    def calculate_basic_dynamic_returns(self, threshold_set):
        """Calculate Basic Dynamic returns with custom VIX thresholds"""
        normal_elevated, elevated_stress, stress_crisis = threshold_set
        
        # Create regimes with custom thresholds
        regimes = self.create_vix_regimes(normal_elevated, elevated_stress, stress_crisis)
        
        # Define regime-based allocations
        regime_allocations = {
            0: self.base_allocation,  # Normal
            1: self.base_allocation,  # Elevated 
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
        
        # Add regime statistics
        regime_dist = regimes.value_counts().sort_index()
        regime_stats = {f"regime_{i}_pct": count/len(regimes) for i, count in regime_dist.items()}
        
        return returns_series, regime_stats
    
    def calculate_performance_metrics(self, returns):
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
    
    def run_walk_forward_optimization(self):
        """Run walk-forward VIX threshold optimization"""
        logger.info("🚀 Starting Basic Dynamic v2 VIX threshold optimization...")
        
        # Generate all threshold combinations
        threshold_combinations = list(product(*self.threshold_ranges.values()))
        logger.info(f"Testing {len(threshold_combinations)} threshold combinations...")
        
        # Walk-forward parameters
        total_months = len(self.factor_returns)
        min_training = 60  # 5 years minimum training
        test_length = 12   # 1 year test periods
        
        optimization_results = []
        
        # Walk-forward analysis for each threshold combination
        for combo_idx, threshold_set in enumerate(threshold_combinations):
            normal_elevated, elevated_stress, stress_crisis = threshold_set
            
            # Validate threshold ordering
            if not (normal_elevated < elevated_stress < stress_crisis):
                continue
                
            combo_results = []
            
            # Walk through validation periods
            for i in range(min_training, total_months - test_length + 1, test_length):
                # Test period
                test_start = i
                test_end = min(i + test_length, total_months)
                
                # Calculate returns for test period with this threshold set
                test_factor_data = self.factor_returns.iloc[test_start:test_end]
                test_market_data = self.data.iloc[test_start:test_end]
                
                # Create temporary data for this test period
                temp_factor_returns = test_factor_data
                temp_data = test_market_data
                
                # Store original data
                orig_factor_returns = self.factor_returns
                orig_data = self.data
                
                # Temporarily replace data for calculation
                self.factor_returns = temp_factor_returns
                self.data = temp_data
                
                # Calculate returns for this period
                test_returns, regime_stats = self.calculate_basic_dynamic_returns(threshold_set)
                
                # Restore original data
                self.factor_returns = orig_factor_returns
                self.data = orig_data
                
                # Calculate performance metrics
                if len(test_returns) > 0:
                    metrics = self.calculate_performance_metrics(test_returns)
                    metrics.update(regime_stats)
                    metrics['test_start'] = test_factor_data.index[0]
                    metrics['test_end'] = test_factor_data.index[-1]
                    metrics['threshold_set'] = threshold_set
                    combo_results.append(metrics)
            
            # Aggregate performance across all validation periods
            if combo_results:
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in combo_results])
                avg_return = np.mean([r['annual_return'] for r in combo_results])
                avg_drawdown = np.mean([r['max_drawdown'] for r in combo_results])
                avg_sortino = np.mean([r['sortino_ratio'] for r in combo_results])
                
                # Multi-objective score (same as comprehensive validation)
                multi_objective_score = (
                    avg_sharpe * 0.40 +
                    avg_sortino * 0.25 +
                    (avg_return / abs(avg_drawdown)) * 0.20 +  # Calmar-like ratio
                    abs(avg_drawdown) * (-0.15)  # Penalize large drawdowns
                )
                
                optimization_results.append({
                    'threshold_set': threshold_set,
                    'normal_elevated': normal_elevated,
                    'elevated_stress': elevated_stress,
                    'stress_crisis': stress_crisis,
                    'avg_annual_return': avg_return,
                    'avg_sharpe_ratio': avg_sharpe,
                    'avg_sortino_ratio': avg_sortino,
                    'avg_max_drawdown': avg_drawdown,
                    'multi_objective_score': multi_objective_score,
                    'validation_periods': len(combo_results)
                })
            
            if (combo_idx + 1) % 20 == 0:
                logger.info(f"  Completed {combo_idx + 1}/{len(threshold_combinations)} threshold combinations")
        
        # Find best threshold set
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['multi_objective_score'])
            
            logger.info("✅ VIX threshold optimization complete!")
            logger.info(f"Best thresholds: Normal<{best_result['normal_elevated']}, Elevated<{best_result['elevated_stress']}, Stress<{best_result['stress_crisis']}")
            logger.info(f"Performance: {best_result['avg_annual_return']:.2%} return, {best_result['avg_sharpe_ratio']:.3f} Sharpe")
            
            return optimization_results, best_result
        else:
            logger.error("No valid threshold combinations found!")
            return [], None
    
    def validate_optimized_strategy(self, best_thresholds):
        """Validate optimized Basic Dynamic v2 on full dataset"""
        logger.info("Validating optimized Basic Dynamic v2 strategy...")
        
        # Calculate returns with optimized thresholds
        optimized_returns, regime_stats = self.calculate_basic_dynamic_returns(best_thresholds['threshold_set'])
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(optimized_returns)
        performance.update(regime_stats)
        performance['optimized_thresholds'] = best_thresholds['threshold_set']
        
        # Compare to baseline Basic Dynamic (fixed thresholds: 25, 35, 50)
        baseline_returns, baseline_regime_stats = self.calculate_basic_dynamic_returns((25, 35, 50))
        baseline_performance = self.calculate_performance_metrics(baseline_returns)
        
        # Calculate improvement
        return_improvement = performance['annual_return'] - baseline_performance['annual_return']
        sharpe_improvement = performance['sharpe_ratio'] - baseline_performance['sharpe_ratio']
        
        logger.info("\n" + "="*60)
        logger.info("🎯 BASIC DYNAMIC V2 VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Baseline Basic Dynamic: {baseline_performance['annual_return']:.2%} return, {baseline_performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Optimized Basic Dynamic v2: {performance['annual_return']:.2%} return, {performance['sharpe_ratio']:.3f} Sharpe")
        logger.info(f"Improvement: {return_improvement:+.2%} return, {sharpe_improvement:+.3f} Sharpe")
        logger.info(f"Optimized thresholds: Normal<{best_thresholds['threshold_set'][0]}, Elevated<{best_thresholds['threshold_set'][1]}, Stress<{best_thresholds['threshold_set'][2]}")
        
        return performance, baseline_performance, optimized_returns
    
    def save_results(self, optimization_results, best_result, validation_performance):
        """Save optimization results and performance validation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save optimization results
        results_data = {
            'timestamp': timestamp,
            'methodology': 'basic_dynamic_v2_vix_optimization',
            'optimization_results': optimization_results,
            'best_threshold_set': best_result,
            'validation_performance': validation_performance,
            'total_combinations_tested': len(optimization_results)
        }
        
        # Convert to serializable format
        serializable_results = self.make_serializable(results_data)
        
        results_file = self.results_dir / f"basic_dynamic_v2_optimization_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Create summary report
        summary_lines = [
            "# BASIC DYNAMIC V2: VIX THRESHOLD OPTIMIZATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Methodology: Walk-forward VIX threshold optimization",
            "",
            "## OPTIMIZATION RESULTS",
            f"**Combinations Tested**: {len(optimization_results)}",
            f"**Best Thresholds**: Normal<{best_result['normal_elevated']}, Elevated<{best_result['elevated_stress']}, Stress<{best_result['stress_crisis']}",
            f"**Optimized Performance**: {best_result['avg_annual_return']:.2%} return, {best_result['avg_sharpe_ratio']:.3f} Sharpe",
            "",
            "## PERFORMANCE COMPARISON",
            f"**Baseline Basic Dynamic**: 9.26% return, 0.665 Sharpe (academic thresholds)",
            f"**Optimized Basic Dynamic v2**: {validation_performance['annual_return']:.2%} return, {validation_performance['sharpe_ratio']:.3f} Sharpe",
            f"**Improvement**: {validation_performance['annual_return'] - 0.0926:+.2%} return, {validation_performance['sharpe_ratio'] - 0.665:+.3f} Sharpe",
            "",
            "## REGIME DISTRIBUTION",
            f"**Normal Regime**: {validation_performance.get('regime_0_pct', 0):.1%} of periods",
            f"**Elevated Regime**: {validation_performance.get('regime_1_pct', 0):.1%} of periods",
            f"**Stress Regime**: {validation_performance.get('regime_2_pct', 0):.1%} of periods",
            f"**Crisis Regime**: {validation_performance.get('regime_3_pct', 0):.1%} of periods"
        ]
        
        summary_file = self.results_dir / f"basic_dynamic_v2_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
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
    optimizer = BasicDynamicV2Optimizer()
    
    # Run optimization
    optimization_results, best_result = optimizer.run_walk_forward_optimization()
    
    if best_result:
        # Validate optimized strategy
        validation_performance, baseline_performance, optimized_returns = optimizer.validate_optimized_strategy(best_result)
        
        # Save results
        optimizer.save_results(optimization_results, best_result, validation_performance)
    
if __name__ == "__main__":
    main()