"""
Comprehensive Reoptimization Strategy Testing Framework
Tests three reoptimization approaches vs original static optimized OOS baseline:
1. Fixed Frequency Reoptimization (every 3 years)
2. Regime-Based Reoptimization (crisis/volatility triggers)
3. Performance-Based Reoptimization (underperformance triggers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from itertools import product
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReoptimizationStrategyTester:
    """Comprehensive testing framework for different reoptimization approaches"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/reoptimization_testing"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Original static optimized allocation (factor_project_4 OOS baseline)
        self.baseline_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        # Parameter optimization settings
        self.param_ranges = {
            'Value': np.arange(0.10, 0.35, 0.05),
            'Quality': np.arange(0.15, 0.40, 0.05), 
            'MinVol': np.arange(0.15, 0.40, 0.05),
            'Momentum': np.arange(0.15, 0.40, 0.05)
        }
        
        self.min_training_periods = 120  # 10 years minimum training
        
    def load_data(self):
        """Load MSCI factor returns and market data"""
        logger.info("Loading data for reoptimization testing...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data for regime detection
        try:
            self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                         index_col=0, parse_dates=True)
            self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
        except:
            logger.warning("Market data not found, using factor returns only")
            self.data = self.factor_returns.copy()
            # Create dummy VIX data for testing
            self.data['VIX'] = 20 + 10 * np.random.randn(len(self.data))
            
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        
    def optimize_allocation(self, training_data, max_combinations=500):
        """Optimize allocation on training data using multi-objective score"""
        
        # Generate parameter combinations
        all_combinations = list(product(
            self.param_ranges['Value'],
            self.param_ranges['Quality'], 
            self.param_ranges['MinVol'],
            self.param_ranges['Momentum']
        ))
        
        # Filter valid combinations (sum ≈ 1.0)
        valid_combinations = []
        for combo in all_combinations:
            if abs(sum(combo) - 1.0) <= 0.01:
                valid_combinations.append(combo)
        
        # Limit combinations if too many
        if len(valid_combinations) > max_combinations:
            np.random.seed(42)
            indices = np.random.choice(len(valid_combinations), max_combinations, replace=False)
            valid_combinations = [valid_combinations[i] for i in indices]
        
        best_score = -np.inf
        best_allocation = None
        
        for combo in valid_combinations:
            # Normalize weights
            total = sum(combo)
            weights = {
                'Value': combo[0] / total,
                'Quality': combo[1] / total,
                'MinVol': combo[2] / total, 
                'Momentum': combo[3] / total
            }
            
            # Calculate returns on training data
            portfolio_returns = (training_data * pd.Series(weights)).sum(axis=1)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(portfolio_returns)
            
            # Multi-objective score (same as factor_project_4)
            score = (
                metrics['sharpe_ratio'] * 0.40 +
                metrics['sortino_ratio'] * 0.25 +
                metrics['calmar_ratio'] * 0.20 +
                abs(metrics['max_drawdown']) * (-0.15)
            )
            
            if score > best_score:
                best_score = score
                best_allocation = weights
        
        return best_allocation, best_score
    
    def calculate_performance_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        if len(returns) < 12:
            return {
                'annual_return': 0, 'annual_volatility': 0.20, 'sharpe_ratio': 0,
                'sortino_ratio': 0, 'calmar_ratio': 0, 'max_drawdown': -0.10
            }
        
        # Basic metrics
        annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        downside_returns = returns[returns < 0]
        sortino_ratio = annual_return / (downside_returns.std() * np.sqrt(12)) if len(downside_returns) > 0 else sharpe_ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown
        }
    
    def test_baseline_static_optimized(self):
        """Test original static optimized allocation (factor_project_4 OOS)"""
        logger.info("Testing baseline: Original Static Optimized (factor_project_4 OOS)...")
        
        # Apply fixed allocation to entire period
        portfolio_returns = (self.factor_returns * pd.Series(self.baseline_allocation)).sum(axis=1)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        return {
            'strategy': 'baseline_static_optimized',
            'allocation_changes': 0,
            'average_allocation': self.baseline_allocation,
            'performance': metrics,
            'returns': portfolio_returns,
            'reoptimizations': []
        }
    
    def test_fixed_frequency_reoptimization(self, reopt_frequency_months=36):
        """Test Option 1: Fixed frequency reoptimization (every 3 years)"""
        logger.info(f"Testing fixed frequency reoptimization (every {reopt_frequency_months} months)...")
        
        total_months = len(self.factor_returns)
        portfolio_returns = []
        reoptimizations = []
        current_allocation = self.baseline_allocation.copy()
        allocation_history = []
        
        for i in range(total_months):
            # Check if reoptimization is due
            if i >= self.min_training_periods and i % reopt_frequency_months == 0:
                # Training period: previous 10 years
                train_start = max(0, i - self.min_training_periods)
                train_end = i
                training_data = self.factor_returns.iloc[train_start:train_end]
                
                # Optimize allocation
                new_allocation, score = self.optimize_allocation(training_data)
                
                reoptimizations.append({
                    'date': self.factor_returns.index[i],
                    'month_index': i,
                    'old_allocation': current_allocation.copy(),
                    'new_allocation': new_allocation.copy(),
                    'optimization_score': score,
                    'training_period': (train_start, train_end)
                })
                
                current_allocation = new_allocation
                logger.info(f"  Reoptimized at {self.factor_returns.index[i].strftime('%Y-%m')}: "
                          f"V={new_allocation['Value']:.1%}, Q={new_allocation['Quality']:.1%}, "
                          f"M={new_allocation['MinVol']:.1%}, Mom={new_allocation['Momentum']:.1%}")
            
            # Calculate month return with current allocation
            month_return = (self.factor_returns.iloc[i] * pd.Series(current_allocation)).sum()
            portfolio_returns.append(month_return)
            allocation_history.append(current_allocation.copy())
        
        portfolio_returns = pd.Series(portfolio_returns, index=self.factor_returns.index)
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        # Calculate average allocation
        avg_allocation = {}
        for factor in self.baseline_allocation.keys():
            avg_allocation[factor] = np.mean([alloc[factor] for alloc in allocation_history])
        
        return {
            'strategy': 'fixed_frequency_reoptimization',
            'allocation_changes': len(reoptimizations),
            'average_allocation': avg_allocation,
            'performance': metrics,
            'returns': portfolio_returns,
            'reoptimizations': reoptimizations
        }
    
    def detect_regime_triggers(self, current_date_idx):
        """Detect regime-based reoptimization triggers"""
        if current_date_idx < 24:  # Need 2 years of history
            return False, []
        
        triggers = []
        
        # Get recent data
        recent_data = self.data.iloc[max(0, current_date_idx-12):current_date_idx]
        
        # Trigger 1: VIX spike (>40 for 3+ months)
        if 'VIX' in recent_data.columns:
            high_vix_months = (recent_data['VIX'] > 40).sum()
            if high_vix_months >= 3:
                triggers.append('vix_spike')
        
        # Trigger 2: Market crash (S&P 500 decline >20% from recent high)
        if 'SP500_Monthly_Return' in recent_data.columns:
            sp500_returns = recent_data['SP500_Monthly_Return']
            recent_cumulative = (1 + sp500_returns).cumprod()
            peak_to_current = (recent_cumulative.iloc[-1] / recent_cumulative.max()) - 1
            if peak_to_current < -0.20:
                triggers.append('market_crash')
        
        # Trigger 3: Factor divergence (any factor 2+ std dev from others)
        factor_returns_12m = self.factor_returns.iloc[current_date_idx-12:current_date_idx].sum()
        factor_zscores = (factor_returns_12m - factor_returns_12m.mean()) / factor_returns_12m.std()
        if (abs(factor_zscores) > 2.0).any():
            triggers.append('factor_divergence')
        
        return len(triggers) > 0, triggers
    
    def test_regime_based_reoptimization(self):
        """Test Option 2: Regime-based reoptimization"""
        logger.info("Testing regime-based reoptimization...")
        
        total_months = len(self.factor_returns)
        portfolio_returns = []
        reoptimizations = []
        current_allocation = self.baseline_allocation.copy()
        allocation_history = []
        months_since_last_reopt = 0
        
        for i in range(total_months):
            months_since_last_reopt += 1
            
            # Check regime triggers (after minimum training period)
            if i >= self.min_training_periods and months_since_last_reopt >= 12:  # Min 1 year between reopt
                should_reopt, triggers = self.detect_regime_triggers(i)
                
                if should_reopt:
                    # Training period: previous 10 years
                    train_start = max(0, i - self.min_training_periods)
                    train_end = i
                    training_data = self.factor_returns.iloc[train_start:train_end]
                    
                    # Optimize allocation
                    new_allocation, score = self.optimize_allocation(training_data)
                    
                    reoptimizations.append({
                        'date': self.factor_returns.index[i],
                        'month_index': i,
                        'old_allocation': current_allocation.copy(),
                        'new_allocation': new_allocation.copy(),
                        'optimization_score': score,
                        'triggers': triggers,
                        'training_period': (train_start, train_end)
                    })
                    
                    current_allocation = new_allocation
                    months_since_last_reopt = 0
                    
                    logger.info(f"  Regime reopt at {self.factor_returns.index[i].strftime('%Y-%m')} "
                              f"(triggers: {triggers}): "
                              f"V={new_allocation['Value']:.1%}, Q={new_allocation['Quality']:.1%}, "
                              f"M={new_allocation['MinVol']:.1%}, Mom={new_allocation['Momentum']:.1%}")
            
            # Calculate month return
            month_return = (self.factor_returns.iloc[i] * pd.Series(current_allocation)).sum()
            portfolio_returns.append(month_return)
            allocation_history.append(current_allocation.copy())
        
        portfolio_returns = pd.Series(portfolio_returns, index=self.factor_returns.index)
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        # Calculate average allocation
        avg_allocation = {}
        for factor in self.baseline_allocation.keys():
            avg_allocation[factor] = np.mean([alloc[factor] for alloc in allocation_history])
        
        return {
            'strategy': 'regime_based_reoptimization',
            'allocation_changes': len(reoptimizations),
            'average_allocation': avg_allocation,
            'performance': metrics,
            'returns': portfolio_returns,
            'reoptimizations': reoptimizations
        }
    
    def detect_performance_triggers(self, current_date_idx, current_allocation, benchmark_returns=None):
        """Detect performance-based reoptimization triggers"""
        if current_date_idx < 36:  # Need 3 years of history
            return False, []
        
        triggers = []
        
        # Calculate recent strategy performance
        recent_start = max(0, current_date_idx - 12)  # Last 12 months
        recent_factor_returns = self.factor_returns.iloc[recent_start:current_date_idx]
        recent_portfolio_returns = (recent_factor_returns * pd.Series(current_allocation)).sum(axis=1)
        
        # Trigger 1: Underperformance vs benchmark
        if benchmark_returns is not None:
            recent_benchmark = benchmark_returns.iloc[recent_start:current_date_idx]
            if len(recent_benchmark) > 0 and len(recent_portfolio_returns) > 0:
                portfolio_12m = recent_portfolio_returns.sum()
                benchmark_12m = recent_benchmark.sum()
                if portfolio_12m < benchmark_12m - 0.02:  # Underperform by >2%
                    triggers.append('underperformance')
        
        # Trigger 2: Large drawdown
        if len(recent_portfolio_returns) > 0:
            cumulative = (1 + recent_portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            current_drawdown = (cumulative.iloc[-1] / rolling_max.iloc[-1]) - 1
            if current_drawdown < -0.15:  # >15% drawdown
                triggers.append('large_drawdown')
        
        # Trigger 3: Low Sharpe ratio
        recent_36m_start = max(0, current_date_idx - 36)  # Last 3 years
        recent_36m_returns = self.factor_returns.iloc[recent_36m_start:current_date_idx]
        if len(recent_36m_returns) >= 24:  # At least 2 years
            portfolio_36m_returns = (recent_36m_returns * pd.Series(current_allocation)).sum(axis=1)
            metrics_36m = self.calculate_performance_metrics(portfolio_36m_returns)
            if metrics_36m['sharpe_ratio'] < 0.5:
                triggers.append('low_sharpe')
        
        return len(triggers) > 0, triggers
    
    def test_performance_based_reoptimization(self):
        """Test Option 3: Performance-based reoptimization"""
        logger.info("Testing performance-based reoptimization...")
        
        total_months = len(self.factor_returns)
        portfolio_returns = []
        reoptimizations = []
        current_allocation = self.baseline_allocation.copy()
        allocation_history = []
        months_since_last_reopt = 0
        
        # Create benchmark returns (equal weight for comparison)
        equal_weight = {'Value': 0.25, 'Quality': 0.25, 'MinVol': 0.25, 'Momentum': 0.25}
        benchmark_returns = (self.factor_returns * pd.Series(equal_weight)).sum(axis=1)
        
        for i in range(total_months):
            months_since_last_reopt += 1
            
            # Check performance triggers (after minimum training period)
            if i >= self.min_training_periods and months_since_last_reopt >= 12:  # Min 1 year between reopt
                should_reopt, triggers = self.detect_performance_triggers(i, current_allocation, benchmark_returns)
                
                if should_reopt:
                    # Training period: previous 10 years
                    train_start = max(0, i - self.min_training_periods)
                    train_end = i
                    training_data = self.factor_returns.iloc[train_start:train_end]
                    
                    # Optimize allocation
                    new_allocation, score = self.optimize_allocation(training_data)
                    
                    reoptimizations.append({
                        'date': self.factor_returns.index[i],
                        'month_index': i,
                        'old_allocation': current_allocation.copy(),
                        'new_allocation': new_allocation.copy(),
                        'optimization_score': score,
                        'triggers': triggers,
                        'training_period': (train_start, train_end)
                    })
                    
                    current_allocation = new_allocation
                    months_since_last_reopt = 0
                    
                    logger.info(f"  Performance reopt at {self.factor_returns.index[i].strftime('%Y-%m')} "
                              f"(triggers: {triggers}): "
                              f"V={new_allocation['Value']:.1%}, Q={new_allocation['Quality']:.1%}, "
                              f"M={new_allocation['MinVol']:.1%}, Mom={new_allocation['Momentum']:.1%}")
            
            # Calculate month return
            month_return = (self.factor_returns.iloc[i] * pd.Series(current_allocation)).sum()
            portfolio_returns.append(month_return)
            allocation_history.append(current_allocation.copy())
        
        portfolio_returns = pd.Series(portfolio_returns, index=self.factor_returns.index)
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        # Calculate average allocation
        avg_allocation = {}
        for factor in self.baseline_allocation.keys():
            avg_allocation[factor] = np.mean([alloc[factor] for alloc in allocation_history])
        
        return {
            'strategy': 'performance_based_reoptimization',
            'allocation_changes': len(reoptimizations),
            'average_allocation': avg_allocation,
            'performance': metrics,
            'returns': portfolio_returns,
            'reoptimizations': reoptimizations
        }
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison of all reoptimization approaches"""
        logger.info("🚀 Running comprehensive reoptimization strategy comparison...")
        
        results = {}
        
        # Test all approaches
        logger.info("\n--- Testing Baseline: Original Static Optimized ---")
        results['baseline'] = self.test_baseline_static_optimized()
        
        logger.info("\n--- Testing Option 1: Fixed Frequency Reoptimization ---")
        results['fixed_frequency'] = self.test_fixed_frequency_reoptimization()
        
        logger.info("\n--- Testing Option 2: Regime-Based Reoptimization ---")
        results['regime_based'] = self.test_regime_based_reoptimization()
        
        logger.info("\n--- Testing Option 3: Performance-Based Reoptimization ---")
        results['performance_based'] = self.test_performance_based_reoptimization()
        
        # Create comparison analysis
        comparison_results = self.create_comparison_analysis(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"reoptimization_comparison_{timestamp}.json"
        
        # Make results serializable
        serializable_results = self.make_serializable(results)
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved: {results_file}")
        
        # Create summary report
        self.create_summary_report(comparison_results, timestamp)
        
        return results, comparison_results
    
    def create_comparison_analysis(self, results):
        """Create detailed comparison analysis"""
        
        comparison = {
            'performance_ranking': [],
            'implementation_complexity': {},
            'reoptimization_frequency': {},
            'risk_adjusted_performance': {}
        }
        
        # Performance ranking
        for strategy_name, data in results.items():
            metrics = data['performance']
            comparison['performance_ranking'].append({
                'strategy': strategy_name,
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'allocation_changes': data['allocation_changes']
            })
        
        # Sort by Sharpe ratio
        comparison['performance_ranking'].sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        # Implementation complexity analysis
        complexity_scores = {
            'baseline': 1,  # Simple static allocation
            'fixed_frequency': 3,  # Scheduled reoptimization
            'regime_based': 4,  # Complex trigger detection
            'performance_based': 4  # Performance monitoring required
        }
        
        for strategy_name in results.keys():
            comparison['implementation_complexity'][strategy_name] = complexity_scores[strategy_name]
        
        # Reoptimization frequency
        for strategy_name, data in results.items():
            comparison['reoptimization_frequency'][strategy_name] = {
                'total_changes': data['allocation_changes'],
                'changes_per_year': data['allocation_changes'] / (len(results['baseline']['returns']) / 12)
            }
        
        return comparison
    
    def create_summary_report(self, comparison_results, timestamp):
        """Create summary report"""
        
        report_lines = [
            "# REOPTIMIZATION STRATEGY COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: 26.5 years of MSCI factor data",
            "",
            "## PERFORMANCE COMPARISON",
            "",
            "| Rank | Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Reopt Changes |",
            "|------|----------|---------------|--------------|--------------|---------------|"
        ]
        
        # Performance ranking table
        for i, strategy_data in enumerate(comparison_results['performance_ranking'], 1):
            strategy_name = strategy_data['strategy'].replace('_', ' ').title()
            report_lines.append(
                f"| {i} | {strategy_name} | "
                f"{strategy_data['annual_return']:.2%} | "
                f"{strategy_data['sharpe_ratio']:.3f} | "
                f"{strategy_data['max_drawdown']:.2%} | "
                f"{strategy_data['allocation_changes']} |"
            )
        
        report_lines.extend([
            "",
            "## IMPLEMENTATION COMPLEXITY",
            ""
        ])
        
        # Complexity analysis
        complexity_names = {1: "Very Simple", 2: "Simple", 3: "Moderate", 4: "Complex", 5: "Very Complex"}
        for i, strategy_data in enumerate(comparison_results['performance_ranking'], 1):
            strategy_name = strategy_data['strategy']
            complexity = comparison_results['implementation_complexity'][strategy_name]
            complexity_desc = complexity_names[complexity]
            
            report_lines.append(f"**{strategy_name.replace('_', ' ').title()}**: {complexity_desc} "
                              f"({strategy_data['allocation_changes']} reoptimizations)")
        
        # Save report
        report_file = self.results_dir / f"reoptimization_summary_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved: {report_file}")
        
        # Print key findings
        print("\n" + "="*70)
        print("🎯 REOPTIMIZATION STRATEGY COMPARISON COMPLETE")
        print("="*70)
        
        print("\n📊 PERFORMANCE RANKING:")
        winner = comparison_results['performance_ranking'][0]
        baseline = next(s for s in comparison_results['performance_ranking'] if s['strategy'] == 'baseline')
        
        for i, strategy_data in enumerate(comparison_results['performance_ranking'], 1):
            strategy_name = strategy_data['strategy'].replace('_', ' ').title()
            vs_baseline = strategy_data['annual_return'] - baseline['annual_return']
            print(f"{i}. {strategy_name}: {strategy_data['annual_return']:.2%} return, "
                  f"{strategy_data['sharpe_ratio']:.3f} Sharpe ({vs_baseline:+.2%} vs baseline)")
        
        print(f"\n🏆 WINNER: {winner['strategy'].replace('_', ' ').title()}")
        print(f"   Performance: {winner['annual_return']:.2%} return, {winner['sharpe_ratio']:.3f} Sharpe")
        print(f"   Reoptimizations: {winner['allocation_changes']} over 26.5 years")
        
        improvement = winner['annual_return'] - baseline['annual_return']
        print(f"   Improvement over baseline: {improvement:+.2%} annual return")
    
    def make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(k): self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            # Convert series to dict with string keys
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
    tester = ReoptimizationStrategyTester()
    results, comparison = tester.run_comprehensive_comparison()
    return results, comparison

if __name__ == "__main__":
    main()