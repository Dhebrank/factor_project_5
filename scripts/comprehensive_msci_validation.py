"""
Comprehensive MSCI Factor Validation Framework
Exact replication of factor_project_4 methodology using 26.5 years of MSCI data
Includes: Static/Dynamic strategies, Walk-forward analysis, Bootstrap validation, 
Parameter optimization, Crisis testing, and S&P 500 benchmarking
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

class ComprehensiveMSCIValidator:
    """Comprehensive validation framework replicating factor_project_4 methodology"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/comprehensive_validation"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_all_data()
        
        # Strategy definitions (exact factor_project_4 replication)
        self.strategies = {
            'static_original': {'Value': 0.25, 'Quality': 0.30, 'MinVol': 0.20, 'Momentum': 0.25},
            'static_optimized': {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275},
            'basic_dynamic': 'regime_based',  # VIX-based regime detection
            'enhanced_dynamic': 'multi_signal'  # Multi-signal with factor momentum
        }
        
        # VIX regime thresholds (exact factor_project_4 values)
        self.vix_thresholds = {
            'normal': 25,
            'elevated': 35, 
            'stress': 50,
            'crisis': 75
        }
        
        # Crisis periods for systematic analysis
        self.crisis_periods = self.identify_crisis_periods()
        
    def load_all_data(self):
        """Load MSCI, market, and benchmark data"""
        logger.info("Loading comprehensive dataset...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data (VIX, S&P 500)
        self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                     index_col=0, parse_dates=True)
        
        # Combine datasets
        self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
        
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        logger.info(f"Factors: {list(self.factor_returns.columns)}")
        
    def identify_crisis_periods(self):
        """Identify major crisis periods in 26.5-year dataset"""
        logger.info("Identifying major crisis periods...")
        
        # Define major crisis periods based on historical events
        crisis_periods = {
            'dot_com_crash': ('1999-03-01', '2002-09-01'),
            'september_11': ('2001-08-01', '2001-12-01'),
            'financial_crisis': ('2007-06-01', '2009-03-01'),
            'european_debt': ('2010-04-01', '2012-06-01'),
            'china_devaluation': ('2015-06-01', '2016-02-01'),
            'volmageddon': ('2018-01-01', '2018-04-01'),
            'covid_pandemic': ('2020-01-01', '2020-06-01'),
            'inflation_shock': ('2022-01-01', '2022-10-01')
        }
        
        # Convert to datetime ranges
        crisis_ranges = {}
        for name, (start, end) in crisis_periods.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            
            # Check if crisis period overlaps with our data
            if start_date <= self.data.index.max() and end_date >= self.data.index.min():
                crisis_ranges[name] = (
                    max(start_date, self.data.index.min()),
                    min(end_date, self.data.index.max())
                )
        
        logger.info(f"Identified {len(crisis_ranges)} crisis periods within data range")
        for name, (start, end) in crisis_ranges.items():
            logger.info(f"  {name}: {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}")
            
        return crisis_ranges
    
    def create_vix_regimes(self):
        """Create VIX-based market regimes (exact factor_project_4 methodology)"""
        vix = self.data['VIX']
        
        regimes = pd.Series(0, index=vix.index)  # 0 = Normal
        regimes[vix > self.vix_thresholds['normal']] = 1      # Elevated
        regimes[vix > self.vix_thresholds['elevated']] = 2    # Stress
        regimes[vix > self.vix_thresholds['stress']] = 3      # Crisis
        
        self.regimes = regimes
        
        # Log regime distribution
        regime_counts = regimes.value_counts().sort_index()
        regime_names = ['Normal', 'Elevated', 'Stress', 'Crisis']
        logger.info("VIX regime distribution:")
        for i, count in regime_counts.items():
            logger.info(f"  {regime_names[i]}: {count} months ({count/len(regimes):.1%})")
        
        return regimes
    
    def calculate_strategy_returns(self, strategy_name, allocation_weights=None):
        """Calculate portfolio returns for a given strategy"""
        
        if strategy_name.startswith('static_'):
            # Static allocation strategy
            weights = allocation_weights or self.strategies[strategy_name]
            portfolio_returns = (self.factor_returns * pd.Series(weights)).sum(axis=1)
            
        elif strategy_name == 'basic_dynamic':
            # Basic dynamic strategy with VIX regime detection
            portfolio_returns = self.calculate_basic_dynamic_returns(allocation_weights)
            
        elif strategy_name == 'enhanced_dynamic':
            # Enhanced dynamic strategy with multi-signal regime detection
            portfolio_returns = self.calculate_enhanced_dynamic_returns(allocation_weights)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        return portfolio_returns
    
    def calculate_basic_dynamic_returns(self, base_allocation=None):
        """Calculate basic dynamic strategy returns (exact factor_project_4 methodology)"""
        if base_allocation is None:
            base_allocation = self.strategies['static_optimized']
        
        # Create VIX regimes
        if not hasattr(self, 'regimes'):
            self.create_vix_regimes()
        
        # Define regime-based allocations (exact factor_project_4 values)
        regime_allocations = {
            0: base_allocation,  # Normal
            1: base_allocation,  # Elevated 
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis
        }
        
        portfolio_returns = []
        for date in self.factor_returns.index:
            regime = self.regimes.loc[date]
            allocation = regime_allocations[regime]
            month_return = (self.factor_returns.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        return pd.Series(portfolio_returns, index=self.factor_returns.index)
    
    def calculate_enhanced_dynamic_returns(self, base_allocation=None):
        """Calculate enhanced dynamic strategy returns (exact factor_project_4 methodology)"""
        if base_allocation is None:
            base_allocation = self.strategies['static_optimized']
        
        # Create VIX regimes
        if not hasattr(self, 'regimes'):
            self.create_vix_regimes()
        
        # Calculate factor momentum (12-month rolling return)
        factor_momentum = self.factor_returns.rolling(12).sum()
        
        # Calculate factor momentum z-scores for tilting
        momentum_zscore = factor_momentum.rolling(36).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x) >= 12 else 0
        )
        
        portfolio_returns = []
        for i, date in enumerate(self.factor_returns.index):
            if i < 12:  # Need momentum history
                allocation = base_allocation
            else:
                regime = self.regimes.loc[date]
                
                # Base allocation based on regime
                if regime <= 1:  # Normal/Elevated
                    allocation = base_allocation.copy()
                    
                    # Apply momentum tilts (exact factor_project_4 methodology)
                    if i >= 36:  # Need z-score history
                        momentum_scores = momentum_zscore.loc[date]
                        
                        # Tilt allocation based on momentum z-scores
                        tilt_strength = 0.05  # 5% maximum tilt
                        for factor in allocation.keys():
                            momentum_tilt = np.clip(momentum_scores[factor] * 0.02, -tilt_strength, tilt_strength)
                            allocation[factor] += momentum_tilt
                        
                        # Normalize to sum to 1
                        total_weight = sum(allocation.values())
                        allocation = {k: v/total_weight for k, v in allocation.items()}
                        
                elif regime == 2:  # Stress
                    allocation = {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10}
                else:  # Crisis
                    allocation = {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}
            
            month_return = (self.factor_returns.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        return pd.Series(portfolio_returns, index=self.factor_returns.index)
    
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """Calculate comprehensive performance metrics (exact factor_project_4 methodology)"""
        
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
    
    def run_walk_forward_analysis(self, strategy_name, allocation_weights=None):
        """Run walk-forward analysis (exact factor_project_4 methodology)"""
        logger.info(f"Running walk-forward analysis for {strategy_name}...")
        
        # Define validation periods (18 periods over 26.5 years)
        total_months = len(self.factor_returns)
        min_training = 60  # 5 years minimum training
        test_length = 12   # 1 year test periods
        
        validation_results = []
        
        for i in range(min_training, total_months - test_length + 1, test_length):
            # Training period
            train_start = 0
            train_end = i
            train_data = self.factor_returns.iloc[train_start:train_end]
            
            # Test period  
            test_start = i
            test_end = min(i + test_length, total_months)
            test_data = self.factor_returns.iloc[test_start:test_end]
            
            # Calculate returns for test period
            test_returns = self.calculate_strategy_returns(strategy_name, allocation_weights)
            test_returns = test_returns.iloc[test_start:test_end]
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(test_returns)
            metrics['train_start'] = train_data.index[0]
            metrics['train_end'] = train_data.index[-1]
            metrics['test_start'] = test_data.index[0] 
            metrics['test_end'] = test_data.index[-1]
            metrics['test_periods'] = len(test_data)
            
            validation_results.append(metrics)
        
        logger.info(f"Completed {len(validation_results)} validation periods")
        return validation_results
    
    def run_bootstrap_validation(self, strategy_name, allocation_weights=None, n_samples=1000):
        """Run bootstrap validation (exact factor_project_4 methodology)"""
        logger.info(f"Running bootstrap validation for {strategy_name} ({n_samples} samples)...")
        
        # Calculate strategy returns
        strategy_returns = self.calculate_strategy_returns(strategy_name, allocation_weights)
        
        bootstrap_results = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_samples):
            # Bootstrap sample (sample with replacement)
            sample_indices = np.random.choice(len(strategy_returns), len(strategy_returns), replace=True)
            sample_returns = strategy_returns.iloc[sample_indices]
            
            # Calculate metrics for bootstrap sample
            metrics = self.calculate_performance_metrics(sample_returns)
            bootstrap_results.append(metrics)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Completed {i + 1}/{n_samples} bootstrap samples")
        
        # Calculate confidence intervals
        bootstrap_df = pd.DataFrame(bootstrap_results)
        confidence_intervals = {}
        
        for metric in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']:
            lower = bootstrap_df[metric].quantile(0.025)
            upper = bootstrap_df[metric].quantile(0.975)
            confidence_intervals[metric] = (lower, upper)
        
        logger.info("Bootstrap confidence intervals (95%):")
        for metric, (lower, upper) in confidence_intervals.items():
            logger.info(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
        
        return bootstrap_results, confidence_intervals
    
    def run_parameter_optimization(self, n_combinations=1680):
        """Run parameter grid search optimization (exact factor_project_4 methodology)"""
        logger.info(f"Running parameter optimization ({n_combinations} combinations)...")
        
        # Define parameter ranges
        value_range = np.arange(0.10, 0.35, 0.05)     # 10% to 30%
        quality_range = np.arange(0.20, 0.40, 0.05)   # 20% to 35%
        minvol_range = np.arange(0.15, 0.40, 0.05)    # 15% to 35%
        momentum_range = np.arange(0.15, 0.40, 0.05)  # 15% to 35%
        
        # Generate all combinations
        all_combinations = list(product(value_range, quality_range, minvol_range, momentum_range))
        
        # Filter valid combinations (sum = 1.0 ± 0.01)
        valid_combinations = []
        for combo in all_combinations:
            if abs(sum(combo) - 1.0) <= 0.01:
                valid_combinations.append(combo)
        
        # Limit to n_combinations if too many
        if len(valid_combinations) > n_combinations:
            np.random.seed(42)
            selected_indices = np.random.choice(len(valid_combinations), n_combinations, replace=False)
            valid_combinations = [valid_combinations[i] for i in selected_indices]
        
        logger.info(f"Testing {len(valid_combinations)} valid allocation combinations...")
        
        optimization_results = []
        
        for i, (value_w, quality_w, minvol_w, momentum_w) in enumerate(valid_combinations):
            # Normalize weights to exactly 1.0
            total = value_w + quality_w + minvol_w + momentum_w
            weights = {
                'Value': value_w / total,
                'Quality': quality_w / total, 
                'MinVol': minvol_w / total,
                'Momentum': momentum_w / total
            }
            
            # Calculate strategy returns
            returns = self.calculate_strategy_returns('static_original', weights)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(returns)
            metrics['allocation'] = weights
            
            # Multi-objective score (exact factor_project_4 methodology)
            metrics['multi_objective_score'] = (
                metrics['sharpe_ratio'] * 0.40 +
                metrics['sortino_ratio'] * 0.25 +
                metrics['calmar_ratio'] * 0.20 +
                abs(metrics['max_drawdown']) * (-0.15)  # Penalize large drawdowns
            )
            
            optimization_results.append(metrics)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Completed {i + 1}/{len(valid_combinations)} combinations")
        
        # Find best allocation
        best_result = max(optimization_results, key=lambda x: x['multi_objective_score'])
        
        logger.info("Optimization complete!")
        logger.info(f"Best allocation: {best_result['allocation']}")
        logger.info(f"Best performance: {best_result['annual_return']:.2%} return, {best_result['sharpe_ratio']:.3f} Sharpe")
        
        return optimization_results, best_result
    
    def analyze_crisis_performance(self):
        """Analyze factor performance during crisis periods (exact factor_project_4 methodology)"""
        logger.info("Analyzing crisis period performance...")
        
        crisis_analysis = {}
        
        for crisis_name, (start_date, end_date) in self.crisis_periods.items():
            # Get crisis period data
            crisis_mask = (self.factor_returns.index >= start_date) & (self.factor_returns.index <= end_date)
            crisis_returns = self.factor_returns[crisis_mask]
            
            if len(crisis_returns) == 0:
                continue
            
            # Calculate factor performance during crisis
            factor_performance = {}
            for factor in self.factor_returns.columns:
                crisis_factor_returns = crisis_returns[factor]
                
                # Crisis performance metrics
                cumulative_return = (1 + crisis_factor_returns).prod() - 1
                max_drawdown = ((1 + crisis_factor_returns).cumprod() / 
                              (1 + crisis_factor_returns).cumprod().expanding().max() - 1).min()
                volatility = crisis_factor_returns.std() * np.sqrt(12)
                
                factor_performance[factor] = {
                    'cumulative_return': cumulative_return,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'periods': len(crisis_factor_returns)
                }
            
            # Portfolio performance during crisis
            portfolio_strategies = {}
            for strategy_name in ['static_original', 'static_optimized', 'basic_dynamic', 'enhanced_dynamic']:
                try:
                    strategy_returns = self.calculate_strategy_returns(strategy_name)
                    crisis_portfolio_returns = strategy_returns[crisis_mask]
                    
                    if len(crisis_portfolio_returns) > 0:
                        portfolio_metrics = self.calculate_performance_metrics(crisis_portfolio_returns)
                        portfolio_strategies[strategy_name] = portfolio_metrics
                except Exception as e:
                    logger.warning(f"Could not calculate {strategy_name} for {crisis_name}: {e}")
            
            crisis_analysis[crisis_name] = {
                'period': (start_date, end_date),
                'duration_months': len(crisis_returns),
                'factor_performance': factor_performance,
                'portfolio_performance': portfolio_strategies
            }
            
            logger.info(f"  {crisis_name}: {len(crisis_returns)} months")
        
        return crisis_analysis
    
    def run_comprehensive_validation(self):
        """Run complete comprehensive validation framework"""
        logger.info("🚀 Starting Comprehensive MSCI Validation Framework...")
        logger.info("Replicating factor_project_4 methodology with 26.5-year MSCI data")
        
        comprehensive_results = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'data_period': (self.data.index.min().isoformat(), self.data.index.max().isoformat()),
                'total_observations': len(self.data),
                'methodology': 'factor_project_4_replication'
            }
        }
        
        # 1. Calculate benchmark (S&P 500) performance
        if 'SP500_Monthly_Return' in self.data.columns:
            sp500_returns = self.data['SP500_Monthly_Return'].dropna()
            benchmark_metrics = self.calculate_performance_metrics(sp500_returns)
            comprehensive_results['benchmark_performance'] = benchmark_metrics
            logger.info(f"S&P 500 Benchmark: {benchmark_metrics['annual_return']:.2%} return, {benchmark_metrics['sharpe_ratio']:.3f} Sharpe")
        
        # 2. Strategy performance comparison
        strategy_performance = {}
        for strategy_name in ['static_original', 'static_optimized', 'basic_dynamic', 'enhanced_dynamic']:
            logger.info(f"\n--- Testing {strategy_name.upper()} Strategy ---")
            
            # Calculate returns and metrics
            strategy_returns = self.calculate_strategy_returns(strategy_name)
            benchmark_returns = sp500_returns if 'SP500_Monthly_Return' in self.data.columns else None
            metrics = self.calculate_performance_metrics(strategy_returns, benchmark_returns)
            
            logger.info(f"Performance: {metrics['annual_return']:.2%} return, {metrics['sharpe_ratio']:.3f} Sharpe, {metrics['max_drawdown']:.2%} max DD")
            
            strategy_performance[strategy_name] = {
                'metrics': metrics,
                'returns': strategy_returns.to_dict()
            }
        
        comprehensive_results['strategy_performance'] = strategy_performance
        
        # 3. Walk-forward analysis
        logger.info("\n--- Walk-Forward Analysis ---")
        walkforward_results = {}
        for strategy_name in ['static_optimized', 'enhanced_dynamic']:
            walkforward_results[strategy_name] = self.run_walk_forward_analysis(strategy_name)
        comprehensive_results['walkforward_analysis'] = walkforward_results
        
        # 4. Bootstrap validation
        logger.info("\n--- Bootstrap Validation ---")
        bootstrap_results = {}
        for strategy_name in ['static_optimized', 'enhanced_dynamic']:
            bootstrap_samples, confidence_intervals = self.run_bootstrap_validation(strategy_name, n_samples=1000)
            bootstrap_results[strategy_name] = {
                'confidence_intervals': confidence_intervals,
                'sample_count': len(bootstrap_samples)
            }
        comprehensive_results['bootstrap_validation'] = bootstrap_results
        
        # 5. Parameter optimization
        logger.info("\n--- Parameter Optimization ---")
        optimization_results, best_allocation = self.run_parameter_optimization(n_combinations=1680)
        comprehensive_results['parameter_optimization'] = {
            'best_allocation': best_allocation,
            'total_combinations_tested': len(optimization_results)
        }
        
        # 6. Crisis analysis
        logger.info("\n--- Crisis Period Analysis ---")
        crisis_analysis = self.analyze_crisis_performance()
        comprehensive_results['crisis_analysis'] = crisis_analysis
        
        # 7. Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_msci_validation_{timestamp}.json"
        
        # Convert non-serializable objects
        serializable_results = self.make_serializable(comprehensive_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Comprehensive validation completed!")
        logger.info(f"Results saved: {results_file}")
        
        # Summary report
        self.create_summary_report(comprehensive_results, timestamp)
        
        return comprehensive_results
    
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
    
    def create_summary_report(self, results, timestamp):
        """Create summary report of validation results"""
        
        summary_lines = [
            "# COMPREHENSIVE MSCI FACTOR VALIDATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Methodology: Exact replication of factor_project_4 with 26.5-year MSCI data",
            "",
            "## PERFORMANCE SUMMARY",
            ""
        ]
        
        # Strategy performance table
        summary_lines.append("| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | vs S&P 500 |")
        summary_lines.append("|----------|---------------|--------------|--------------|-------------|")
        
        benchmark_return = results.get('benchmark_performance', {}).get('annual_return', 0)
        
        for strategy_name, data in results['strategy_performance'].items():
            metrics = data['metrics']
            outperformance = metrics['annual_return'] - benchmark_return
            summary_lines.append(
                f"| {strategy_name.replace('_', ' ').title()} | "
                f"{metrics['annual_return']:.2%} | "
                f"{metrics['sharpe_ratio']:.3f} | "
                f"{metrics['max_drawdown']:.2%} | "
                f"{outperformance:+.2%} |"
            )
        
        # Best allocation from optimization
        if 'parameter_optimization' in results:
            best_alloc = results['parameter_optimization']['best_allocation']['allocation']
            summary_lines.extend([
                "",
                "## OPTIMAL ALLOCATION (Parameter Optimization)",
                f"Value: {best_alloc['Value']:.1%}, Quality: {best_alloc['Quality']:.1%}, "
                f"MinVol: {best_alloc['MinVol']:.1%}, Momentum: {best_alloc['Momentum']:.1%}",
                f"Performance: {results['parameter_optimization']['best_allocation']['annual_return']:.2%} return, "
                f"{results['parameter_optimization']['best_allocation']['sharpe_ratio']:.3f} Sharpe ratio"
            ])
        
        # Crisis analysis summary
        if 'crisis_analysis' in results:
            summary_lines.extend([
                "",
                "## CRISIS PERIOD ANALYSIS",
                f"Analyzed {len(results['crisis_analysis'])} major crisis periods:",
                ""
            ])
            
            for crisis_name, crisis_data in results['crisis_analysis'].items():
                summary_lines.append(f"- **{crisis_name.replace('_', ' ').title()}**: {crisis_data['duration_months']} months")
        
        # Save summary report
        summary_file = self.results_dir / f"comprehensive_validation_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary report saved: {summary_file}")
        
        # Print key findings
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE MSCI VALIDATION COMPLETE")
        print("="*60)
        print(f"Methodology: factor_project_4 replication with 26.5-year MSCI data")
        print(f"Period: {results['metadata']['data_period'][0][:10]} to {results['metadata']['data_period'][1][:10]}")
        print(f"Observations: {results['metadata']['total_observations']} months")
        
        print("\n📊 STRATEGY RANKINGS:")
        strategy_rankings = sorted(
            results['strategy_performance'].items(),
            key=lambda x: x[1]['metrics']['sharpe_ratio'],
            reverse=True
        )
        
        for i, (strategy, data) in enumerate(strategy_rankings, 1):
            metrics = data['metrics']
            print(f"{i}. {strategy.replace('_', ' ').title()}: "
                  f"{metrics['annual_return']:.2%} return, "
                  f"{metrics['sharpe_ratio']:.3f} Sharpe")

def main():
    """Main execution"""
    validator = ComprehensiveMSCIValidator()
    results = validator.run_comprehensive_validation()

if __name__ == "__main__":
    main()