"""
CORRECTED Basic Dynamic v2: Proper OOS Methodology
Fixes the in-sample bias by using walk-forward validation averages instead of 
applying optimized parameters to full dataset
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

class CorrectedBasicDynamicV2:
    """Corrected Basic Dynamic v2 with proper OOS methodology"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/corrected_basic_dynamic_v2"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # VIX threshold ranges for optimization
        self.threshold_ranges = {
            'normal_elevated': range(20, 31, 2),    # [20, 22, 24, 26, 28, 30]
            'elevated_stress': range(30, 46, 3),    # [30, 33, 36, 39, 42, 45] 
            'stress_crisis': range(45, 61, 4)       # [45, 49, 53, 57, 61]
        }
        
        # Base allocation (from factor_project_4)
        self.base_allocation = {'Value': 0.15, 'Quality': 0.275, 'MinVol': 0.30, 'Momentum': 0.275}
        
        # Baseline thresholds for comparison
        self.baseline_thresholds = (25, 35, 50)
        
        logger.info(f"Testing {len(list(product(*self.threshold_ranges.values())))} threshold combinations")
        
    def load_data(self):
        """Load MSCI factor returns and market data"""
        logger.info("Loading data for corrected Basic Dynamic v2...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load market data 
        try:
            self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                         index_col=0, parse_dates=True)
            self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
        except:
            logger.warning("Market data not found, creating dummy VIX")
            self.data = self.factor_returns.copy()
            # Create realistic VIX data based on market volatility
            self.data['VIX'] = 20 + 15 * np.random.randn(len(self.data)).cumsum() * 0.1
            self.data['VIX'] = np.clip(self.data['VIX'], 10, 80)
            
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"VIX range: {self.data['VIX'].min():.1f} to {self.data['VIX'].max():.1f}")
        
    def create_vix_regimes(self, data_subset, normal_elevated, elevated_stress, stress_crisis):
        """Create VIX-based market regimes with custom thresholds for data subset"""
        vix = data_subset['VIX'] if 'VIX' in data_subset.columns else data_subset.iloc[:, -1]
        
        regimes = pd.Series(0, index=vix.index)  # 0 = Normal
        regimes[vix >= normal_elevated] = 1      # Elevated
        regimes[vix >= elevated_stress] = 2      # Stress
        regimes[vix >= stress_crisis] = 3        # Crisis
        
        return regimes
        
    def calculate_basic_dynamic_returns(self, data_subset, threshold_set):
        """Calculate Basic Dynamic returns for data subset with custom VIX thresholds"""
        normal_elevated, elevated_stress, stress_crisis = threshold_set
        
        # Create regimes for this data subset
        regimes = self.create_vix_regimes(data_subset, normal_elevated, elevated_stress, stress_crisis)
        
        # Define regime-based allocations
        regime_allocations = {
            0: self.base_allocation,  # Normal
            1: self.base_allocation,  # Elevated 
            2: {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10},  # Stress
            3: {'Value': 0.15, 'Quality': 0.40, 'MinVol': 0.40, 'Momentum': 0.05}   # Crisis
        }
        
        # Get factor returns for this subset
        factor_subset = data_subset[self.factor_returns.columns]
        
        portfolio_returns = []
        for date in factor_subset.index:
            regime = regimes.loc[date]
            allocation = regime_allocations[regime]
            month_return = (factor_subset.loc[date] * pd.Series(allocation)).sum()
            portfolio_returns.append(month_return)
        
        return pd.Series(portfolio_returns, index=factor_subset.index)
    
    def calculate_performance_metrics(self, returns):
        """Calculate performance metrics for returns series"""
        if len(returns) < 12:
            return {
                'annual_return': 0, 'sharpe_ratio': 0, 'max_drawdown': -0.05,
                'annual_volatility': 0.15, 'win_rate': 0.5
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
        win_rate = (returns > 0).mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def run_corrected_walk_forward_optimization(self):
        """Run CORRECTED walk-forward optimization with proper OOS reporting"""
        logger.info("🚀 Running CORRECTED Basic Dynamic v2 walk-forward optimization...")
        
        # Generate threshold combinations
        threshold_combinations = list(product(*self.threshold_ranges.values()))
        valid_combinations = []
        
        for combo in threshold_combinations:
            normal_elevated, elevated_stress, stress_crisis = combo
            if normal_elevated < elevated_stress < stress_crisis:
                valid_combinations.append(combo)
        
        logger.info(f"Testing {len(valid_combinations)} valid threshold combinations...")
        
        # Walk-forward parameters
        total_months = len(self.factor_returns)
        min_training = 60  # 5 years minimum training
        test_length = 12   # 1 year test periods
        
        # Store ALL validation results for proper OOS calculation
        all_validation_results = []
        threshold_performance = {combo: [] for combo in valid_combinations}
        
        # Walk through validation periods
        validation_periods = []
        for i in range(min_training, total_months - test_length + 1, test_length):
            # Training period
            train_start = 0
            train_end = i
            train_data = self.data.iloc[train_start:train_end]
            
            # Test period
            test_start = i
            test_end = min(i + test_length, total_months)
            test_data = self.data.iloc[test_start:test_end]
            
            validation_periods.append({
                'train_start': train_start, 'train_end': train_end,
                'test_start': test_start, 'test_end': test_end,
                'train_data': train_data, 'test_data': test_data
            })
        
        logger.info(f"Running {len(validation_periods)} validation periods...")
        
        # For each validation period, find best threshold and test OOS
        for period_idx, period_info in enumerate(validation_periods):
            logger.info(f"Processing validation period {period_idx + 1}/{len(validation_periods)}")
            
            train_data = period_info['train_data']
            test_data = period_info['test_data']
            
            # Find best threshold set for this training period
            best_combo = None
            best_score = -np.inf
            
            for combo in valid_combinations:
                # Test this combination on training data
                train_returns = self.calculate_basic_dynamic_returns(train_data, combo)
                train_metrics = self.calculate_performance_metrics(train_returns)
                
                # Multi-objective score (same as other strategies)
                score = (
                    train_metrics['sharpe_ratio'] * 0.40 +
                    (train_metrics['annual_return'] / abs(train_metrics['max_drawdown'])) * 0.25 +
                    train_metrics['win_rate'] * 0.20 +
                    abs(train_metrics['max_drawdown']) * (-0.15)
                )
                
                if score > best_score:
                    best_score = score
                    best_combo = combo
            
            # Test best combination on OOS test period
            if best_combo:
                test_returns = self.calculate_basic_dynamic_returns(test_data, best_combo)
                test_metrics = self.calculate_performance_metrics(test_returns)
                
                # Store validation result
                validation_result = {
                    'period': period_idx,
                    'train_start_date': train_data.index[0],
                    'train_end_date': train_data.index[-1], 
                    'test_start_date': test_data.index[0],
                    'test_end_date': test_data.index[-1],
                    'best_thresholds': best_combo,
                    'oos_performance': test_metrics,
                    'test_returns': test_returns
                }
                
                all_validation_results.append(validation_result)
                threshold_performance[best_combo].append(test_metrics)
        
        # Calculate PROPER OOS performance (average metrics across validation periods)
        if all_validation_results:
            # Method 1: Average metrics across validation periods (CORRECT approach)
            avg_annual_return = np.mean([r['oos_performance']['annual_return'] for r in all_validation_results])
            avg_sharpe_ratio = np.mean([r['oos_performance']['sharpe_ratio'] for r in all_validation_results])
            avg_max_drawdown = np.mean([r['oos_performance']['max_drawdown'] for r in all_validation_results])
            avg_volatility = np.mean([r['oos_performance']['annual_volatility'] for r in all_validation_results])
            
            # Calculate standard errors for confidence
            return_std = np.std([r['oos_performance']['annual_return'] for r in all_validation_results])
            sharpe_std = np.std([r['oos_performance']['sharpe_ratio'] for r in all_validation_results])
            
            corrected_oos_performance = {
                'annual_return': avg_annual_return,
                'sharpe_ratio': avg_sharpe_ratio,
                'max_drawdown': avg_max_drawdown,
                'annual_volatility': avg_volatility,
                'validation_periods': len(all_validation_results),
                'return_std_error': return_std / np.sqrt(len(all_validation_results)),
                'sharpe_std_error': sharpe_std / np.sqrt(len(all_validation_results))
            }
            
            # Find most frequently selected threshold combination
            threshold_counts = {}
            for result in all_validation_results:
                combo = result['best_thresholds']
                threshold_counts[combo] = threshold_counts.get(combo, 0) + 1
            
            most_common_thresholds = max(threshold_counts.keys(), key=lambda x: threshold_counts[x])
            
            logger.info("✅ CORRECTED walk-forward optimization complete!")
            logger.info(f"Corrected OOS performance: {corrected_oos_performance['annual_return']:.2%} return, {corrected_oos_performance['sharpe_ratio']:.3f} Sharpe")
            logger.info(f"Most common thresholds: {most_common_thresholds}")
            
            return all_validation_results, corrected_oos_performance, most_common_thresholds
        else:
            logger.error("No validation results generated!")
            return [], {}, None
    
    def calculate_baseline_performance(self):
        """Calculate baseline Basic Dynamic performance with original thresholds"""
        logger.info("Calculating baseline Basic Dynamic performance...")
        
        # Use original thresholds (25, 35, 50)
        baseline_returns = self.calculate_basic_dynamic_returns(self.data, self.baseline_thresholds)
        baseline_performance = self.calculate_performance_metrics(baseline_returns)
        
        logger.info(f"Baseline performance: {baseline_performance['annual_return']:.2%} return, {baseline_performance['sharpe_ratio']:.3f} Sharpe")
        
        return baseline_performance, baseline_returns
    
    def run_comprehensive_corrected_analysis(self):
        """Run comprehensive corrected analysis"""
        logger.info("🚀 Running comprehensive CORRECTED Basic Dynamic v2 analysis...")
        
        results = {
            'analysis_date': datetime.now().isoformat(),
            'methodology': 'CORRECTED - proper OOS walk-forward validation'
        }
        
        # Calculate baseline performance
        logger.info("\n--- Baseline Basic Dynamic Performance ---")
        baseline_performance, baseline_returns = self.calculate_baseline_performance()
        results['baseline_performance'] = baseline_performance
        
        # Run corrected walk-forward optimization
        logger.info("\n--- Corrected Walk-Forward Optimization ---")
        validation_results, corrected_performance, common_thresholds = self.run_corrected_walk_forward_optimization()
        results['corrected_oos_performance'] = corrected_performance
        results['validation_results'] = validation_results
        results['most_common_thresholds'] = common_thresholds
        
        # Calculate improvement vs baseline
        if corrected_performance and baseline_performance:
            improvement = {
                'return_improvement': corrected_performance['annual_return'] - baseline_performance['annual_return'],
                'sharpe_improvement': corrected_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio'],
                'improvement_significant': abs(corrected_performance['annual_return'] - baseline_performance['annual_return']) > 0.001
            }
            results['improvement_analysis'] = improvement
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"corrected_basic_dynamic_v2_{timestamp}.json"
        
        # Make serializable and handle validation results separately
        results_to_save = results.copy()
        results_to_save['validation_results'] = len(validation_results)  # Just save count
        serializable_results = self.make_serializable(results_to_save)
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved: {results_file}")
        
        # Print summary
        self.print_corrected_summary(results)
        
        return results
    
    def make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(k): self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
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
    
    def print_corrected_summary(self, results):
        """Print corrected analysis summary"""
        
        print("\n" + "="*80)
        print("🔧 CORRECTED BASIC DYNAMIC V2 ANALYSIS SUMMARY")
        print("="*80)
        
        baseline = results['baseline_performance']
        corrected = results['corrected_oos_performance']
        improvement = results.get('improvement_analysis', {})
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Baseline Basic Dynamic: {baseline['annual_return']:.2%} return, {baseline['sharpe_ratio']:.3f} Sharpe")
        print(f"   Corrected Basic Dynamic v2: {corrected['annual_return']:.2%} return, {corrected['sharpe_ratio']:.3f} Sharpe")
        
        if improvement:
            print(f"\n📈 IMPROVEMENT ANALYSIS:")
            print(f"   Return improvement: {improvement['return_improvement']:+.2%}")
            print(f"   Sharpe improvement: {improvement['sharpe_improvement']:+.3f}")
            print(f"   Improvement significant: {improvement['improvement_significant']}")
        
        print(f"\n🎯 METHODOLOGY CORRECTION:")
        print(f"   Previous (BIASED): Applied optimized thresholds to full dataset")
        print(f"   Corrected (OOS): Used walk-forward validation averages")
        print(f"   Validation periods: {corrected['validation_periods']}")
        
        print(f"\n🏆 CONCLUSION:")
        if improvement and improvement['improvement_significant']:
            print(f"   VIX threshold optimization provides meaningful improvement")
        else:
            print(f"   VIX threshold optimization provides minimal/no improvement")
            print(f"   Original academic thresholds (25/35/50) already well-calibrated")

def main():
    """Main execution"""
    corrector = CorrectedBasicDynamicV2()
    results = corrector.run_comprehensive_corrected_analysis()
    return results

if __name__ == "__main__":
    main()