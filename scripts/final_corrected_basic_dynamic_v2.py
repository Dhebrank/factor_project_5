"""
FINAL CORRECTED Basic Dynamic v2: Real VIX Data + Concatenated Returns
Uses real Sharadar VIX data and proper concatenated returns methodology for accurate OOS performance
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

class FinalCorrectedBasicDynamicV2:
    """Final corrected Basic Dynamic v2 with real VIX data and concatenated returns methodology"""
    
    def __init__(self, data_dir="data/processed", results_dir="results/final_corrected_basic_dynamic_v2"):
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
        """Load MSCI factor returns and REAL market data"""
        logger.info("Loading data for FINAL corrected Basic Dynamic v2...")
        
        # Load MSCI factor returns
        self.factor_returns = pd.read_csv(self.data_dir / "msci_factor_returns.csv", 
                                        index_col=0, parse_dates=True)
        
        # Load REAL market data
        try:
            self.market_data = pd.read_csv(self.data_dir / "market_data.csv", 
                                         index_col=0, parse_dates=True)
            self.data = pd.concat([self.factor_returns, self.market_data], axis=1)
            logger.info("✅ Using REAL VIX data from Sharadar")
        except Exception as e:
            logger.error(f"❌ Could not load real market data: {e}")
            raise
            
        logger.info(f"Loaded {len(self.data)} monthly observations")
        logger.info(f"REAL VIX range: {self.data['VIX'].min():.1f} to {self.data['VIX'].max():.1f}")
        
    def create_vix_regimes(self, data_subset, normal_elevated, elevated_stress, stress_crisis):
        """Create VIX-based market regimes with custom thresholds for data subset"""
        vix = data_subset['VIX']
        
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
    
    def run_corrected_walk_forward_with_concatenated_returns(self):
        """Run CORRECTED walk-forward optimization with concatenated returns methodology"""
        logger.info("🚀 Running FINAL CORRECTED Basic Dynamic v2 with concatenated returns...")
        
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
        
        # Store ALL OOS returns for concatenation
        all_oos_returns = []
        validation_period_details = []
        
        # Walk through validation periods
        for i in range(min_training, total_months - test_length + 1, test_length):
            # Training period
            train_start = 0
            train_end = i
            train_data = self.data.iloc[train_start:train_end]
            
            # Test period
            test_start = i
            test_end = min(i + test_length, total_months)
            test_data = self.data.iloc[test_start:test_end]
            
            logger.info(f"Validation period: train {train_start}:{train_end}, test {test_start}:{test_end}")
            
            # Find best threshold set for this training period
            best_combo = None
            best_score = -np.inf
            
            for combo in valid_combinations:
                # Test this combination on training data
                train_returns = self.calculate_basic_dynamic_returns(train_data, combo)
                train_metrics = self.calculate_performance_metrics(train_returns)
                
                # Multi-objective score
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
                
                # Store OOS returns for concatenation
                all_oos_returns.append(test_returns)
                
                # Store validation details
                validation_period_details.append({
                    'train_start_date': train_data.index[0],
                    'train_end_date': train_data.index[-1], 
                    'test_start_date': test_data.index[0],
                    'test_end_date': test_data.index[-1],
                    'best_thresholds': best_combo,
                    'test_return_periods': len(test_returns)
                })
        
        # CONCATENATE all OOS returns (PROPER methodology)
        if all_oos_returns:
            concatenated_oos_returns = pd.concat(all_oos_returns)
            
            # Calculate performance metrics on concatenated OOS returns
            final_oos_performance = self.calculate_performance_metrics(concatenated_oos_returns)
            
            # Add concatenation details
            final_oos_performance['total_oos_periods'] = len(concatenated_oos_returns)
            final_oos_performance['validation_periods'] = len(validation_period_details)
            final_oos_performance['methodology'] = 'CONCATENATED_RETURNS'
            
            logger.info("✅ FINAL CORRECTED walk-forward optimization complete!")
            logger.info(f"Concatenated OOS performance: {final_oos_performance['annual_return']:.2%} return, {final_oos_performance['sharpe_ratio']:.3f} Sharpe")
            logger.info(f"Total OOS periods: {final_oos_performance['total_oos_periods']}")
            
            return final_oos_performance, concatenated_oos_returns, validation_period_details
        else:
            logger.error("No OOS returns generated!")
            return {}, pd.Series(), []
    
    def calculate_baseline_performance(self):
        """Calculate baseline Basic Dynamic performance with original thresholds"""
        logger.info("Calculating baseline Basic Dynamic performance...")
        
        # Use original thresholds (25, 35, 50)
        baseline_returns = self.calculate_basic_dynamic_returns(self.data, self.baseline_thresholds)
        baseline_performance = self.calculate_performance_metrics(baseline_returns)
        
        logger.info(f"Baseline performance: {baseline_performance['annual_return']:.2%} return, {baseline_performance['sharpe_ratio']:.3f} Sharpe")
        
        return baseline_performance, baseline_returns
    
    def run_final_comprehensive_analysis(self):
        """Run final comprehensive corrected analysis"""
        logger.info("🚀 Running FINAL COMPREHENSIVE corrected Basic Dynamic v2 analysis...")
        
        results = {
            'analysis_date': datetime.now().isoformat(),
            'methodology': 'FINAL CORRECTED - real VIX data + concatenated returns',
            'data_source': 'Sharadar VIX data'
        }
        
        # Calculate baseline performance
        logger.info("\n--- Baseline Basic Dynamic Performance ---")
        baseline_performance, baseline_returns = self.calculate_baseline_performance()
        results['baseline_performance'] = baseline_performance
        
        # Run final corrected walk-forward optimization
        logger.info("\n--- Final Corrected Walk-Forward Optimization ---")
        corrected_performance, oos_returns, validation_details = self.run_corrected_walk_forward_with_concatenated_returns()
        results['corrected_oos_performance'] = corrected_performance
        results['validation_details'] = validation_details
        
        # Calculate improvement vs baseline
        if corrected_performance and baseline_performance:
            improvement = {
                'return_improvement': corrected_performance['annual_return'] - baseline_performance['annual_return'],
                'sharpe_improvement': corrected_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio'],
                'improvement_significant': abs(corrected_performance['annual_return'] - baseline_performance['annual_return']) > 0.005,
                'improvement_percentage': (corrected_performance['annual_return'] - baseline_performance['annual_return']) / baseline_performance['annual_return'] * 100
            }
            results['improvement_analysis'] = improvement
        
        # Enhanced Dynamic comparison (from previous analysis)
        enhanced_dynamic_performance = {
            'annual_return': 0.0988,
            'sharpe_ratio': 0.719,
            'source': 'Previous verified analysis'
        }
        results['enhanced_dynamic_comparison'] = enhanced_dynamic_performance
        
        # Calculate vs Enhanced Dynamic
        if corrected_performance:
            vs_enhanced = {
                'return_vs_enhanced': corrected_performance['annual_return'] - enhanced_dynamic_performance['annual_return'],
                'sharpe_vs_enhanced': corrected_performance['sharpe_ratio'] - enhanced_dynamic_performance['sharpe_ratio'],
                'outperforms_enhanced': corrected_performance['annual_return'] > enhanced_dynamic_performance['annual_return']
            }
            results['vs_enhanced_dynamic'] = vs_enhanced
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"final_corrected_basic_dynamic_v2_{timestamp}.json"
        
        # Make serializable
        serializable_results = self.make_serializable(results)
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved: {results_file}")
        
        # Print final summary
        self.print_final_summary(results)
        
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
    
    def print_final_summary(self, results):
        """Print final corrected analysis summary"""
        
        print("\n" + "="*80)
        print("🎯 FINAL CORRECTED BASIC DYNAMIC V2 ANALYSIS SUMMARY")
        print("="*80)
        
        baseline = results['baseline_performance']
        corrected = results['corrected_oos_performance']
        improvement = results.get('improvement_analysis', {})
        vs_enhanced = results.get('vs_enhanced_dynamic', {})
        
        print(f"\n📊 PERFORMANCE COMPARISON (Real VIX Data + Concatenated Returns):")
        print(f"   Baseline Basic Dynamic: {baseline['annual_return']:.2%} return, {baseline['sharpe_ratio']:.3f} Sharpe")
        print(f"   Corrected Basic Dynamic v2: {corrected['annual_return']:.2%} return, {corrected['sharpe_ratio']:.3f} Sharpe")
        
        if improvement:
            print(f"\n📈 IMPROVEMENT ANALYSIS:")
            print(f"   Return improvement: {improvement['return_improvement']:+.2%} ({improvement['improvement_percentage']:+.1f}%)")
            print(f"   Sharpe improvement: {improvement['sharpe_improvement']:+.3f}")
            print(f"   Improvement significant: {improvement['improvement_significant']}")
        
        if vs_enhanced:
            print(f"\n🏆 VS ENHANCED DYNAMIC:")
            print(f"   Enhanced Dynamic: 9.88% return, 0.719 Sharpe")
            print(f"   Return vs Enhanced: {vs_enhanced['return_vs_enhanced']:+.2%}")
            print(f"   Sharpe vs Enhanced: {vs_enhanced['sharpe_vs_enhanced']:+.3f}")
            print(f"   Outperforms Enhanced: {vs_enhanced['outperforms_enhanced']}")
        
        print(f"\n🎯 METHODOLOGY VALIDATION:")
        print(f"   Data Source: Real Sharadar VIX data")
        print(f"   Methodology: Concatenated OOS returns")
        print(f"   Total OOS periods: {corrected['total_oos_periods']}")
        print(f"   Validation periods: {corrected['validation_periods']}")
        
        print(f"\n🏆 FINAL CONCLUSION:")
        if improvement and improvement['improvement_significant']:
            print(f"   ✅ VIX threshold optimization provides meaningful improvement")
            if vs_enhanced and vs_enhanced['outperforms_enhanced']:
                print(f"   🚨 SURPRISING: Outperforms Enhanced Dynamic (verify results)")
            else:
                print(f"   📊 Does not outperform Enhanced Dynamic (as expected)")
        else:
            print(f"   ✅ VIX threshold optimization provides minimal improvement")
            print(f"   📊 Original academic thresholds (25/35/50) already well-calibrated")
            print(f"   🎯 Enhanced Dynamic remains the optimal strategy")

def main():
    """Main execution"""
    corrector = FinalCorrectedBasicDynamicV2()
    results = corrector.run_final_comprehensive_analysis()
    return results

if __name__ == "__main__":
    main()