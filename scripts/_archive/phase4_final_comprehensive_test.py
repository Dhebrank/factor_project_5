"""
Phase 4 Final Comprehensive End-to-End Test
Ensures the main Phase 4 implementation works perfectly and produces expected outputs
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the main analyzer
from business_cycle_factor_analysis import BusinessCycleFactorAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase4FinalTester:
    """Final comprehensive test for Phase 4 implementation"""
    
    def __init__(self):
        self.results_dir = Path("results/business_cycle_analysis")
        self.analyzer = BusinessCycleFactorAnalyzer()
        
    def setup_analyzer(self):
        """Setup analyzer with required data"""
        logger.info("Setting up analyzer for Phase 4 testing...")
        
        # Load data
        self.analyzer.load_data()
        
        # Load aligned data
        aligned_file = self.results_dir / 'aligned_master_dataset_FIXED.csv'
        if aligned_file.exists():
            self.analyzer.aligned_data = pd.read_csv(aligned_file, index_col=0, parse_dates=True)
            logger.info(f"✓ Loaded aligned data: {self.analyzer.aligned_data.shape}")
        else:
            logger.error("❌ Aligned data not found")
            return False
            
        # Load performance metrics
        perf_file = self.results_dir / 'phase2_performance_analysis.json'
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                self.analyzer.performance_metrics = json.load(f)
            logger.info("✓ Loaded performance metrics")
        else:
            logger.error("❌ Performance metrics not found")
            return False
            
        return True
        
    def test_phase4_step_4_1(self):
        """Test Phase 4.1: Regime Transition Analytics"""
        logger.info("🧪 Testing Phase 4.1: Regime Transition Analytics")
        
        test_results = {
            'step': 'Phase 4.1',
            'substeps': {},
            'success': False
        }
        
        try:
            # Test Step 4.1a: Transition probability matrix
            transition_analysis = self.analyzer._create_transition_probability_matrix()
            
            test_results['substeps']['4.1a'] = {
                'transition_probabilities_created': len(transition_analysis.get('transition_probabilities', {})) > 0,
                'expected_durations_calculated': len(transition_analysis.get('expected_durations', {})) > 0,
                'early_warning_signals': len(transition_analysis.get('early_warning_signals', {})) > 0,
                'regime_stability_metrics': len(transition_analysis.get('regime_stability', {})) > 0
            }
            
            # Test Step 4.1b: Performance during regime changes
            transition_performance = self.analyzer._analyze_transition_performance()
            
            test_results['substeps']['4.1b'] = {
                'transition_performance_analyzed': len(transition_performance) > 0,
                'statistical_significance_tested': any('statistical_significance' in data for data in transition_performance.values()),
                'volatility_analysis_included': any('volatility_change' in data for data in transition_performance.values())
            }
            
            # Overall Phase 4.1 success
            all_4_1a_tests = all(test_results['substeps']['4.1a'].values())
            all_4_1b_tests = all(test_results['substeps']['4.1b'].values())
            test_results['success'] = all_4_1a_tests and all_4_1b_tests
            
            logger.info(f"✓ Phase 4.1 Test Result: {'PASS' if test_results['success'] else 'FAIL'}")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Phase 4.1 test failed: {e}")
            
        return test_results
        
    def test_phase4_step_4_2(self):
        """Test Phase 4.2: Cyclical Pattern Detection"""
        logger.info("🧪 Testing Phase 4.2: Cyclical Pattern Detection")
        
        test_results = {
            'step': 'Phase 4.2',
            'substeps': {},
            'success': False
        }
        
        try:
            # Test Step 4.2a: Intra-regime performance evolution
            intra_regime_analysis = self.analyzer._analyze_intra_regime_evolution()
            
            test_results['substeps']['4.2a'] = {
                'intra_regime_analysis_created': len(intra_regime_analysis) > 0,
                'early_middle_late_phases': any('early_phase_performance' in data.get(factor, {}) 
                                               for data in intra_regime_analysis.values() 
                                               for factor in ['Value', 'Quality', 'MinVol', 'Momentum'] 
                                               if factor in data),
                'optimal_phase_identification': any('optimal_phase' in data.get(factor, {}) 
                                                  for data in intra_regime_analysis.values() 
                                                  for factor in ['Value', 'Quality', 'MinVol', 'Momentum'] 
                                                  if factor in data),
                'regime_maturity_indicators': any('regime_maturity_indicators' in data 
                                                for data in intra_regime_analysis.values())
            }
            
            # Test Step 4.2b: Macro-factor relationships
            macro_factor_analysis = self.analyzer._analyze_macro_factor_relationships()
            
            test_results['substeps']['4.2b'] = {
                'macro_factor_relationships': len(macro_factor_analysis.get('regime_specific_relationships', {})) > 0,
                'cross_regime_sensitivity': len(macro_factor_analysis.get('cross_regime_sensitivity', {})) > 0,
                'correlation_analysis': any('correlation' in factor_data.get(macro_var, {}) 
                                          for regime_data in macro_factor_analysis.get('regime_specific_relationships', {}).values()
                                          for factor_data in regime_data.values()
                                          for macro_var in factor_data.keys()),
                'beta_sensitivity_analysis': any('beta_sensitivity' in factor_data.get(macro_var, {}) 
                                                for regime_data in macro_factor_analysis.get('regime_specific_relationships', {}).values()
                                                for factor_data in regime_data.values()
                                                for macro_var in factor_data.keys())
            }
            
            # Overall Phase 4.2 success
            all_4_2a_tests = all(test_results['substeps']['4.2a'].values())
            all_4_2b_tests = all(test_results['substeps']['4.2b'].values())
            test_results['success'] = all_4_2a_tests and all_4_2b_tests
            
            logger.info(f"✓ Phase 4.2 Test Result: {'PASS' if test_results['success'] else 'FAIL'}")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Phase 4.2 test failed: {e}")
            
        return test_results
        
    def test_phase4_step_4_3(self):
        """Test Phase 4.3: Portfolio Construction Insights"""
        logger.info("🧪 Testing Phase 4.3: Portfolio Construction Insights")
        
        test_results = {
            'step': 'Phase 4.3',
            'substeps': {},
            'success': False
        }
        
        try:
            # Test Step 4.3a: Regime-aware allocation frameworks
            allocation_frameworks = self.analyzer._create_allocation_frameworks()
            
            test_results['substeps']['4.3a'] = {
                'allocation_frameworks_created': len(allocation_frameworks.get('regime_specific_allocations', {})) > 0,
                'risk_parity_allocations': any('risk_parity' in regime_data 
                                             for regime_data in allocation_frameworks.get('regime_specific_allocations', {}).values()),
                'sharpe_optimized_allocations': any('sharpe_optimized' in regime_data 
                                                  for regime_data in allocation_frameworks.get('regime_specific_allocations', {}).values()),
                'equal_weight_allocations': any('equal_weight' in regime_data 
                                              for regime_data in allocation_frameworks.get('regime_specific_allocations', {}).values()),
                'dynamic_recommendations': len(allocation_frameworks.get('dynamic_recommendations', {})) > 0
            }
            
            # Test Step 4.3b: Factor timing models
            timing_models = self.analyzer._develop_timing_models()
            
            test_results['substeps']['4.3b'] = {
                'momentum_signals_created': len(timing_models.get('momentum_signals', {})) > 0,
                'mean_reversion_signals': len(timing_models.get('mean_reversion_signals', {})) > 0,
                'regime_persistence_analysis': len(timing_models.get('regime_persistence', {})) > 0,
                'strategy_performance_attribution': len(timing_models.get('strategy_performance', {})) > 0,
                'current_regime_identification': timing_models.get('current_regime') != 'Unknown'
            }
            
            # Overall Phase 4.3 success
            all_4_3a_tests = all(test_results['substeps']['4.3a'].values())
            all_4_3b_tests = all(test_results['substeps']['4.3b'].values())
            test_results['success'] = all_4_3a_tests and all_4_3b_tests
            
            logger.info(f"✓ Phase 4.3 Test Result: {'PASS' if test_results['success'] else 'FAIL'}")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Phase 4.3 test failed: {e}")
            
        return test_results
        
    def test_full_phase4_execution(self):
        """Test full Phase 4 execution"""
        logger.info("🧪 Testing Full Phase 4 Execution")
        
        test_results = {
            'step': 'Full Phase 4',
            'success': False,
            'files_created': []
        }
        
        try:
            # Run full Phase 4
            phase4_success = self.analyzer.run_phase4()
            
            # Check expected output files
            expected_files = [
                'phase4_regime_transition_analytics.json',
                'phase4_cyclical_pattern_detection.json', 
                'phase4_portfolio_construction_insights.json',
                'phase4_complete_summary.json'
            ]
            
            files_exist = []
            for filename in expected_files:
                file_path = self.results_dir / filename
                if file_path.exists():
                    files_exist.append(filename)
                    test_results['files_created'].append(filename)
            
            test_results['success'] = phase4_success and len(files_exist) == len(expected_files)
            test_results['files_created_count'] = len(files_exist)
            test_results['expected_files_count'] = len(expected_files)
            
            logger.info(f"✓ Full Phase 4 Test Result: {'PASS' if test_results['success'] else 'FAIL'}")
            logger.info(f"✓ Files Created: {len(files_exist)}/{len(expected_files)}")
            
        except Exception as e:
            test_results['error'] = str(e)
            logger.error(f"❌ Full Phase 4 test failed: {e}")
            
        return test_results
        
    def validate_output_content(self):
        """Validate the content of Phase 4 output files"""
        logger.info("🧪 Validating Phase 4 Output Content")
        
        validation_results = {
            'files_validated': {},
            'overall_success': False
        }
        
        # Validate transition analytics file
        transition_file = self.results_dir / 'phase4_regime_transition_analytics.json'
        if transition_file.exists():
            with open(transition_file, 'r') as f:
                transition_data = json.load(f)
            
            validation_results['files_validated']['transition_analytics'] = {
                'has_transition_probabilities': 'transition_probabilities' in transition_data,
                'has_transition_performance': 'transition_performance' in transition_data,
                'has_timestamp': 'analysis_timestamp' in transition_data
            }
        
        # Validate cyclical patterns file
        cyclical_file = self.results_dir / 'phase4_cyclical_pattern_detection.json'
        if cyclical_file.exists():
            with open(cyclical_file, 'r') as f:
                cyclical_data = json.load(f)
            
            validation_results['files_validated']['cyclical_patterns'] = {
                'has_intra_regime_evolution': 'intra_regime_evolution' in cyclical_data,
                'has_macro_factor_relationships': 'macro_factor_relationships' in cyclical_data,
                'has_timestamp': 'analysis_timestamp' in cyclical_data
            }
        
        # Validate portfolio insights file
        portfolio_file = self.results_dir / 'phase4_portfolio_construction_insights.json'
        if portfolio_file.exists():
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
            
            validation_results['files_validated']['portfolio_insights'] = {
                'has_allocation_frameworks': 'allocation_frameworks' in portfolio_data,
                'has_timing_models': 'timing_models' in portfolio_data,
                'has_timestamp': 'analysis_timestamp' in portfolio_data
            }
        
        # Validate summary file
        summary_file = self.results_dir / 'phase4_complete_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            validation_results['files_validated']['summary'] = {
                'has_completion_info': 'phase4_completion' in summary_data,
                'has_analysis_files': 'analysis_files_created' in summary_data,
                'has_key_insights': 'key_insights' in summary_data
            }
        
        # Overall validation success
        all_files_valid = all(
            all(checks.values()) for checks in validation_results['files_validated'].values()
        )
        validation_results['overall_success'] = all_files_valid
        
        logger.info(f"✓ Content Validation: {'PASS' if all_files_valid else 'FAIL'}")
        
        return validation_results
        
    def run_comprehensive_test(self):
        """Run comprehensive Phase 4 test"""
        logger.info("=" * 80)
        logger.info("🔬 PHASE 4 FINAL COMPREHENSIVE END-TO-END TEST")
        logger.info("=" * 80)
        
        # Setup
        if not self.setup_analyzer():
            logger.error("❌ Failed to setup analyzer")
            return False
        
        # Run tests
        test_4_1 = self.test_phase4_step_4_1()
        test_4_2 = self.test_phase4_step_4_2()
        test_4_3 = self.test_phase4_step_4_3()
        full_test = self.test_full_phase4_execution()
        content_validation = self.validate_output_content()
        
        # Overall assessment
        all_tests = [test_4_1, test_4_2, test_4_3, full_test]
        passed_tests = sum(1 for test in all_tests if test['success'])
        overall_success = (passed_tests == len(all_tests)) and content_validation['overall_success']
        
        # Compile results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'testing_type': 'FINAL_COMPREHENSIVE_END_TO_END',
            'test_results': {
                'phase_4_1': test_4_1,
                'phase_4_2': test_4_2,
                'phase_4_3': test_4_3,
                'full_execution': full_test
            },
            'content_validation': content_validation,
            'overall_assessment': {
                'tests_passed': passed_tests,
                'total_tests': len(all_tests),
                'content_validation_passed': content_validation['overall_success'],
                'overall_success': overall_success,
                'phase5_ready': overall_success
            }
        }
        
        # Save results
        with open(self.results_dir / 'phase4_final_comprehensive_test_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("📊 FINAL COMPREHENSIVE TEST RESULTS:")
        logger.info(f"✅ Phase 4.1 (Transition Analytics): {'PASS' if test_4_1['success'] else 'FAIL'}")
        logger.info(f"✅ Phase 4.2 (Cyclical Patterns): {'PASS' if test_4_2['success'] else 'FAIL'}")
        logger.info(f"✅ Phase 4.3 (Portfolio Insights): {'PASS' if test_4_3['success'] else 'FAIL'}")
        logger.info(f"✅ Full Phase 4 Execution: {'PASS' if full_test['success'] else 'FAIL'}")
        logger.info(f"✅ Content Validation: {'PASS' if content_validation['overall_success'] else 'FAIL'}")
        logger.info(f"📈 Overall Success Rate: {passed_tests}/{len(all_tests)} + Content")
        logger.info(f"🎯 Overall Success: {overall_success}")
        logger.info(f"🚀 Ready for Phase 5: {overall_success}")
        logger.info("=" * 80)
        
        return final_results

def main():
    """Run final comprehensive Phase 4 test"""
    tester = Phase4FinalTester()
    results = tester.run_comprehensive_test()
    
    print(f"\n🎯 PHASE 4 FINAL COMPREHENSIVE TEST SUMMARY:")
    print(f"✅ Overall Success: {results['overall_assessment']['overall_success']}")
    print(f"🧪 Tests Passed: {results['overall_assessment']['tests_passed']}/{results['overall_assessment']['total_tests']}")
    print(f"📝 Content Validation: {results['overall_assessment']['content_validation_passed']}")
    print(f"🚀 Phase 5 Ready: {results['overall_assessment']['phase5_ready']}")
    
    return results

if __name__ == "__main__":
    main() 