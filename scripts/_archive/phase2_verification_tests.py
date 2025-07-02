"""
Phase 2 Verification Tests: Comprehensive validation of roadmap requirements
Verifies that every Phase 2 sub-step is properly implemented and working
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import sys
import os

# Add the scripts directory to the path so we can import the main analyzer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from business_cycle_factor_analysis import BusinessCycleFactorAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2VerificationTester:
    """
    Comprehensive verification of Phase 2 implementation against roadmap
    """
    
    def __init__(self):
        self.analyzer = BusinessCycleFactorAnalyzer()
        self.results_dir = Path("results/business_cycle_analysis")
        self.test_results = {}
        self.verification_status = {}
        
        logger.info("Phase 2 Verification Tester initialized")
    
    def test_step_2_1a_regime_duration_analysis(self):
        """
        TEST: Step 2.1a - Regime duration and frequency analysis
        Roadmap Requirements:
        - Calculate average regime length by type ✓
        - Analyze regime transitions per decade ✓ 
        - Compute regime stability metrics ✓
        - Calculate transition probability matrices ✓
        - Identify seasonal patterns in regime changes ✓
        - Create regime summary statistics table ✓
        """
        logger.info("=== TESTING STEP 2.1a: Regime Duration Analysis ===")
        
        try:
            # Load Phase 1 data first
            success = self.analyzer.load_data()
            if not success:
                raise Exception("Failed to load data")
            
            # Run Phase 1 to get aligned data
            phase1_success = self.analyzer.run_phase1()
            if not phase1_success:
                raise Exception("Phase 1 failed")
            
            # Test the duration analysis method
            duration_analysis = self.analyzer._analyze_regime_durations_and_transitions()
            
            # Verify roadmap requirements
            tests = {}
            
            # Requirement 1: Calculate average regime length by type
            tests['average_regime_length'] = (
                'regime_statistics' in duration_analysis and
                all(regime in duration_analysis['regime_statistics'] for regime in ['Overheating', 'Stagflation', 'Recession', 'Goldilocks']) and
                all('avg_duration_months' in duration_analysis['regime_statistics'][regime] for regime in duration_analysis['regime_statistics'])
            )
            
            # Requirement 2: Analyze regime transitions per decade
            tests['decade_transitions'] = (
                'decade_transitions' in duration_analysis and
                len(duration_analysis['decade_transitions']) >= 3  # Should have 1990s, 2000s, 2010s, 2020s
            )
            
            # Requirement 3: Compute regime stability metrics
            tests['stability_metrics'] = (
                'regime_statistics' in duration_analysis and
                all('std_duration_months' in duration_analysis['regime_statistics'][regime] for regime in duration_analysis['regime_statistics']) and
                all('frequency_percentage' in duration_analysis['regime_statistics'][regime] for regime in duration_analysis['regime_statistics'])
            )
            
            # Requirement 4: Calculate transition probability matrices
            tests['transition_matrices'] = (
                'transition_matrix_counts' in duration_analysis and
                'transition_probabilities' in duration_analysis and
                len(duration_analysis['transition_probabilities']) >= 4  # 4 regimes
            )
            
            # Requirement 5: Identify seasonal patterns
            tests['seasonal_patterns'] = (
                'seasonal_transition_patterns' in duration_analysis and
                len(duration_analysis['seasonal_transition_patterns']) == 12  # 12 months
            )
            
            # Requirement 6: Create regime summary statistics table
            tests['summary_statistics'] = (
                'total_transitions' in duration_analysis and
                'regime_runs_detail' in duration_analysis
            )
            
            # Verify specific values match expected results
            regime_stats = duration_analysis['regime_statistics']
            tests['expected_frequencies'] = (
                abs(regime_stats['Overheating']['frequency_percentage'] - 39.0) < 1.0 and
                abs(regime_stats['Stagflation']['frequency_percentage'] - 30.5) < 1.0 and
                abs(regime_stats['Recession']['frequency_percentage'] - 17.6) < 1.0 and
                abs(regime_stats['Goldilocks']['frequency_percentage'] - 12.9) < 1.0
            )
            
            self.test_results['step_2_1a'] = tests
            all_passed = all(tests.values())
            
            logger.info(f"Step 2.1a Tests: {sum(tests.values())}/{len(tests)} passed")
            for test_name, result in tests.items():
                logger.info(f"  {test_name}: {'✓' if result else '❌'}")
            
            return all_passed, duration_analysis
            
        except Exception as e:
            logger.error(f"Step 2.1a Test failed: {e}")
            return False, None
    
    def test_step_2_1b_economic_validation(self):
        """
        TEST: Step 2.1b - Economic signal validation per regime
        Roadmap Requirements:
        - Analyze GDP growth rates during each regime ✓
        - Validate inflation trajectory confirmation per regime ✓
        - Examine yield curve behavior by regime ✓
        - Study employment trends (UNRATE, PAYEMS) across regimes ✓
        - Create regime validation report with economic indicators ✓
        - Document regime-indicator relationships ✓
        """
        logger.info("=== TESTING STEP 2.1b: Economic Signal Validation ===")
        
        try:
            # Test the economic validation method
            economic_validation = self.analyzer._validate_economic_signals_by_regime()
            
            tests = {}
            
            # Requirement 1: Analyze GDP growth rates
            tests['gdp_analysis'] = (
                'regime_validations' in economic_validation and
                any('GDPC1' in str(key) or 'GROWTH_COMPOSITE' in str(key) for validation in economic_validation['regime_validations'].values() for key in validation.keys())
            )
            
            # Requirement 2: Validate inflation trajectory
            tests['inflation_analysis'] = (
                'regime_validations' in economic_validation and
                any('INFLATION' in str(key) or 'CPI' in str(key) for validation in economic_validation['regime_validations'].values() for key in validation.keys())
            )
            
            # Requirement 3: Examine yield curve behavior
            tests['yield_curve_analysis'] = (
                'regime_validations' in economic_validation and
                any('DGS' in str(key) or 'TERM_SPREAD' in str(key) for validation in economic_validation['regime_validations'].values() for key in validation.keys())
            )
            
            # Requirement 4: Study employment trends
            tests['employment_analysis'] = (
                'regime_validations' in economic_validation and
                any('UNRATE' in str(key) or 'PAYEMS' in str(key) for validation in economic_validation['regime_validations'].values() for key in validation.keys())
            )
            
            # Requirement 5: Create regime validation report
            tests['validation_report'] = (
                'regime_validations' in economic_validation and
                len(economic_validation['regime_validations']) >= 4 and  # 4 regimes
                all('observations' in validation for validation in economic_validation['regime_validations'].values())
            )
            
            # Requirement 6: Document regime-indicator relationships
            tests['cross_regime_comparisons'] = (
                'cross_regime_comparisons' in economic_validation and
                len(economic_validation['cross_regime_comparisons']) > 0
            )
            
            # Additional validation: VIX analysis
            tests['vix_analysis'] = (
                any('VIX' in validation for validation in economic_validation['regime_validations'].values())
            )
            
            self.test_results['step_2_1b'] = tests
            all_passed = all(tests.values())
            
            logger.info(f"Step 2.1b Tests: {sum(tests.values())}/{len(tests)} passed")
            for test_name, result in tests.items():
                logger.info(f"  {test_name}: {'✓' if result else '❌'}")
            
            return all_passed, economic_validation
            
        except Exception as e:
            logger.error(f"Step 2.1b Test failed: {e}")
            return False, None
    
    def test_step_2_2a_performance_metrics(self):
        """
        TEST: Step 2.2a - Comprehensive performance metrics by regime
        Roadmap Requirements:
        - Calculate absolute returns: Mean, median, std dev per regime ✓
        - Compute risk-adjusted returns: Sharpe, Sortino, Calmar ratios ✓
        - Analyze tail risk: Maximum drawdown, VaR (5%), Expected Shortfall ✓
        - Measure consistency: Win rate, positive months percentage ✓
        - Calculate all metrics for: Value, Quality, MinVol, Momentum, S&P 500 ✓
        - Create performance metrics summary tables ✓
        """
        logger.info("=== TESTING STEP 2.2a: Performance Metrics ===")
        
        try:
            # Test the performance metrics method
            performance_metrics = self.analyzer._calculate_comprehensive_performance_metrics()
            
            tests = {}
            
            # Requirement 1: Calculate absolute returns (mean, median, std dev)
            tests['absolute_returns'] = True
            for regime in performance_metrics:
                for factor in performance_metrics[regime]:
                    if not all(metric in performance_metrics[regime][factor] for metric in ['mean_monthly_return', 'median_monthly_return', 'std_monthly_return']):
                        tests['absolute_returns'] = False
                        break
            
            # Requirement 2: Compute risk-adjusted returns (Sharpe, Sortino, Calmar)
            tests['risk_adjusted_returns'] = True
            for regime in performance_metrics:
                for factor in performance_metrics[regime]:
                    if not all(metric in performance_metrics[regime][factor] for metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']):
                        tests['risk_adjusted_returns'] = False
                        break
            
            # Requirement 3: Analyze tail risk (drawdown, VaR, ES)
            tests['tail_risk_analysis'] = True
            for regime in performance_metrics:
                for factor in performance_metrics[regime]:
                    if not all(metric in performance_metrics[regime][factor] for metric in ['max_drawdown', 'var_5_percent', 'expected_shortfall_5_percent']):
                        tests['tail_risk_analysis'] = False
                        break
            
            # Requirement 4: Measure consistency (win rate, positive months)
            tests['consistency_metrics'] = True
            for regime in performance_metrics:
                for factor in performance_metrics[regime]:
                    if not all(metric in performance_metrics[regime][factor] for metric in ['win_rate', 'positive_months']):
                        tests['consistency_metrics'] = False
                        break
            
            # Requirement 5: Calculate for all factors (Value, Quality, MinVol, Momentum)
            required_factors = ['Value', 'Quality', 'MinVol', 'Momentum']
            tests['all_factors_covered'] = True
            for regime in performance_metrics:
                available_factors = list(performance_metrics[regime].keys())
                if not all(factor in available_factors for factor in required_factors):
                    tests['all_factors_covered'] = False
                    break
            
            # Requirement 6: Performance metrics summary tables (annualized returns)
            tests['annualized_metrics'] = True
            for regime in performance_metrics:
                for factor in performance_metrics[regime]:
                    if not all(metric in performance_metrics[regime][factor] for metric in ['annualized_return', 'annualized_volatility']):
                        tests['annualized_metrics'] = False
                        break
            
            # Verify expected performance patterns from roadmap
            tests['goldilocks_best_performance'] = (
                'Goldilocks' in performance_metrics and
                performance_metrics['Goldilocks']['Value']['sharpe_ratio'] > 2.0
            )
            
            tests['recession_defensive_quality'] = (
                'Recession' in performance_metrics and
                performance_metrics['Recession']['Quality']['sharpe_ratio'] > 0 and
                performance_metrics['Recession']['Value']['sharpe_ratio'] < 0
            )
            
            self.test_results['step_2_2a'] = tests
            all_passed = all(tests.values())
            
            logger.info(f"Step 2.2a Tests: {sum(tests.values())}/{len(tests)} passed")
            for test_name, result in tests.items():
                logger.info(f"  {test_name}: {'✓' if result else '❌'}")
            
            return all_passed, performance_metrics
            
        except Exception as e:
            logger.error(f"Step 2.2a Test failed: {e}")
            return False, None
    
    def test_step_2_2b_statistical_tests(self):
        """
        TEST: Step 2.2b - Statistical significance testing
        Roadmap Requirements:
        - Implement ANOVA tests for performance differences across regimes ✓
        - Run pairwise t-tests comparing each factor vs S&P 500 by regime ✓
        - Generate bootstrap confidence intervals for robust performance bands ✓
        - Analyze regime change impact on performance during transition periods ✓
        - Document statistical significance results ✓
        - Create significance indicator system for visualizations ✓
        """
        logger.info("=== TESTING STEP 2.2b: Statistical Significance Testing ===")
        
        try:
            # Test the statistical testing method
            statistical_tests = self.analyzer._run_statistical_significance_tests()
            
            tests = {}
            
            # Requirement 1: ANOVA tests for regime differences
            tests['anova_tests'] = (
                'anova_tests' in statistical_tests and
                len(statistical_tests['anova_tests']) > 0 and
                all('f_statistic' in test and 'p_value' in test for test in statistical_tests['anova_tests'].values() if 'error' not in test)
            )
            
            # Requirement 2: Pairwise t-tests vs S&P 500
            tests['pairwise_ttests'] = (
                'pairwise_tests' in statistical_tests and
                len(statistical_tests['pairwise_tests']) > 0
            )
            
            # Requirement 3: Bootstrap confidence intervals
            tests['bootstrap_confidence'] = (
                'bootstrap_confidence_intervals' in statistical_tests and
                len(statistical_tests['bootstrap_confidence_intervals']) > 0 and
                any('ci_lower' in bootstrap and 'ci_upper' in bootstrap 
                    for regime_data in statistical_tests['bootstrap_confidence_intervals'].values() 
                    for bootstrap in regime_data.values() if 'error' not in bootstrap)
            )
            
            # Requirement 4: Regime transition impact analysis
            tests['transition_impact'] = (
                'regime_transition_impact' in statistical_tests and
                len(statistical_tests['regime_transition_impact']) > 0 and
                any('pre_transition_mean' in analysis and 'post_transition_mean' in analysis
                    for analysis in statistical_tests['regime_transition_impact'].values() if 'error' not in analysis)
            )
            
            # Requirement 5: Document statistical significance
            tests['significance_documentation'] = (
                all(test_type in statistical_tests for test_type in ['anova_tests', 'pairwise_tests', 'bootstrap_confidence_intervals'])
            )
            
            # Requirement 6: Significance indicator system
            tests['significance_indicators'] = (
                any('significant' in test for test in statistical_tests['anova_tests'].values() if 'error' not in test) if 'anova_tests' in statistical_tests else False
            )
            
            # Additional verification: 147 transitions analyzed
            if 'regime_transition_impact' in statistical_tests:
                for factor_analysis in statistical_tests['regime_transition_impact'].values():
                    if 'total_transitions_analyzed' in factor_analysis:
                        tests['correct_transition_count'] = factor_analysis['total_transitions_analyzed'] == 147
                        break
                else:
                    tests['correct_transition_count'] = False
            else:
                tests['correct_transition_count'] = False
            
            self.test_results['step_2_2b'] = tests
            all_passed = all(tests.values())
            
            logger.info(f"Step 2.2b Tests: {sum(tests.values())}/{len(tests)} passed")
            for test_name, result in tests.items():
                logger.info(f"  {test_name}: {'✓' if result else '❌'}")
            
            return all_passed, statistical_tests
            
        except Exception as e:
            logger.error(f"Step 2.2b Test failed: {e}")
            return False, None
    
    def test_output_files_generation(self):
        """
        TEST: Verify all required output files are generated
        Roadmap Requirements:
        - phase2_regime_analysis.json
        - phase2_performance_analysis.json  
        - phase2_complete_summary.json
        """
        logger.info("=== TESTING OUTPUT FILE GENERATION ===")
        
        # Run complete Phase 2
        phase2_success = self.analyzer.run_phase2()
        
        tests = {}
        
        # Check file existence and content
        required_files = [
            'phase2_regime_analysis.json',
            'phase2_performance_analysis.json', 
            'phase2_complete_summary.json'
        ]
        
        for filename in required_files:
            file_path = self.results_dir / filename
            tests[f'{filename}_exists'] = file_path.exists()
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    tests[f'{filename}_valid_json'] = len(data) > 0
                    tests[f'{filename}_has_content'] = any(len(str(v)) > 10 for v in data.values() if isinstance(v, (dict, list, str)))
                except:
                    tests[f'{filename}_valid_json'] = False
                    tests[f'{filename}_has_content'] = False
        
        # Verify Phase 2 completion status
        tests['phase2_execution_success'] = phase2_success
        
        self.test_results['output_files'] = tests
        all_passed = all(tests.values())
        
        logger.info(f"Output Files Tests: {sum(tests.values())}/{len(tests)} passed")
        for test_name, result in tests.items():
            logger.info(f"  {test_name}: {'✓' if result else '❌'}")
        
        return all_passed
    
    def run_comprehensive_verification(self):
        """
        Run all Phase 2 verification tests and generate comprehensive report
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE PHASE 2 VERIFICATION")
        logger.info("=" * 80)
        
        verification_results = {}
        
        # Test Step 2.1a: Regime Duration Analysis
        step_2_1a_pass, _ = self.test_step_2_1a_regime_duration_analysis()
        verification_results['step_2_1a'] = step_2_1a_pass
        
        # Test Step 2.1b: Economic Signal Validation
        step_2_1b_pass, _ = self.test_step_2_1b_economic_validation()
        verification_results['step_2_1b'] = step_2_1b_pass
        
        # Test Step 2.2a: Performance Metrics
        step_2_2a_pass, _ = self.test_step_2_2a_performance_metrics()
        verification_results['step_2_2a'] = step_2_2a_pass
        
        # Test Step 2.2b: Statistical Testing
        step_2_2b_pass, _ = self.test_step_2_2b_statistical_tests()
        verification_results['step_2_2b'] = step_2_2b_pass
        
        # Test Output Files
        output_files_pass = self.test_output_files_generation()
        verification_results['output_files'] = output_files_pass
        
        # Calculate overall score
        total_tests = sum(len(step_tests) for step_tests in self.test_results.values())
        passed_tests = sum(sum(step_tests.values()) for step_tests in self.test_results.values())
        
        overall_pass = all(verification_results.values())
        
        # Generate verification report
        verification_report = {
            'verification_timestamp': pd.Timestamp.now().isoformat(),
            'overall_status': 'PASSED' if overall_pass else 'FAILED',
            'overall_score': f"{passed_tests}/{total_tests}",
            'step_results': verification_results,
            'detailed_test_results': self.test_results,
            'roadmap_compliance': {
                'step_2_1a_requirements': 7,  # 6 main + 1 verification
                'step_2_1b_requirements': 7,  # 6 main + 1 additional
                'step_2_2a_requirements': 8,  # 6 main + 2 verification
                'step_2_2b_requirements': 7,  # 6 main + 1 verification
                'total_requirements': 29
            }
        }
        
        # Save verification report
        with open(self.results_dir / 'phase2_verification_report.json', 'w') as f:
            json.dump(verification_report, f, indent=2, default=str)
        
        # Print final summary
        logger.info("=" * 80)
        logger.info("PHASE 2 VERIFICATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
        logger.info(f"Test Score: {passed_tests}/{total_tests}")
        logger.info(f"Step 2.1a (Regime Analysis): {'✅' if step_2_1a_pass else '❌'}")
        logger.info(f"Step 2.1b (Economic Validation): {'✅' if step_2_1b_pass else '❌'}")
        logger.info(f"Step 2.2a (Performance Metrics): {'✅' if step_2_2a_pass else '❌'}")
        logger.info(f"Step 2.2b (Statistical Testing): {'✅' if step_2_2b_pass else '❌'}")
        logger.info(f"Output Files Generation: {'✅' if output_files_pass else '❌'}")
        logger.info("=" * 80)
        
        if overall_pass:
            logger.info("🚀 PHASE 2 FULLY VERIFIED - READY FOR PHASE 3!")
        else:
            logger.error("⚠️  PHASE 2 VERIFICATION ISSUES DETECTED - REVIEW REQUIRED")
        
        return overall_pass, verification_report

def main():
    """
    Run comprehensive Phase 2 verification
    """
    tester = Phase2VerificationTester()
    success, report = tester.run_comprehensive_verification()
    
    if success:
        print("\n🎉 ALL PHASE 2 REQUIREMENTS VERIFIED!")
        print("📋 Implementation follows roadmap to the letter")
        print("✅ Ready to proceed to Phase 3: Advanced Visualization Suite")
    else:
        print("\n⚠️  PHASE 2 VERIFICATION ISSUES FOUND")
        print("📋 Review verification report for details")
        print("🔧 Fix issues before proceeding to Phase 3")

if __name__ == "__main__":
    main() 