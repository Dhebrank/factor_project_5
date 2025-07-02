"""
Final Comprehensive Phase 2 Verification
Ultimate validation before proceeding to Phase 3
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from business_cycle_factor_analysis import BusinessCycleFactorAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalPhase2Verification:
    """
    Final comprehensive verification of all Phase 2 requirements
    """
    
    def __init__(self):
        self.analyzer = BusinessCycleFactorAnalyzer()
        self.results_dir = Path("results/business_cycle_analysis")
        self.verification_results = {}
        
        logger.info("Final Phase 2 Comprehensive Verification initialized")
    
    def verify_all_output_files_exist_and_valid(self):
        """
        Verify all required output files exist and contain valid data
        """
        logger.info("=== VERIFYING ALL OUTPUT FILES ===")
        
        required_files = {
            'phase2_regime_analysis.json': 'Multi-dimensional regime analysis',
            'phase2_performance_analysis.json': 'Factor performance deep-dive',
            'phase2_complete_summary.json': 'Comprehensive Phase 2 summary',
            'phase2_verification_report.json': '100% verification results',
            'roadmap_compliance_verification.json': 'Roadmap compliance verification',
            'aligned_master_dataset_FIXED.csv': 'Fixed aligned master dataset',
            'factor_returns_aligned_FIXED.csv': 'Fixed factor returns',
            'regime_classifications_FIXED.csv': 'Fixed regime classifications',
            'regime_methodology.json': 'Regime methodology documentation'
        }
        
        file_verification = {}
        
        for filename, description in required_files.items():
            file_path = self.results_dir / filename
            
            file_verification[filename] = {
                'exists': file_path.exists(),
                'description': description,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'valid_content': False
            }
            
            if file_path.exists():
                try:
                    if filename.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        file_verification[filename]['valid_content'] = len(data) > 0
                        file_verification[filename]['keys_count'] = len(data.keys()) if isinstance(data, dict) else 1
                    
                    elif filename.endswith('.csv'):
                        data = pd.read_csv(file_path)
                        file_verification[filename]['valid_content'] = len(data) > 0
                        file_verification[filename]['shape'] = list(data.shape)
                        file_verification[filename]['columns_count'] = len(data.columns)
                
                except Exception as e:
                    file_verification[filename]['error'] = str(e)
        
        self.verification_results['output_files'] = file_verification
        
        # Summary
        files_exist = sum(1 for f in file_verification.values() if f['exists'])
        files_valid = sum(1 for f in file_verification.values() if f.get('valid_content', False))
        
        logger.info(f"Output Files: {files_exist}/{len(required_files)} exist, {files_valid}/{len(required_files)} valid")
        
        return files_exist == len(required_files) and files_valid == len(required_files)
    
    def verify_phase2_functionality_complete(self):
        """
        Verify all Phase 2 functionality is complete and working
        """
        logger.info("=== VERIFYING PHASE 2 FUNCTIONALITY ===")
        
        # Initialize and run full Phase 2
        success = self.analyzer.load_data()
        if not success:
            return False
        
        phase1_success = self.analyzer.run_phase1()
        if not phase1_success:
            return False
        
        phase2_success = self.analyzer.run_phase2()
        if not phase2_success:
            return False
        
        # Verify specific functionality
        functionality_tests = {}
        
        # Test regime analysis
        try:
            regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
            functionality_tests['regime_analysis'] = {
                'success': True,
                'regimes_found': len(regime_analysis['regime_statistics']),
                'transitions_analyzed': len(regime_analysis.get('regime_runs_detail', []))
            }
        except Exception as e:
            functionality_tests['regime_analysis'] = {'success': False, 'error': str(e)}
        
        # Test economic validation
        try:
            economic_validation = self.analyzer._validate_economic_signals_by_regime()
            functionality_tests['economic_validation'] = {
                'success': True,
                'regimes_validated': len(economic_validation['regime_validations']),
                'indicators_compared': len(economic_validation['cross_regime_comparisons'])
            }
        except Exception as e:
            functionality_tests['economic_validation'] = {'success': False, 'error': str(e)}
        
        # Test performance metrics
        try:
            performance_metrics = self.analyzer._calculate_comprehensive_performance_metrics()
            functionality_tests['performance_metrics'] = {
                'success': True,
                'regimes_analyzed': len(performance_metrics),
                'factors_per_regime': {regime: len(factors) for regime, factors in performance_metrics.items()}
            }
        except Exception as e:
            functionality_tests['performance_metrics'] = {'success': False, 'error': str(e)}
        
        # Test statistical significance
        try:
            statistical_tests = self.analyzer._run_statistical_significance_tests()
            functionality_tests['statistical_tests'] = {
                'success': True,
                'anova_tests': len(statistical_tests.get('anova_tests', {})),
                'bootstrap_intervals': len(statistical_tests.get('bootstrap_confidence_intervals', {})),
                'transition_impact': 'regime_transition_impact' in statistical_tests
            }
        except Exception as e:
            functionality_tests['statistical_tests'] = {'success': False, 'error': str(e)}
        
        self.verification_results['functionality'] = functionality_tests
        
        all_success = all(test.get('success', False) for test in functionality_tests.values())
        logger.info(f"Functionality Tests: {sum(1 for t in functionality_tests.values() if t.get('success', False))}/{len(functionality_tests)} passed")
        
        return all_success
    
    def verify_data_quality_and_integrity(self):
        """
        Verify data quality and integrity across all datasets
        """
        logger.info("=== VERIFYING DATA QUALITY & INTEGRITY ===")
        
        quality_checks = {}
        
        # Check aligned data quality
        if self.analyzer.aligned_data is not None:
            aligned_data = self.analyzer.aligned_data
            
            quality_checks['aligned_data'] = {
                'shape': list(aligned_data.shape),
                'date_range': f"{aligned_data.index.min()} to {aligned_data.index.max()}",
                'total_nulls': int(aligned_data.isnull().sum().sum()),
                'regime_diversity': aligned_data['ECONOMIC_REGIME'].value_counts().to_dict(),
                'factor_completeness': {
                    factor: float((~aligned_data[factor].isnull()).mean()) 
                    for factor in ['Value', 'Quality', 'MinVol', 'Momentum'] 
                    if factor in aligned_data.columns
                }
            }
            
            # Check regime distribution is reasonable
            regime_counts = aligned_data['ECONOMIC_REGIME'].value_counts()
            quality_checks['regime_quality'] = {
                'total_regimes': len(regime_counts),
                'min_regime_count': int(regime_counts.min()),
                'max_regime_count': int(regime_counts.max()),
                'regime_balance_ratio': float(regime_counts.min() / regime_counts.max()),
                'all_regimes_present': len(regime_counts) >= 4
            }
            
            # Check factor data quality
            factors = ['Value', 'Quality', 'MinVol', 'Momentum']
            factor_quality = {}
            
            for factor in factors:
                if factor in aligned_data.columns:
                    factor_data = aligned_data[factor].dropna()
                    factor_quality[factor] = {
                        'completeness': float(len(factor_data) / len(aligned_data)),
                        'mean_return': float(factor_data.mean()),
                        'volatility': float(factor_data.std()),
                        'min_return': float(factor_data.min()),
                        'max_return': float(factor_data.max()),
                        'reasonable_range': abs(factor_data.mean()) < 1 and factor_data.std() < 1  # Assuming monthly returns
                    }
            
            quality_checks['factor_quality'] = factor_quality
        
        self.verification_results['data_quality'] = quality_checks
        
        # Assess overall data quality
        data_quality_pass = (
            quality_checks.get('aligned_data', {}).get('shape', [0, 0])[0] > 300 and  # Reasonable data size
            quality_checks.get('regime_quality', {}).get('all_regimes_present', False) and  # All regimes present
            all(fq.get('reasonable_range', False) for fq in quality_checks.get('factor_quality', {}).values())  # Reasonable factor ranges
        )
        
        logger.info(f"Data Quality: {'PASS' if data_quality_pass else 'FAIL'}")
        return data_quality_pass
    
    def verify_roadmap_exact_compliance(self):
        """
        Verify exact compliance with roadmap requirements
        """
        logger.info("=== VERIFYING EXACT ROADMAP COMPLIANCE ===")
        
        # Load compliance verification results
        compliance_file = self.results_dir / 'roadmap_compliance_verification.json'
        
        if compliance_file.exists():
            with open(compliance_file, 'r') as f:
                compliance_data = json.load(f)
            
            roadmap_compliance = {
                'total_requirements': compliance_data.get('total_requirements', 0),
                'requirements_passed': compliance_data.get('requirements_passed', 0),
                'compliance_percentage': compliance_data.get('compliance_percentage', 0),
                'overall_compliance': compliance_data.get('overall_compliance', 'UNKNOWN'),
                'step_results': compliance_data.get('step_results', {})
            }
            
            self.verification_results['roadmap_compliance'] = roadmap_compliance
            
            full_compliance = (
                roadmap_compliance['compliance_percentage'] == 100.0 and
                roadmap_compliance['overall_compliance'] == 'FULL COMPLIANCE'
            )
            
            logger.info(f"Roadmap Compliance: {roadmap_compliance['requirements_passed']}/{roadmap_compliance['total_requirements']} ({roadmap_compliance['compliance_percentage']:.1f}%)")
            return full_compliance
        else:
            logger.error("Roadmap compliance file not found")
            return False
    
    def create_phase3_readiness_assessment(self):
        """
        Create comprehensive Phase 3 readiness assessment
        """
        logger.info("=== CREATING PHASE 3 READINESS ASSESSMENT ===")
        
        # Assess each dependency for Phase 3
        output_files = self.verification_results.get('output_files', {})
        
        phase3_dependencies = {
            'regime_data_ready': {
                'required': 'Phase 2 regime analysis data',
                'status': output_files.get('phase2_regime_analysis.json', {}).get('exists', False),
                'critical': True
            },
            'performance_metrics_ready': {
                'required': 'Phase 2 performance metrics data',
                'status': output_files.get('phase2_performance_analysis.json', {}).get('exists', False),
                'critical': True
            },
            'statistical_significance_ready': {
                'required': 'Statistical significance data for visualizations',
                'status': any('anova_tests' in str(test) for test in self.verification_results.get('functionality', {}).values()),
                'critical': True
            },
            'factor_returns_aligned': {
                'required': 'Aligned factor returns data',
                'status': output_files.get('factor_returns_aligned_FIXED.csv', {}).get('exists', False),
                'critical': True
            },
            'regime_classifications_fixed': {
                'required': 'Fixed regime classifications',
                'status': output_files.get('regime_classifications_FIXED.csv', {}).get('exists', False),
                'critical': True
            },
            'data_quality_validated': {
                'required': 'Data quality validation passed',
                'status': self.verification_results.get('data_quality', {}) != {},
                'critical': True
            }
        }
        
        # Calculate readiness score
        critical_deps = {k: v for k, v in phase3_dependencies.items() if v['critical']}
        readiness_score = sum(1 for dep in critical_deps.values() if dep['status']) / len(critical_deps)
        
        readiness_assessment = {
            'overall_readiness_score': readiness_score,
            'ready_for_phase3': readiness_score == 1.0,
            'dependencies': phase3_dependencies,
            'critical_dependencies_met': sum(1 for dep in critical_deps.values() if dep['status']),
            'total_critical_dependencies': len(critical_deps),
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        self.verification_results['phase3_readiness'] = readiness_assessment
        
        logger.info(f"Phase 3 Readiness: {readiness_assessment['critical_dependencies_met']}/{readiness_assessment['total_critical_dependencies']} critical dependencies met")
        return readiness_assessment['ready_for_phase3']
    
    def run_comprehensive_final_verification(self):
        """
        Run comprehensive final verification of everything
        """
        logger.info("=" * 80)
        logger.info("FINAL COMPREHENSIVE PHASE 2 VERIFICATION")
        logger.info("=" * 80)
        
        verification_steps = [
            ("Output Files", self.verify_all_output_files_exist_and_valid),
            ("Functionality", self.verify_phase2_functionality_complete),
            ("Data Quality", self.verify_data_quality_and_integrity),
            ("Roadmap Compliance", self.verify_roadmap_exact_compliance),
            ("Phase 3 Readiness", self.create_phase3_readiness_assessment)
        ]
        
        verification_results = {}
        overall_success = True
        
        for step_name, verification_func in verification_steps:
            logger.info(f"\n--- Verifying {step_name} ---")
            try:
                step_result = verification_func()
                verification_results[step_name] = step_result
                logger.info(f"{step_name}: {'✅ PASS' if step_result else '❌ FAIL'}")
                
                if not step_result:
                    overall_success = False
            except Exception as e:
                logger.error(f"{step_name}: ❌ ERROR - {e}")
                verification_results[step_name] = False
                overall_success = False
        
        # Create final verification report
        final_report = {
            'verification_timestamp': datetime.now().isoformat(),
            'overall_verification_status': 'COMPLETE_SUCCESS' if overall_success else 'ISSUES_DETECTED',
            'verification_steps': verification_results,
            'detailed_results': self.verification_results,
            'phase2_status': 'FULLY_VERIFIED_AND_READY' if overall_success else 'REQUIRES_ATTENTION',
            'phase3_authorization': 'AUTHORIZED' if overall_success else 'NOT_AUTHORIZED'
        }
        
        # Save final verification report
        with open(self.results_dir / 'final_phase2_verification_report.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL VERIFICATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {'✅ COMPLETE SUCCESS' if overall_success else '❌ ISSUES DETECTED'}")
        logger.info(f"Verification Steps: {sum(verification_results.values())}/{len(verification_results)} passed")
        
        for step_name, result in verification_results.items():
            logger.info(f"  {step_name}: {'✅' if result else '❌'}")
        
        logger.info("=" * 80)
        
        if overall_success:
            logger.info("🎉 PHASE 2 FULLY VERIFIED AND READY!")
            logger.info("🚀 PHASE 3 AUTHORIZATION: GRANTED")
            logger.info("📋 All roadmap requirements met with 100% compliance")
            logger.info("✅ Implementation is smooth and comprehensive")
        else:
            logger.error("⚠️  VERIFICATION ISSUES DETECTED")
            logger.error("❌ PHASE 3 AUTHORIZATION: DENIED")
            logger.error("🔧 Please review and fix issues before proceeding")
        
        logger.info("=" * 80)
        
        return overall_success, final_report

def main():
    """
    Main function for final verification
    """
    verifier = FinalPhase2Verification()
    success, report = verifier.run_comprehensive_final_verification()
    
    if success:
        print("\n✅ FINAL PHASE 2 VERIFICATION: COMPLETE SUCCESS")
        print("🚀 PHASE 3 AUTHORIZED - READY TO PROCEED")
    else:
        print("\n❌ FINAL PHASE 2 VERIFICATION: ISSUES DETECTED") 
        print("⚠️  PHASE 3 NOT AUTHORIZED - REVIEW REQUIRED")
        exit(1)

if __name__ == "__main__":
    main() 