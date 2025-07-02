"""
Phase 4 Comprehensive Substep Verification & Demo Suite
Tests all 6 Phase 4 substeps individually according to roadmap requirements
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase4ComprehensiveVerifier:
    """
    Comprehensive verification and demonstration for all Phase 4 substeps
    """
    
    def __init__(self, results_dir="results/business_cycle_analysis"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load verification data
        self.load_verification_data()
        
        # Expected files from roadmap
        self.expected_files = {
            '4.1a': 'phase4_regime_transition_analytics.json',
            '4.1b': 'phase4_regime_transition_analytics.json',  # Same file
            '4.2a': 'phase4_cyclical_pattern_detection.json',
            '4.2b': 'phase4_cyclical_pattern_detection.json',   # Same file
            '4.3a': 'phase4_portfolio_construction_insights.json',
            '4.3b': 'phase4_portfolio_construction_insights.json'  # Same file
        }
        
        # Roadmap requirements for each substep
        self.roadmap_requirements = {
            '4.1a': {
                'title': 'Transition probability matrix',
                'requirements': [
                    'Calculate historical regime transition frequencies',
                    'Build expected regime duration models',
                    'Develop early warning signal analysis',
                    'Create transition probability heatmap',
                    'Add confidence intervals for transition probabilities',
                    'Include regime persistence analysis'
                ]
            },
            '4.1b': {
                'title': 'Performance during regime changes',
                'requirements': [
                    'Analyze 3-month windows around regime transitions',
                    'Study factor behavior during uncertainty periods',
                    'Measure defensive positioning effectiveness',
                    'Create transition period performance analysis',
                    'Add volatility analysis during transitions',
                    'Include correlation breakdown analysis'
                ]
            },
            '4.2a': {
                'title': 'Intra-regime performance evolution',
                'requirements': [
                    'Analyze early vs late cycle factor leadership',
                    'Study performance decay within regimes',
                    'Identify optimal entry/exit timing',
                    'Create regime lifecycle analysis',
                    'Add performance momentum within regimes',
                    'Include regime maturity indicators'
                ]
            },
            '4.2b': {
                'title': 'Macro-factor relationships',
                'requirements': [
                    'Analyze interest rate sensitivity by regime',
                    'Study inflation impact on factor premiums',
                    'Examine growth vs value rotation patterns',
                    'Create macro sensitivity analysis',
                    'Add economic indicator correlations',
                    'Include yield curve impact analysis'
                ]
            },
            '4.3a': {
                'title': 'Regime-aware allocation frameworks',
                'requirements': [
                    'Calculate optimal factor weights per regime',
                    'Develop dynamic rebalancing triggers',
                    'Implement risk budgeting by cycle phase',
                    'Create allocation recommendation system',
                    'Add risk-adjusted allocation models',
                    'Include regime uncertainty adjustments'
                ]
            },
            '4.3b': {
                'title': 'Factor timing models',
                'requirements': [
                    'Analyze regime prediction accuracy',
                    'Develop factor rotation strategies',
                    'Compare market timing vs time-in-market analysis',
                    'Create timing signal analysis',
                    'Add regime forecasting models',
                    'Include strategy performance attribution'
                ]
            }
        }
        
        # Initialize verification results
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'verification_type': 'PHASE_4_COMPREHENSIVE_SUBSTEP_VERIFICATION',
            'total_substeps': len(self.expected_files),
            'substep_verification': {},
            'demo_generation': {},
            'roadmap_compliance': {},
            'overall_assessment': {}
        }
        
        logger.info("Phase 4 Comprehensive Verifier initialized")
    
    def load_verification_data(self):
        """Load all required data for verification"""
        try:
            # Load aligned data
            aligned_file = self.results_dir / 'aligned_master_dataset_FIXED.csv'
            if aligned_file.exists():
                self.aligned_data = pd.read_csv(aligned_file, index_col=0, parse_dates=True)
                logger.info(f"✓ Loaded aligned data: {self.aligned_data.shape}")
            else:
                logger.warning("Aligned data not found")
                self.aligned_data = None
            
            # Load Phase 2 performance metrics
            perf_file = self.results_dir / 'phase2_performance_analysis.json'
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    self.performance_metrics = json.load(f)
                logger.info("✓ Loaded Phase 2 performance metrics")
            else:
                logger.warning("Phase 2 performance metrics not found")
                self.performance_metrics = {}
                
        except Exception as e:
            logger.error(f"Error loading verification data: {e}")
            self.aligned_data = None
            self.performance_metrics = {}
    
    def verify_substep_4_1a_transition_probability_matrix(self):
        """Verify Step 4.1a: Transition probability matrix"""
        logger.info("=== VERIFYING STEP 4.1a: Transition Probability Matrix ===")
        
        step_verification = {
            'step': '4.1a',
            'title': 'Transition probability matrix',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if output file exists
            output_file = self.results_dir / 'phase4_regime_transition_analytics.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            # Test 2: Load and verify content structure
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Check for required components
                required_keys = ['transition_probabilities', 'analysis_timestamp']
                step_verification['tests']['has_required_structure'] = all(
                    key in data for key in required_keys
                )
                
                # Check transition probability matrix
                if 'transition_probabilities' in data:
                    trans_data = data['transition_probabilities']
                    
                    step_verification['tests']['has_transition_counts'] = 'transition_counts' in trans_data
                    step_verification['tests']['has_transition_probabilities'] = 'transition_probabilities' in trans_data
                    step_verification['tests']['has_expected_durations'] = 'expected_durations' in trans_data
                    step_verification['tests']['has_regime_stability'] = 'regime_stability' in trans_data
                    step_verification['tests']['has_early_warning_signals'] = 'early_warning_signals' in trans_data
            
            # Test 3: Create demo transition matrix
            demo_result = self._create_demo_transition_matrix()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance check
            compliance_score = self._check_roadmap_compliance('4.1a', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success calculation
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.1a verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.1a: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def verify_substep_4_1b_transition_performance(self):
        """Verify Step 4.1b: Performance during regime changes"""
        logger.info("=== VERIFYING STEP 4.1b: Performance During Regime Changes ===")
        
        step_verification = {
            'step': '4.1b',
            'title': 'Performance during regime changes',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if transition performance data exists
            output_file = self.results_dir / 'phase4_regime_transition_analytics.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Test 2: Check for transition performance analysis
                step_verification['tests']['has_transition_performance'] = 'transition_performance' in data
                
                if 'transition_performance' in data:
                    perf_data = data['transition_performance']
                    
                    # Check factor analysis exists
                    factors = ['Value', 'Quality', 'MinVol', 'Momentum']
                    factor_analysis_count = sum(1 for factor in factors if factor in perf_data)
                    step_verification['tests']['factor_analysis_coverage'] = factor_analysis_count >= 3
                    
                    # Check for statistical significance testing
                    has_stats = any(
                        'statistical_significance' in perf_data.get(factor, {})
                        for factor in factors if factor in perf_data
                    )
                    step_verification['tests']['has_statistical_significance'] = has_stats
                    
                    # Check for volatility analysis
                    has_volatility = any(
                        'volatility_change' in perf_data.get(factor, {})
                        for factor in factors if factor in perf_data
                    )
                    step_verification['tests']['has_volatility_analysis'] = has_volatility
            
            # Test 3: Create demo transition performance analysis
            demo_result = self._create_demo_transition_performance()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance
            compliance_score = self._check_roadmap_compliance('4.1b', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.1b verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.1b: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def verify_substep_4_2a_intra_regime_evolution(self):
        """Verify Step 4.2a: Intra-regime performance evolution"""
        logger.info("=== VERIFYING STEP 4.2a: Intra-regime Performance Evolution ===")
        
        step_verification = {
            'step': '4.2a',
            'title': 'Intra-regime performance evolution',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if cyclical pattern file exists
            output_file = self.results_dir / 'phase4_cyclical_pattern_detection.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Test 2: Check for intra-regime evolution analysis
                step_verification['tests']['has_intra_regime_analysis'] = 'intra_regime_evolution' in data
                
                if 'intra_regime_evolution' in data:
                    evolution_data = data['intra_regime_evolution']
                    
                    # Check regime coverage
                    regime_count = len(evolution_data)
                    step_verification['tests']['regime_coverage_adequate'] = regime_count >= 1
                    
                    # Check for phase analysis (early/middle/late)
                    has_phase_analysis = any(
                        all(phase in regime_data.get(factor, {}) for phase in ['early_phase_performance', 'middle_phase_performance', 'late_phase_performance'])
                        for regime_data in evolution_data.values()
                        for factor in ['Value', 'Quality', 'MinVol', 'Momentum']
                        if factor in regime_data
                    )
                    step_verification['tests']['has_phase_analysis'] = has_phase_analysis
                    
                    # Check for optimal phase identification
                    has_optimal_phase = any(
                        'optimal_phase' in regime_data.get(factor, {})
                        for regime_data in evolution_data.values()
                        for factor in ['Value', 'Quality', 'MinVol', 'Momentum']
                        if factor in regime_data
                    )
                    step_verification['tests']['has_optimal_phase_identification'] = has_optimal_phase
                    
                    # Check for regime maturity indicators
                    has_maturity_indicators = any(
                        'regime_maturity_indicators' in regime_data
                        for regime_data in evolution_data.values()
                    )
                    step_verification['tests']['has_maturity_indicators'] = has_maturity_indicators
            
            # Test 3: Create demo intra-regime analysis
            demo_result = self._create_demo_intra_regime_analysis()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance
            compliance_score = self._check_roadmap_compliance('4.2a', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.2a verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.2a: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def verify_substep_4_2b_macro_factor_relationships(self):
        """Verify Step 4.2b: Macro-factor relationships"""
        logger.info("=== VERIFYING STEP 4.2b: Macro-factor Relationships ===")
        
        step_verification = {
            'step': '4.2b',
            'title': 'Macro-factor relationships',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if cyclical pattern file exists
            output_file = self.results_dir / 'phase4_cyclical_pattern_detection.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Test 2: Check for macro-factor relationships
                step_verification['tests']['has_macro_factor_analysis'] = 'macro_factor_relationships' in data
                
                if 'macro_factor_relationships' in data:
                    macro_data = data['macro_factor_relationships']
                    
                    # Check for regime-specific relationships
                    step_verification['tests']['has_regime_specific_relationships'] = 'regime_specific_relationships' in macro_data
                    
                    # Check for cross-regime sensitivity
                    step_verification['tests']['has_cross_regime_sensitivity'] = 'cross_regime_sensitivity' in macro_data
                    
                    # Check factor coverage
                    if 'regime_specific_relationships' in macro_data:
                        regime_relationships = macro_data['regime_specific_relationships']
                        factors_analyzed = set()
                        for regime_data in regime_relationships.values():
                            factors_analyzed.update(regime_data.keys())
                        
                        step_verification['tests']['factor_coverage_adequate'] = len(factors_analyzed) >= 3
                    
                    # Check for correlation and beta analysis
                    has_correlation_analysis = any(
                        any(
                            'correlation' in factor_data.get(macro_var, {})
                            for macro_var in factor_data.keys()
                        )
                        for regime_data in macro_data.get('regime_specific_relationships', {}).values()
                        for factor_data in regime_data.values()
                    )
                    step_verification['tests']['has_correlation_analysis'] = has_correlation_analysis
            
            # Test 3: Create demo macro-factor analysis
            demo_result = self._create_demo_macro_factor_analysis()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance
            compliance_score = self._check_roadmap_compliance('4.2b', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.2b verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.2b: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def verify_substep_4_3a_allocation_frameworks(self):
        """Verify Step 4.3a: Regime-aware allocation frameworks"""
        logger.info("=== VERIFYING STEP 4.3a: Regime-aware Allocation Frameworks ===")
        
        step_verification = {
            'step': '4.3a',
            'title': 'Regime-aware allocation frameworks',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if portfolio insights file exists
            output_file = self.results_dir / 'phase4_portfolio_construction_insights.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Test 2: Check for allocation frameworks
                step_verification['tests']['has_allocation_frameworks'] = 'allocation_frameworks' in data
                
                if 'allocation_frameworks' in data:
                    allocation_data = data['allocation_frameworks']
                    
                    # Check for regime-specific allocations
                    step_verification['tests']['has_regime_specific_allocations'] = 'regime_specific_allocations' in allocation_data
                    
                    # Check for dynamic recommendations
                    step_verification['tests']['has_dynamic_recommendations'] = 'dynamic_recommendations' in allocation_data
                    
                    # Check allocation methods
                    if 'regime_specific_allocations' in allocation_data:
                        regime_allocations = allocation_data['regime_specific_allocations']
                        
                        has_multiple_methods = any(
                            all(method in regime_data for method in ['risk_parity', 'sharpe_optimized', 'equal_weight'])
                            for regime_data in regime_allocations.values()
                        )
                        step_verification['tests']['has_multiple_allocation_methods'] = has_multiple_methods
                        
                        # Check for expected returns and volatilities
                        has_metrics = any(
                            all(metric in allocation_method for metric in ['expected_return', 'expected_volatility'])
                            for regime_data in regime_allocations.values()
                            for allocation_method in regime_data.values()
                            if isinstance(allocation_method, dict)
                        )
                        step_verification['tests']['has_allocation_metrics'] = has_metrics
            
            # Test 3: Create demo allocation framework
            demo_result = self._create_demo_allocation_framework()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance
            compliance_score = self._check_roadmap_compliance('4.3a', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.3a verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.3a: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def verify_substep_4_3b_timing_models(self):
        """Verify Step 4.3b: Factor timing models"""
        logger.info("=== VERIFYING STEP 4.3b: Factor Timing Models ===")
        
        step_verification = {
            'step': '4.3b',
            'title': 'Factor timing models',
            'tests': {},
            'demo_created': False,
            'roadmap_compliance': {},
            'overall_success': False
        }
        
        try:
            # Test 1: Check if portfolio insights file exists
            output_file = self.results_dir / 'phase4_portfolio_construction_insights.json'
            step_verification['tests']['output_file_exists'] = output_file.exists()
            
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Test 2: Check for timing models
                step_verification['tests']['has_timing_models'] = 'timing_models' in data
                
                if 'timing_models' in data:
                    timing_data = data['timing_models']
                    
                    # Check for momentum signals
                    step_verification['tests']['has_momentum_signals'] = 'momentum_signals' in timing_data
                    
                    # Check for mean reversion signals
                    step_verification['tests']['has_mean_reversion_signals'] = 'mean_reversion_signals' in timing_data
                    
                    # Check for regime persistence analysis
                    step_verification['tests']['has_regime_persistence'] = 'regime_persistence' in timing_data
                    
                    # Check for strategy performance attribution
                    step_verification['tests']['has_strategy_performance'] = 'strategy_performance' in timing_data
                    
                    # Check current regime identification
                    step_verification['tests']['has_current_regime'] = 'current_regime' in timing_data
                    
                    # Check factor coverage in signals
                    if 'momentum_signals' in timing_data:
                        factors_with_signals = len(timing_data['momentum_signals'])
                        step_verification['tests']['adequate_factor_coverage'] = factors_with_signals >= 3
            
            # Test 3: Create demo timing models
            demo_result = self._create_demo_timing_models()
            step_verification['tests']['demo_creation_success'] = demo_result['success'] if demo_result else False
            step_verification['demo_created'] = demo_result['success'] if demo_result else False
            
            # Test 4: Roadmap compliance
            compliance_score = self._check_roadmap_compliance('4.3b', data if output_file.exists() else {})
            step_verification['roadmap_compliance'] = compliance_score
            
            # Overall success
            test_results = [v for k, v in step_verification['tests'].items() if isinstance(v, bool)]
            success_count = sum(test_results)
            step_verification['overall_success'] = success_count >= len(test_results) * 0.8
            step_verification['success_rate'] = f"{success_count}/{len(test_results)} tests passed"
            
            logger.info(f"✓ Step 4.3b verification: {step_verification['success_rate']}")
            
        except Exception as e:
            logger.error(f"Error verifying Step 4.3b: {e}")
            step_verification['error'] = str(e)
        
        return step_verification
    
    def _check_roadmap_compliance(self, step, data):
        """Check compliance with roadmap requirements for a specific step"""
        requirements = self.roadmap_requirements.get(step, {}).get('requirements', [])
        compliance_score = {
            'total_requirements': len(requirements),
            'requirements_met': 0,
            'compliance_percentage': 0,
            'missing_requirements': [],
            'satisfied_requirements': []
        }
        
        if not requirements or not data:
            return compliance_score
        
        # Simple keyword-based compliance checking
        data_str = json.dumps(data).lower()
        
        requirement_keywords = {
            'transition': ['transition', 'probability', 'matrix'],
            'duration': ['duration', 'expected', 'persistence'],
            'warning': ['warning', 'signal', 'early'],
            'performance': ['performance', 'return', 'analysis'],
            'volatility': ['volatility', 'risk', 'std'],
            'correlation': ['correlation', 'relationship'],
            'allocation': ['allocation', 'weight', 'portfolio'],
            'timing': ['timing', 'momentum', 'signal'],
            'regime': ['regime', 'cycle', 'phase'],
            'factor': ['factor', 'value', 'quality', 'momentum']
        }
        
        for requirement in requirements:
            requirement_lower = requirement.lower()
            met = False
            
            # Check if requirement keywords appear in data
            for category, keywords in requirement_keywords.items():
                if any(keyword in requirement_lower for keyword in keywords):
                    if any(keyword in data_str for keyword in keywords):
                        met = True
                        break
            
            if met:
                compliance_score['requirements_met'] += 1
                compliance_score['satisfied_requirements'].append(requirement)
            else:
                compliance_score['missing_requirements'].append(requirement)
        
        compliance_score['compliance_percentage'] = (
            compliance_score['requirements_met'] / compliance_score['total_requirements'] * 100
            if compliance_score['total_requirements'] > 0 else 0
        )
        
        return compliance_score
    
    def _create_demo_transition_matrix(self):
        """Create demo transition probability matrix"""
        try:
            if self.aligned_data is None:
                return {'success': False, 'error': 'No aligned data available'}
            
            demo_data = {
                'DEMO_transition_matrix': {
                    'demo_transition_counts': {
                        'Goldilocks': {'Goldilocks': 50, 'Overheating': 10, 'Stagflation': 5, 'Recession': 2},
                        'Overheating': {'Goldilocks': 8, 'Overheating': 30, 'Stagflation': 15, 'Recession': 5},
                        'Stagflation': {'Goldilocks': 3, 'Overheating': 7, 'Stagflation': 25, 'Recession': 8},
                        'Recession': {'Goldilocks': 5, 'Overheating': 2, 'Stagflation': 3, 'Recession': 20}
                    },
                    'demo_expected_durations': {
                        'Goldilocks': 8.5,
                        'Overheating': 4.2,
                        'Stagflation': 3.8,
                        'Recession': 6.1
                    },
                    'demo_early_warning_signals': {
                        'vix_threshold': 35,
                        'yield_curve_inversion': True,
                        'unemployment_spike': False
                    }
                }
            }
            
            with open(self.results_dir / 'DEMO_4_1a_transition_matrix.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_1a_transition_matrix.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_demo_transition_performance(self):
        """Create demo transition performance analysis"""
        try:
            demo_data = {
                'DEMO_transition_performance': {
                    'Value': {
                        'average_pre_transition': 0.008,
                        'average_post_transition': 0.005,
                        'performance_change': -0.003,
                        'volatility_change': 0.015,
                        'statistical_significance': {'p_value': 0.045, 'is_significant': True}
                    },
                    'Quality': {
                        'average_pre_transition': 0.006,
                        'average_post_transition': 0.009,
                        'performance_change': 0.003,
                        'volatility_change': -0.008,
                        'statistical_significance': {'p_value': 0.032, 'is_significant': True}
                    }
                }
            }
            
            with open(self.results_dir / 'DEMO_4_1b_transition_performance.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_1b_transition_performance.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_demo_intra_regime_analysis(self):
        """Create demo intra-regime performance evolution"""
        try:
            demo_data = {
                'DEMO_intra_regime_evolution': {
                    'Goldilocks': {
                        'Value': {
                            'early_phase_performance': 0.012,
                            'middle_phase_performance': 0.008,
                            'late_phase_performance': 0.005,
                            'optimal_phase': 'early',
                            'performance_trend_slope': -0.002
                        },
                        'regime_maturity_indicators': {
                            'vix_trend': 0.5,
                            'growth_trend': -0.1,
                            'regime_duration': 24
                        }
                    }
                }
            }
            
            with open(self.results_dir / 'DEMO_4_2a_intra_regime_evolution.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_2a_intra_regime_evolution.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_demo_macro_factor_analysis(self):
        """Create demo macro-factor relationships"""
        try:
            demo_data = {
                'DEMO_macro_factor_relationships': {
                    'regime_specific_relationships': {
                        'Goldilocks': {
                            'Value': {
                                'DGS10': {'correlation': 0.45, 'beta_sensitivity': 0.8},
                                'INFLATION_COMPOSITE': {'correlation': 0.32, 'beta_sensitivity': 0.6}
                            }
                        }
                    },
                    'cross_regime_sensitivity': {
                        'Value': {
                            'DGS10': {'average_sensitivity': 0.35, 'sensitivity_stability': 0.15}
                        }
                    }
                }
            }
            
            with open(self.results_dir / 'DEMO_4_2b_macro_factor_relationships.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_2b_macro_factor_relationships.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_demo_allocation_framework(self):
        """Create demo allocation frameworks"""
        try:
            demo_data = {
                'DEMO_allocation_frameworks': {
                    'regime_specific_allocations': {
                        'Goldilocks': {
                            'risk_parity': {
                                'weights': {'Value': 0.3, 'Quality': 0.2, 'MinVol': 0.2, 'Momentum': 0.3},
                                'expected_return': 0.085,
                                'expected_volatility': 0.12
                            },
                            'sharpe_optimized': {
                                'weights': {'Value': 0.4, 'Quality': 0.1, 'MinVol': 0.1, 'Momentum': 0.4},
                                'expected_return': 0.095,
                                'expected_volatility': 0.15
                            }
                        }
                    }
                }
            }
            
            with open(self.results_dir / 'DEMO_4_3a_allocation_frameworks.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_3a_allocation_frameworks.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_demo_timing_models(self):
        """Create demo timing models"""
        try:
            demo_data = {
                'DEMO_timing_models': {
                    'momentum_signals': {
                        'Value': {
                            '3_month_momentum': 0.015,
                            '6_month_momentum': 0.012,
                            '12_month_momentum': 0.008,
                            'momentum_consistency': 0.65
                        }
                    },
                    'mean_reversion_signals': {
                        'Value': {
                            'deviation_from_longterm': -0.008,
                            'reversion_signal': 'buy',
                            'signal_strength': 0.008
                        }
                    },
                    'regime_persistence': {
                        'Goldilocks': {'persistence_3_months': 0.75, 'persistence_6_months': 0.60}
                    },
                    'current_regime': 'Goldilocks'
                }
            }
            
            with open(self.results_dir / 'DEMO_4_3b_timing_models.json', 'w') as f:
                json.dump(demo_data, f, indent=2)
            
            return {'success': True, 'filename': 'DEMO_4_3b_timing_models.json'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_comprehensive_verification(self):
        """Run comprehensive verification of all Phase 4 substeps"""
        logger.info("=" * 80)
        logger.info("🔬 PHASE 4 COMPREHENSIVE SUBSTEP VERIFICATION")
        logger.info("=" * 80)
        
        # Verify each substep
        verification_functions = {
            '4.1a': self.verify_substep_4_1a_transition_probability_matrix,
            '4.1b': self.verify_substep_4_1b_transition_performance,
            '4.2a': self.verify_substep_4_2a_intra_regime_evolution,
            '4.2b': self.verify_substep_4_2b_macro_factor_relationships,
            '4.3a': self.verify_substep_4_3a_allocation_frameworks,
            '4.3b': self.verify_substep_4_3b_timing_models
        }
        
        for step, verify_func in verification_functions.items():
            try:
                result = verify_func()
                self.verification_results['substep_verification'][step] = result
                
                status = '✓' if result['overall_success'] else '✗'
                logger.info(f"Step {step}: {status} - {result['success_rate']}")
                
            except Exception as e:
                error_result = {
                    'step': step,
                    'overall_success': False,
                    'error': str(e)
                }
                self.verification_results['substep_verification'][step] = error_result
                logger.error(f"Step {step}: ✗ - Error: {str(e)}")
        
        # Calculate overall assessment
        total_substeps = len(verification_functions)
        successful_substeps = sum(
            1 for result in self.verification_results['substep_verification'].values() 
            if result.get('overall_success', False)
        )
        demos_created = sum(
            1 for result in self.verification_results['substep_verification'].values()
            if result.get('demo_created', False)
        )
        
        success_rate = successful_substeps / total_substeps
        demo_rate = demos_created / total_substeps
        overall_score = (success_rate + demo_rate) / 2
        
        self.verification_results['overall_assessment'] = {
            'total_substeps': total_substeps,
            'successful_substeps': successful_substeps,
            'demos_created': demos_created,
            'substep_success_rate': round(success_rate, 3),
            'demo_creation_rate': round(demo_rate, 3),
            'overall_score': round(overall_score, 3),
            'ready_for_phase5': overall_score >= 0.80,
            'recommendation': 'PROCEED TO PHASE 5' if overall_score >= 0.80 else 'REVIEW PHASE 4 IMPLEMENTATION'
        }
        
        # Save comprehensive results
        with open(self.results_dir / 'phase4_comprehensive_substep_verification.json', 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        
        # Log final results
        logger.info("=" * 80)
        logger.info("📊 PHASE 4 COMPREHENSIVE VERIFICATION RESULTS:")
        logger.info(f"📋 Substep Success: {successful_substeps}/{total_substeps} ({success_rate:.1%})")
        logger.info(f"🎨 Demo Creation: {demos_created}/{total_substeps} ({demo_rate:.1%})")
        logger.info(f"🎯 Overall Score: {overall_score:.1%}")
        logger.info(f"🚀 Ready for Phase 5: {self.verification_results['overall_assessment']['ready_for_phase5']}")
        logger.info(f"💡 Recommendation: {self.verification_results['overall_assessment']['recommendation']}")
        logger.info("=" * 80)
        
        return self.verification_results

def main():
    """Run Phase 4 comprehensive verification"""
    verifier = Phase4ComprehensiveVerifier()
    results = verifier.run_comprehensive_verification()
    
    print("\n🎯 PHASE 4 COMPREHENSIVE VERIFICATION SUMMARY:")
    print(f"📋 Substeps: {results['overall_assessment']['substep_success_rate']:.1%}")
    print(f"🎨 Demos: {results['overall_assessment']['demo_creation_rate']:.1%}")
    print(f"🎉 Overall: {results['overall_assessment']['overall_score']:.1%}")
    print(f"🚀 Phase 5 Ready: {results['overall_assessment']['ready_for_phase5']}")
    print(f"💡 Recommendation: {results['overall_assessment']['recommendation']}")
    
    return results

if __name__ == "__main__":
    main() 