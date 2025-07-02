"""
Phase 3 Comprehensive Testing Suite
Validates all Phase 3 visualization components against roadmap specifications
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import plotly.io as pio
from bs4 import BeautifulSoup
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3TestingSuite:
    """
    Comprehensive testing suite for Phase 3 visualizations
    """
    
    def __init__(self, results_dir="results/business_cycle_analysis"):
        self.results_dir = Path(results_dir)
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
        
        logger.info("Phase 3 Comprehensive Testing Suite initialized")
    
    def test_step_3_1_master_dashboard_layout(self):
        """
        Test Step 3.1: Master Business Cycle Dashboard Layout
        """
        logger.info("=== TESTING STEP 3.1: Master Business Cycle Dashboard Layout ===")
        
        step_tests = {}
        
        # Test 3.1a: Interactive timeline with regime overlay
        logger.info("Testing 3.1a: Interactive timeline with regime overlay...")
        timeline_tests = self._test_interactive_timeline()
        step_tests['3.1a'] = timeline_tests
        
        # Test 3.1b: Dynamic regime statistics panel
        logger.info("Testing 3.1b: Dynamic regime statistics panel...")
        stats_tests = self._test_regime_statistics_panel()
        step_tests['3.1b'] = stats_tests
        
        self.test_results['step_3_1'] = step_tests
        
        # Summary
        total_tests = sum(len(tests['individual_tests']) for tests in step_tests.values())
        passed_tests = sum(sum(test['passed'] for test in tests['individual_tests']) for tests in step_tests.values())
        
        logger.info(f"Step 3.1 Results: {passed_tests}/{total_tests} tests passed")
        return step_tests
    
    def _test_interactive_timeline(self):
        """
        Test interactive timeline with regime overlay
        """
        tests = {
            'file_exists': False,
            'color_coded_bands': False,
            'regime_transitions': False,
            'hover_details': False,
            'sp500_performance': False,
            'factor_performance': False,
            'vix_subplot': False,
            'threshold_markers': False,
            'individual_tests': []
        }
        
        timeline_file = self.results_dir / 'interactive_timeline_regime_overlay.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Timeline file exists', 'passed': timeline_file.exists()}
        tests['individual_tests'].append(test_1)
        tests['file_exists'] = test_1['passed']
        
        if timeline_file.exists():
            try:
                # Read and parse HTML content
                with open(timeline_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains regime color specifications
                required_colors = ['#2E8B57', '#FF6347', '#FFD700', '#8B0000']  # Goldilocks, Overheating, Stagflation, Recession
                color_test = all(color in content for color in required_colors)
                test_2 = {'name': 'Color-coded regime bands present', 'passed': color_test}
                tests['individual_tests'].append(test_2)
                tests['color_coded_bands'] = test_2['passed']
                
                # Test 3: Contains regime transition logic
                transition_test = 'regime_periods' in content or 'ECONOMIC_REGIME' in content
                test_3 = {'name': 'Regime transition indicators present', 'passed': transition_test}
                tests['individual_tests'].append(test_3)
                tests['regime_transitions'] = test_3['passed']
                
                # Test 4: Contains hover template specifications
                hover_test = 'hovertemplate' in content and 'Date:' in content
                test_4 = {'name': 'Interactive hover details present', 'passed': hover_test}
                tests['individual_tests'].append(test_4)
                tests['hover_details'] = test_4['passed']
                
                # Test 5: Contains S&P 500 performance
                sp500_test = 'SP500' in content or 'S&P 500' in content
                test_5 = {'name': 'S&P 500 performance line present', 'passed': sp500_test}
                tests['individual_tests'].append(test_5)
                tests['sp500_performance'] = test_5['passed']
                
                # Test 6: Contains factor performance lines
                factors = ['Value', 'Quality', 'MinVol', 'Momentum']
                factor_test = all(factor in content for factor in factors)
                test_6 = {'name': 'All factor performance lines present', 'passed': factor_test}
                tests['individual_tests'].append(test_6)
                tests['factor_performance'] = test_6['passed']
                
                # Test 7: Contains VIX subplot
                vix_test = 'VIX' in content and 'row=2' in content
                test_7 = {'name': 'VIX stress level subplot present', 'passed': vix_test}
                tests['individual_tests'].append(test_7)
                tests['vix_subplot'] = test_7['passed']
                
                # Test 8: Contains VIX threshold markers
                thresholds = ['25', '35', '50']
                threshold_test = all(threshold in content for threshold in thresholds)
                test_8 = {'name': 'VIX threshold markers present', 'passed': threshold_test}
                tests['individual_tests'].append(test_8)
                tests['threshold_markers'] = test_8['passed']
                
            except Exception as e:
                logger.error(f"Error testing timeline file: {e}")
                for i in range(2, 9):
                    tests['individual_tests'].append({'name': f'Timeline test {i}', 'passed': False})
        
        return tests
    
    def _test_regime_statistics_panel(self):
        """
        Test regime statistics panel
        """
        tests = {
            'file_exists': False,
            'regime_duration_stats': False,
            'factor_performance': False,
            'vix_statistics': False,
            'percentage_calculations': False,
            'individual_tests': []
        }
        
        stats_file = self.results_dir / 'regime_statistics_panel.json'
        
        # Test 1: File exists
        test_1 = {'name': 'Regime statistics file exists', 'passed': stats_file.exists()}
        tests['individual_tests'].append(test_1)
        tests['file_exists'] = test_1['passed']
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                
                # Test 2: Contains regime duration statistics
                regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
                duration_test = all(regime in stats_data for regime in regimes)
                test_2 = {'name': 'All regime duration statistics present', 'passed': duration_test}
                tests['individual_tests'].append(test_2)
                tests['regime_duration_stats'] = test_2['passed']
                
                # Test 3: Contains factor performance data
                if duration_test and regimes[0] in stats_data:
                    factors = ['Value', 'Quality', 'MinVol', 'Momentum']
                    factor_test = 'factor_performance' in stats_data[regimes[0]]
                    if factor_test:
                        factor_data = stats_data[regimes[0]]['factor_performance']
                        factor_test = all(factor in factor_data for factor in factors)
                    test_3 = {'name': 'Factor performance data present', 'passed': factor_test}
                    tests['individual_tests'].append(test_3)
                    tests['factor_performance'] = test_3['passed']
                
                # Test 4: Contains VIX statistics
                if duration_test and regimes[0] in stats_data:
                    vix_test = 'vix_statistics' in stats_data[regimes[0]]
                    test_4 = {'name': 'VIX statistics present', 'passed': vix_test}
                    tests['individual_tests'].append(test_4)
                    tests['vix_statistics'] = test_4['passed']
                
                # Test 5: Contains percentage calculations
                if duration_test and regimes[0] in stats_data:
                    pct_test = 'percentage_of_period' in stats_data[regimes[0]]
                    test_5 = {'name': 'Percentage calculations present', 'passed': pct_test}
                    tests['individual_tests'].append(test_5)
                    tests['percentage_calculations'] = test_5['passed']
                
            except Exception as e:
                logger.error(f"Error testing stats file: {e}")
                for i in range(2, 6):
                    tests['individual_tests'].append({'name': f'Stats test {i}', 'passed': False})
        
        return tests
    
    def test_step_3_2_multilayer_heatmaps(self):
        """
        Test Step 3.2: Multi-Layer Performance Heatmaps
        """
        logger.info("=== TESTING STEP 3.2: Multi-Layer Performance Heatmaps ===")
        
        step_tests = {}
        
        # Test 3.2a: Primary performance heatmap
        logger.info("Testing 3.2a: Primary performance heatmap...")
        primary_tests = self._test_primary_heatmap()
        step_tests['3.2a'] = primary_tests
        
        # Test 3.2b: Risk-adjusted performance heatmap
        logger.info("Testing 3.2b: Risk-adjusted performance heatmap...")
        risk_adjusted_tests = self._test_risk_adjusted_heatmap()
        step_tests['3.2b'] = risk_adjusted_tests
        
        # Test 3.2c: Relative performance heatmap
        logger.info("Testing 3.2c: Relative performance heatmap...")
        relative_tests = self._test_relative_performance_heatmap()
        step_tests['3.2c'] = relative_tests
        
        self.test_results['step_3_2'] = step_tests
        
        # Summary
        total_tests = sum(len(tests['individual_tests']) for tests in step_tests.values())
        passed_tests = sum(sum(test['passed'] for test in tests['individual_tests']) for tests in step_tests.values())
        
        logger.info(f"Step 3.2 Results: {passed_tests}/{total_tests} tests passed")
        return step_tests
    
    def _test_primary_heatmap(self):
        """
        Test primary performance heatmap
        """
        tests = {'individual_tests': []}
        
        heatmap_file = self.results_dir / 'primary_performance_heatmap.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Primary heatmap file exists', 'passed': heatmap_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if heatmap_file.exists():
            try:
                with open(heatmap_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains required factors
                factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'S&P 500']
                factor_test = all(factor in content for factor in factors)
                test_2 = {'name': 'All factors present in heatmap', 'passed': factor_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains required regimes
                regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
                regime_test = all(regime in content for regime in regimes)
                test_3 = {'name': 'All regimes present in heatmap', 'passed': regime_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains color scale
                color_test = 'RdYlGn' in content or 'colorscale' in content
                test_4 = {'name': 'Color coding present', 'passed': color_test}
                tests['individual_tests'].append(test_4)
                
                # Test 5: Contains annotations
                annotation_test = 'add_annotation' in content or 'text=' in content
                test_5 = {'name': 'Data labels/annotations present', 'passed': annotation_test}
                tests['individual_tests'].append(test_5)
                
                # Test 6: Contains hover templates
                hover_test = 'hovertemplate' in content
                test_6 = {'name': 'Hover tooltips present', 'passed': hover_test}
                tests['individual_tests'].append(test_6)
                
            except Exception as e:
                logger.error(f"Error testing primary heatmap: {e}")
                for i in range(2, 7):
                    tests['individual_tests'].append({'name': f'Primary heatmap test {i}', 'passed': False})
        
        return tests
    
    def _test_risk_adjusted_heatmap(self):
        """
        Test risk-adjusted performance heatmap
        """
        tests = {'individual_tests': []}
        
        heatmap_file = self.results_dir / 'risk_adjusted_heatmap.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Risk-adjusted heatmap file exists', 'passed': heatmap_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if heatmap_file.exists():
            try:
                with open(heatmap_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains Sharpe ratio references
                sharpe_test = 'Sharpe' in content
                test_2 = {'name': 'Sharpe ratio metrics present', 'passed': sharpe_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains risk metrics in hover
                risk_hover_test = 'Sortino' in content or 'Max Drawdown' in content
                test_3 = {'name': 'Risk metrics in hover details', 'passed': risk_hover_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains proper color scale
                color_test = 'zmid=0' in content or 'zmid:0' in content
                test_4 = {'name': 'Appropriate color scale for ratios', 'passed': color_test}
                tests['individual_tests'].append(test_4)
                
            except Exception as e:
                logger.error(f"Error testing risk-adjusted heatmap: {e}")
                for i in range(2, 5):
                    tests['individual_tests'].append({'name': f'Risk-adjusted test {i}', 'passed': False})
        
        return tests
    
    def _test_relative_performance_heatmap(self):
        """
        Test relative performance heatmap
        """
        tests = {'individual_tests': []}
        
        heatmap_file = self.results_dir / 'relative_performance_heatmap.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Relative performance heatmap file exists', 'passed': heatmap_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if heatmap_file.exists():
            try:
                with open(heatmap_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains excess return calculations
                excess_test = 'Excess Return' in content
                test_2 = {'name': 'Excess return calculations present', 'passed': excess_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains S&P 500 comparison
                sp500_test = 'vs S&P 500' in content
                test_3 = {'name': 'S&P 500 comparison present', 'passed': sp500_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains outperformance indicators
                outperform_test = 'Factor Return' in content and 'S&P 500 Return' in content
                test_4 = {'name': 'Outperformance indicators present', 'passed': outperform_test}
                tests['individual_tests'].append(test_4)
                
            except Exception as e:
                logger.error(f"Error testing relative heatmap: {e}")
                for i in range(2, 5):
                    tests['individual_tests'].append({'name': f'Relative heatmap test {i}', 'passed': False})
        
        return tests
    
    def test_step_3_3_advanced_charts(self):
        """
        Test Step 3.3: Advanced Analytical Charts
        """
        logger.info("=== TESTING STEP 3.3: Advanced Analytical Charts ===")
        
        step_tests = {}
        
        # Test 3.3a: Factor rotation wheel
        logger.info("Testing 3.3a: Factor rotation wheel...")
        rotation_tests = self._test_factor_rotation_wheel()
        step_tests['3.3a'] = rotation_tests
        
        # Test 3.3b: Risk-return scatter plots
        logger.info("Testing 3.3b: Risk-return scatter plots...")
        scatter_tests = self._test_risk_return_scatter()
        step_tests['3.3b'] = scatter_tests
        
        # Test 3.3c: Rolling regime analysis
        logger.info("Testing 3.3c: Rolling regime analysis...")
        rolling_tests = self._test_rolling_regime_analysis()
        step_tests['3.3c'] = rolling_tests
        
        self.test_results['step_3_3'] = step_tests
        
        # Summary
        total_tests = sum(len(tests['individual_tests']) for tests in step_tests.values())
        passed_tests = sum(sum(test['passed'] for test in tests['individual_tests']) for tests in step_tests.values())
        
        logger.info(f"Step 3.3 Results: {passed_tests}/{total_tests} tests passed")
        return step_tests
    
    def _test_factor_rotation_wheel(self):
        """
        Test factor rotation wheel
        """
        tests = {'individual_tests': []}
        
        wheel_file = self.results_dir / 'factor_rotation_wheel.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Factor rotation wheel file exists', 'passed': wheel_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if wheel_file.exists():
            try:
                with open(wheel_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains polar/radar chart
                polar_test = 'Scatterpolar' in content or 'polar' in content
                test_2 = {'name': 'Polar chart structure present', 'passed': polar_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains regime subplots
                subplot_test = 'make_subplots' in content and 'polar' in content
                test_3 = {'name': 'Regime subplot structure present', 'passed': subplot_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains factor leadership
                factors = ['Value', 'Quality', 'MinVol', 'Momentum']
                factor_test = all(factor in content for factor in factors)
                test_4 = {'name': 'Factor leadership indicators present', 'passed': factor_test}
                tests['individual_tests'].append(test_4)
                
            except Exception as e:
                logger.error(f"Error testing rotation wheel: {e}")
                for i in range(2, 5):
                    tests['individual_tests'].append({'name': f'Rotation wheel test {i}', 'passed': False})
        
        return tests
    
    def _test_risk_return_scatter(self):
        """
        Test risk-return scatter plots
        """
        tests = {'individual_tests': []}
        
        scatter_file = self.results_dir / 'risk_return_scatter_plots.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Risk-return scatter file exists', 'passed': scatter_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if scatter_file.exists():
            try:
                with open(scatter_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains regime clustering
                regime_colors = ['#2E8B57', '#FF6347', '#FFD700', '#8B0000']
                cluster_test = any(color in content for color in regime_colors)
                test_2 = {'name': 'Regime clustering colors present', 'passed': cluster_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains efficient frontier
                frontier_test = 'Sharpe Ratio = 1' in content or 'efficient frontier' in content
                test_3 = {'name': 'Efficient frontier reference present', 'passed': frontier_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains volatility and return axes
                axes_test = 'Volatility' in content and 'Return' in content
                test_4 = {'name': 'Risk-return axes properly labeled', 'passed': axes_test}
                tests['individual_tests'].append(test_4)
                
            except Exception as e:
                logger.error(f"Error testing scatter plots: {e}")
                for i in range(2, 5):
                    tests['individual_tests'].append({'name': f'Scatter plot test {i}', 'passed': False})
        
        return tests
    
    def _test_rolling_regime_analysis(self):
        """
        Test rolling regime analysis
        """
        tests = {'individual_tests': []}
        
        rolling_file = self.results_dir / 'rolling_regime_analysis.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Rolling analysis file exists', 'passed': rolling_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if rolling_file.exists():
            try:
                with open(rolling_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains 12-month rolling windows
                rolling_test = '12' in content and ('rolling' in content.lower() or 'window' in content.lower())
                test_2 = {'name': '12-month rolling windows present', 'passed': rolling_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains regime transition markers
                transition_test = 'regime' in content.lower() and 'transition' in content.lower()
                test_3 = {'name': 'Regime transition markers present', 'passed': transition_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains multiple subplots
                subplot_test = 'rows=3' in content or 'subplot' in content
                test_4 = {'name': 'Multiple subplot structure present', 'passed': subplot_test}
                tests['individual_tests'].append(test_4)
                
                # Test 5: Contains rolling volatility
                vol_test = 'volatility' in content.lower() or 'vol' in content
                test_5 = {'name': 'Rolling volatility analysis present', 'passed': vol_test}
                tests['individual_tests'].append(test_5)
                
            except Exception as e:
                logger.error(f"Error testing rolling analysis: {e}")
                for i in range(2, 6):
                    tests['individual_tests'].append({'name': f'Rolling analysis test {i}', 'passed': False})
        
        return tests
    
    def test_step_3_4_correlation_analysis(self):
        """
        Test Step 3.4: Correlation & Dependency Analysis
        """
        logger.info("=== TESTING STEP 3.4: Correlation & Dependency Analysis ===")
        
        step_tests = {}
        
        # Test 3.4a: Dynamic correlation matrices
        logger.info("Testing 3.4a: Dynamic correlation matrices...")
        correlation_tests = self._test_correlation_matrices()
        step_tests['3.4a'] = correlation_tests
        
        # Test 3.4b: Factor momentum persistence
        logger.info("Testing 3.4b: Factor momentum persistence...")
        momentum_tests = self._test_momentum_persistence()
        step_tests['3.4b'] = momentum_tests
        
        self.test_results['step_3_4'] = step_tests
        
        # Summary
        total_tests = sum(len(tests['individual_tests']) for tests in step_tests.values())
        passed_tests = sum(sum(test['passed'] for test in tests['individual_tests']) for tests in step_tests.values())
        
        logger.info(f"Step 3.4 Results: {passed_tests}/{total_tests} tests passed")
        return step_tests
    
    def _test_correlation_matrices(self):
        """
        Test correlation matrices
        """
        tests = {'individual_tests': []}
        
        corr_file = self.results_dir / 'correlation_matrices_by_regime.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Correlation matrices file exists', 'passed': corr_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if corr_file.exists():
            try:
                with open(corr_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains correlation heatmaps
                heatmap_test = 'Heatmap' in content
                test_2 = {'name': 'Correlation heatmap structure present', 'passed': heatmap_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains regime-specific matrices
                regime_test = all(regime in content for regime in ['Goldilocks', 'Overheating', 'Stagflation', 'Recession'])
                test_3 = {'name': 'All regime-specific matrices present', 'passed': regime_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains correlation color scale
                color_test = 'RdBu' in content or 'colorscale' in content
                test_4 = {'name': 'Correlation color scale present', 'passed': color_test}
                tests['individual_tests'].append(test_4)
                
                # Test 5: Contains factor correlation values
                factors = ['Value', 'Quality', 'MinVol', 'Momentum']
                factor_test = all(factor in content for factor in factors)
                test_5 = {'name': 'All factors in correlation matrix', 'passed': factor_test}
                tests['individual_tests'].append(test_5)
                
            except Exception as e:
                logger.error(f"Error testing correlation matrices: {e}")
                for i in range(2, 6):
                    tests['individual_tests'].append({'name': f'Correlation test {i}', 'passed': False})
        
        return tests
    
    def _test_momentum_persistence(self):
        """
        Test momentum persistence analysis
        """
        tests = {'individual_tests': []}
        
        momentum_file = self.results_dir / 'momentum_persistence_analysis.html'
        
        # Test 1: File exists
        test_1 = {'name': 'Momentum persistence file exists', 'passed': momentum_file.exists()}
        tests['individual_tests'].append(test_1)
        
        if momentum_file.exists():
            try:
                with open(momentum_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test 2: Contains autocorrelation analysis
                autocorr_test = 'autocorr' in content.lower() or 'Autocorrelation' in content
                test_2 = {'name': 'Autocorrelation analysis present', 'passed': autocorr_test}
                tests['individual_tests'].append(test_2)
                
                # Test 3: Contains lag analysis
                lag_test = 'lag' in content.lower() and 'months' in content.lower()
                test_3 = {'name': 'Lag period analysis present', 'passed': lag_test}
                tests['individual_tests'].append(test_3)
                
                # Test 4: Contains significance bounds
                significance_test = 'significance' in content.lower() or '1.96' in content
                test_4 = {'name': 'Statistical significance bounds present', 'passed': significance_test}
                tests['individual_tests'].append(test_4)
                
                # Test 5: Contains momentum decay analysis
                decay_test = 'momentum' in content.lower() and 'persistence' in content.lower()
                test_5 = {'name': 'Momentum persistence analysis present', 'passed': decay_test}
                tests['individual_tests'].append(test_5)
                
            except Exception as e:
                logger.error(f"Error testing momentum persistence: {e}")
                for i in range(2, 6):
                    tests['individual_tests'].append({'name': f'Momentum test {i}', 'passed': False})
        
        return tests
    
    def run_comprehensive_test_suite(self):
        """
        Run complete comprehensive test suite
        """
        logger.info("=" * 80)
        logger.info("RUNNING PHASE 3 COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Test each step
        step_3_1_results = self.test_step_3_1_master_dashboard_layout()
        step_3_2_results = self.test_step_3_2_multilayer_heatmaps()
        step_3_3_results = self.test_step_3_3_advanced_charts()
        step_3_4_results = self.test_step_3_4_correlation_analysis()
        
        # Calculate overall results
        all_tests = []
        for step_results in [step_3_1_results, step_3_2_results, step_3_3_results, step_3_4_results]:
            for substep_key, substep_tests in step_results.items():
                all_tests.extend(substep_tests['individual_tests'])
        
        total_tests = len(all_tests)
        passed_tests = sum(test['passed'] for test in all_tests)
        failed_tests = [test for test in all_tests if not test['passed']]
        
        # Summary report
        logger.info("=" * 80)
        logger.info("PHASE 3 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Tests Run: {total_tests}")
        logger.info(f"Tests Passed: {passed_tests}")
        logger.info(f"Tests Failed: {len(failed_tests)}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests:
            logger.info("\nFAILED TESTS:")
            for test in failed_tests:
                logger.info(f"  ❌ {test['name']}")
        else:
            logger.info("\n🎉 ALL TESTS PASSED!")
        
        # Create detailed test report
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': len(failed_tests),
            'success_rate': (passed_tests/total_tests)*100,
            'detailed_results': self.test_results,
            'failed_test_details': failed_tests
        }
        
        # Save test report
        with open(self.results_dir / 'phase3_comprehensive_test_report.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        logger.info(f"\n📄 Detailed test report saved: phase3_comprehensive_test_report.json")
        
        return test_report

def main():
    """
    Run Phase 3 comprehensive testing suite
    """
    tester = Phase3TestingSuite()
    test_report = tester.run_comprehensive_test_suite()
    
    if test_report['success_rate'] >= 95:
        logger.info("✅ Phase 3 testing PASSED - Ready to proceed to Phase 4")
        return True
    else:
        logger.error("❌ Phase 3 testing FAILED - Address issues before proceeding")
        return False

if __name__ == "__main__":
    main() 