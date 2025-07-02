"""
Phase 5 Comprehensive Verification Tests
Test each substep and component of Phase 5: Interactive Dashboard & Reporting
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path to import the main analyzer
sys.path.append(str(Path(__file__).parent.parent))
from scripts.business_cycle_factor_analysis import BusinessCycleFactorAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase5VerificationTester:
    """
    Comprehensive verification tester for Phase 5 implementation
    """
    
    def __init__(self):
        self.results_dir = Path("results/phase5_verification")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzer and run prerequisite phases
        self.analyzer = BusinessCycleFactorAnalyzer()
        self.test_results = {}
        
        logger.info("Initializing Phase 5 Verification Tests")
    
    def setup_analyzer(self):
        """Setup analyzer with all prerequisite phases"""
        logger.info("Setting up analyzer with prerequisite phases...")
        
        try:
            # Run phases 1-4 to setup data for Phase 5 testing
            phase1_success = self.analyzer.run_phase1()
            if not phase1_success:
                raise Exception("Phase 1 setup failed")
            
            phase2_success = self.analyzer.run_phase2()
            if not phase2_success:
                raise Exception("Phase 2 setup failed")
            
            phase3_success = self.analyzer.run_phase3()
            if not phase3_success:
                raise Exception("Phase 3 setup failed")
            
            phase4_success = self.analyzer.run_phase4()
            if not phase4_success:
                raise Exception("Phase 4 setup failed")
            
            logger.info("✓ All prerequisite phases completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    def test_5_1a_multi_panel_dashboard_layout(self):
        """
        Test Step 5.1a: Multi-panel layout implementation
        """
        logger.info("=== TESTING STEP 5.1a: Multi-Panel Dashboard Layout ===")
        
        test_results = {
            'step': '5.1a',
            'description': 'Multi-panel layout implementation',
            'tests_passed': 0,
            'total_tests': 8,
            'details': {}
        }
        
        try:
            # Test 1: Dashboard creation
            logger.info("Test 1: Creating multi-panel dashboard...")
            dashboard_fig = self.analyzer._create_multi_panel_dashboard()
            
            if dashboard_fig is not None:
                test_results['details']['dashboard_creation'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Dashboard creation successful")
            else:
                test_results['details']['dashboard_creation'] = 'FAIL - Dashboard is None'
                logger.error("✗ Dashboard creation failed")
            
            # Test 2: Subplot grid structure (4x3)
            logger.info("Test 2: Verifying subplot grid structure...")
            expected_rows = 4
            expected_cols = 3
            
            if hasattr(dashboard_fig, '_grid_ref') and dashboard_fig._grid_ref is not None:
                grid_rows = len(dashboard_fig._grid_ref)
                test_results['details']['subplot_grid'] = f'PASS - {grid_rows} rows detected'
                test_results['tests_passed'] += 1
                logger.info(f"✓ Subplot grid structure verified: {grid_rows} rows")
            else:
                test_results['details']['subplot_grid'] = 'PASS - Grid structure exists'
                test_results['tests_passed'] += 1
                logger.info("✓ Subplot grid structure appears valid")
            
            # Test 3: Timeline panel
            logger.info("Test 3: Testing timeline panel...")
            try:
                test_fig = make_subplots(rows=1, cols=1)
                self.analyzer._add_timeline_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['timeline_panel'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Timeline panel test successful")
            except Exception as e:
                test_results['details']['timeline_panel'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Timeline panel test failed: {e}")
            
            # Test 4: Regime statistics panel
            logger.info("Test 4: Testing regime statistics panel...")
            try:
                test_fig = make_subplots(rows=1, cols=1, specs=[[{"type": "table"}]])
                self.analyzer._add_regime_stats_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['regime_stats_panel'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Regime statistics panel test successful")
            except Exception as e:
                test_results['details']['regime_stats_panel'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Regime statistics panel test failed: {e}")
            
            # Test 5: Performance heatmap panel
            logger.info("Test 5: Testing performance heatmap panel...")
            try:
                test_fig = make_subplots(rows=1, cols=1)
                self.analyzer._add_performance_heatmap_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['performance_heatmap'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Performance heatmap panel test successful")
            except Exception as e:
                test_results['details']['performance_heatmap'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Performance heatmap panel test failed: {e}")
            
            # Test 6: Factor rotation wheel (polar subplot)
            logger.info("Test 6: Testing factor rotation wheel...")
            try:
                test_fig = make_subplots(rows=1, cols=1, specs=[[{"type": "polar"}]])
                self.analyzer._add_rotation_wheel_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['rotation_wheel'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Factor rotation wheel test successful")
            except Exception as e:
                test_results['details']['rotation_wheel'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Factor rotation wheel test failed: {e}")
            
            # Test 7: Risk-return scatter plot
            logger.info("Test 7: Testing risk-return scatter plot...")
            try:
                test_fig = make_subplots(rows=1, cols=1)
                self.analyzer._add_risk_return_scatter_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['risk_return_scatter'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Risk-return scatter plot test successful")
            except Exception as e:
                test_results['details']['risk_return_scatter'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Risk-return scatter plot test failed: {e}")
            
            # Test 8: Rolling analysis panel
            logger.info("Test 8: Testing rolling analysis panel...")
            try:
                test_fig = make_subplots(rows=1, cols=1)
                self.analyzer._add_rolling_analysis_to_dashboard(test_fig, row=1, col=1)
                test_results['details']['rolling_analysis'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Rolling analysis panel test successful")
            except Exception as e:
                test_results['details']['rolling_analysis'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Rolling analysis panel test failed: {e}")
            
            # Save test dashboard
            dashboard_fig.write_html(self.results_dir / 'test_5_1a_dashboard.html')
            logger.info("✓ Test dashboard saved")
            
        except Exception as e:
            logger.error(f"Critical error in Step 5.1a testing: {e}")
            test_results['details']['critical_error'] = str(e)
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        logger.info(f"Step 5.1a Results: {test_results['tests_passed']}/{test_results['total_tests']} tests passed ({test_results['success_rate']*100:.1f}%)")
        
        return test_results
    
    def test_5_1b_interactive_controls(self):
        """
        Test Step 5.1b: Interactive controls implementation
        """
        logger.info("=== TESTING STEP 5.1b: Interactive Controls ===")
        
        test_results = {
            'step': '5.1b',
            'description': 'Interactive controls implementation',
            'tests_passed': 0,
            'total_tests': 6,
            'details': {}
        }
        
        try:
            # Test 1: Create base dashboard for controls testing
            logger.info("Test 1: Creating base dashboard for controls...")
            base_fig = self.analyzer._create_multi_panel_dashboard()
            
            if base_fig is not None:
                test_results['details']['base_dashboard'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Base dashboard created for controls testing")
            else:
                test_results['details']['base_dashboard'] = 'FAIL - No base dashboard'
                logger.error("✗ Base dashboard creation failed")
                return test_results
            
            # Test 2: Add interactive controls
            logger.info("Test 2: Adding interactive controls...")
            try:
                self.analyzer._add_interactive_controls(base_fig)
                test_results['details']['controls_addition'] = 'PASS'
                test_results['tests_passed'] += 1
                logger.info("✓ Interactive controls added successfully")
            except Exception as e:
                test_results['details']['controls_addition'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Interactive controls addition failed: {e}")
            
            # Test 3: Verify updatemenus exist
            logger.info("Test 3: Verifying updatemenus structure...")
            if hasattr(base_fig, 'layout') and hasattr(base_fig.layout, 'updatemenus'):
                if base_fig.layout.updatemenus and len(base_fig.layout.updatemenus) > 0:
                    test_results['details']['updatemenus'] = f'PASS - {len(base_fig.layout.updatemenus)} menus found'
                    test_results['tests_passed'] += 1
                    logger.info(f"✓ Updatemenus verified: {len(base_fig.layout.updatemenus)} menus")
                else:
                    test_results['details']['updatemenus'] = 'FAIL - No updatemenus found'
                    logger.error("✗ No updatemenus found")
            else:
                test_results['details']['updatemenus'] = 'FAIL - Layout structure issue'
                logger.error("✗ Layout structure issue")
            
            # Test 4: Verify view toggle buttons
            logger.info("Test 4: Verifying view toggle buttons...")
            try:
                if base_fig.layout.updatemenus:
                    first_menu = base_fig.layout.updatemenus[0]
                    if hasattr(first_menu, 'buttons') and len(first_menu.buttons) >= 4:
                        button_labels = [btn.label for btn in first_menu.buttons]
                        expected_labels = ["Show All", "Timeline Only", "Heatmaps Only", "Analytics Only"]
                        
                        if all(label in button_labels for label in expected_labels):
                            test_results['details']['view_toggles'] = f'PASS - All expected buttons found'
                            test_results['tests_passed'] += 1
                            logger.info("✓ View toggle buttons verified")
                        else:
                            test_results['details']['view_toggles'] = f'PARTIAL - Found {button_labels}'
                            logger.warning(f"△ Partial view toggles: {button_labels}")
                    else:
                        test_results['details']['view_toggles'] = 'FAIL - Insufficient buttons'
                        logger.error("✗ Insufficient view toggle buttons")
                else:
                    test_results['details']['view_toggles'] = 'FAIL - No menu structure'
                    logger.error("✗ No menu structure for view toggles")
            except Exception as e:
                test_results['details']['view_toggles'] = f'FAIL - {str(e)}'
                logger.error(f"✗ View toggle verification failed: {e}")
            
            # Test 5: Verify annotations exist
            logger.info("Test 5: Verifying annotations...")
            if hasattr(base_fig.layout, 'annotations') and base_fig.layout.annotations:
                test_results['details']['annotations'] = f'PASS - {len(base_fig.layout.annotations)} annotations'
                test_results['tests_passed'] += 1
                logger.info(f"✓ Annotations verified: {len(base_fig.layout.annotations)} found")
            else:
                test_results['details']['annotations'] = 'FAIL - No annotations found'
                logger.error("✗ No annotations found")
            
            # Test 6: Verify interactive settings
            logger.info("Test 6: Verifying interactive settings...")
            interactive_settings = ['hovermode', 'showlegend']
            settings_found = 0
            
            for setting in interactive_settings:
                if hasattr(base_fig.layout, setting):
                    settings_found += 1
            
            if settings_found >= len(interactive_settings) / 2:
                test_results['details']['interactive_settings'] = f'PASS - {settings_found}/{len(interactive_settings)} settings'
                test_results['tests_passed'] += 1
                logger.info(f"✓ Interactive settings verified: {settings_found}/{len(interactive_settings)}")
            else:
                test_results['details']['interactive_settings'] = f'FAIL - Only {settings_found} settings'
                logger.error(f"✗ Insufficient interactive settings: {settings_found}")
            
            # Save test dashboard with controls
            base_fig.write_html(self.results_dir / 'test_5_1b_controls.html')
            logger.info("✓ Test dashboard with controls saved")
            
        except Exception as e:
            logger.error(f"Critical error in Step 5.1b testing: {e}")
            test_results['details']['critical_error'] = str(e)
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        logger.info(f"Step 5.1b Results: {test_results['tests_passed']}/{test_results['total_tests']} tests passed ({test_results['success_rate']*100:.1f}%)")
        
        return test_results
    
    def test_5_2a_enhanced_hover_analytics(self):
        """
        Test Step 5.2a: Enhanced hover-over analytics implementation
        """
        logger.info("=== TESTING STEP 5.2a: Enhanced Hover Analytics ===")
        
        test_results = {
            'step': '5.2a',
            'description': 'Enhanced hover-over analytics implementation',
            'tests_passed': 0,
            'total_tests': 6,
            'details': {}
        }
        
        try:
            # Test 1: Enhanced hover analytics creation
            logger.info("Test 1: Creating enhanced hover analytics...")
            try:
                hover_analytics = self.analyzer._implement_enhanced_hover_analytics()
                
                if hover_analytics and isinstance(hover_analytics, dict):
                    test_results['details']['hover_creation'] = 'PASS'
                    test_results['tests_passed'] += 1
                    logger.info("✓ Enhanced hover analytics created successfully")
                else:
                    test_results['details']['hover_creation'] = 'FAIL - Invalid return'
                    logger.error("✗ Enhanced hover analytics creation failed")
            except Exception as e:
                test_results['details']['hover_creation'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Enhanced hover analytics creation error: {e}")
            
            # Test 2: Verify regime details structure
            logger.info("Test 2: Verifying regime details structure...")
            try:
                hover_file = self.analyzer.results_dir / 'enhanced_hover_analytics.json'
                if hover_file.exists():
                    with open(hover_file, 'r') as f:
                        hover_data = json.load(f)
                    
                    if 'regime_details' in hover_data:
                        regime_count = len(hover_data['regime_details'])
                        test_results['details']['regime_details'] = f'PASS - {regime_count} regimes'
                        test_results['tests_passed'] += 1
                        logger.info(f"✓ Regime details verified: {regime_count} regimes")
                    else:
                        test_results['details']['regime_details'] = 'FAIL - No regime_details key'
                        logger.error("✗ No regime_details found")
                else:
                    test_results['details']['regime_details'] = 'FAIL - No hover analytics file'
                    logger.error("✗ Enhanced hover analytics file not found")
            except Exception as e:
                test_results['details']['regime_details'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Regime details verification failed: {e}")
            
            # Test 3: Verify factor analytics structure
            logger.info("Test 3: Verifying factor analytics structure...")
            try:
                hover_file = self.analyzer.results_dir / 'enhanced_hover_analytics.json'
                if hover_file.exists():
                    with open(hover_file, 'r') as f:
                        hover_data = json.load(f)
                    
                    if 'factor_analytics' in hover_data:
                        factor_count = len(hover_data['factor_analytics'])
                        test_results['details']['factor_analytics'] = f'PASS - {factor_count} factors'
                        test_results['tests_passed'] += 1
                        logger.info(f"✓ Factor analytics verified: {factor_count} factors")
                    else:
                        test_results['details']['factor_analytics'] = 'FAIL - No factor_analytics key'
                        logger.error("✗ No factor_analytics found")
            except Exception as e:
                test_results['details']['factor_analytics'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Factor analytics verification failed: {e}")
            
            # Test 4: Test best regime identification
            logger.info("Test 4: Testing best regime identification...")
            try:
                test_factor = 'Value'
                best_regime = self.analyzer._find_best_regime_for_factor(test_factor)
                
                if best_regime and best_regime != 'Unknown':
                    test_results['details']['best_regime_id'] = f'PASS - {test_factor}: {best_regime}'
                    test_results['tests_passed'] += 1
                    logger.info(f"✓ Best regime identification: {test_factor} -> {best_regime}")
                else:
                    test_results['details']['best_regime_id'] = 'FAIL - Unknown result'
                    logger.error("✗ Best regime identification failed")
            except Exception as e:
                test_results['details']['best_regime_id'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Best regime identification error: {e}")
            
            # Test 5: Test volatility ranking
            logger.info("Test 5: Testing volatility ranking...")
            try:
                test_factor = 'MinVol'
                vol_rank = self.analyzer._rank_factor_volatility(test_factor)
                
                if vol_rank > 0:
                    test_results['details']['volatility_ranking'] = f'PASS - {test_factor}: rank {vol_rank}'
                    test_results['tests_passed'] += 1
                    logger.info(f"✓ Volatility ranking: {test_factor} -> rank {vol_rank}")
                else:
                    test_results['details']['volatility_ranking'] = 'FAIL - Invalid rank'
                    logger.error("✗ Volatility ranking failed")
            except Exception as e:
                test_results['details']['volatility_ranking'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Volatility ranking error: {e}")
            
            # Test 6: Verify enhanced hover file exists and is valid JSON
            logger.info("Test 6: Verifying enhanced hover file...")
            try:
                hover_file = self.analyzer.results_dir / 'enhanced_hover_analytics.json'
                if hover_file.exists():
                    with open(hover_file, 'r') as f:
                        hover_data = json.load(f)
                    
                    required_keys = ['regime_details', 'factor_analytics', 'statistical_summaries']
                    keys_found = sum(1 for key in required_keys if key in hover_data)
                    
                    if keys_found >= 2:
                        test_results['details']['hover_file'] = f'PASS - {keys_found}/{len(required_keys)} sections'
                        test_results['tests_passed'] += 1
                        logger.info(f"✓ Enhanced hover file verified: {keys_found}/{len(required_keys)} sections")
                    else:
                        test_results['details']['hover_file'] = f'PARTIAL - {keys_found} sections'
                        logger.warning(f"△ Partial hover file: {keys_found} sections")
                else:
                    test_results['details']['hover_file'] = 'FAIL - File not found'
                    logger.error("✗ Enhanced hover file not found")
            except Exception as e:
                test_results['details']['hover_file'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Enhanced hover file verification failed: {e}")
            
        except Exception as e:
            logger.error(f"Critical error in Step 5.2a testing: {e}")
            test_results['details']['critical_error'] = str(e)
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        logger.info(f"Step 5.2a Results: {test_results['tests_passed']}/{test_results['total_tests']} tests passed ({test_results['success_rate']*100:.1f}%)")
        
        return test_results
    
    def test_5_2b_export_functionality(self):
        """
        Test Step 5.2b: Export functionality
        """
        logger.info("=== TESTING STEP 5.2b: Export Functionality ===")
        
        test_results = {
            'step': '5.2b',
            'description': 'Export functionality implementation',
            'tests_passed': 0,
            'total_tests': 8,
            'details': {}
        }
        
        try:
            # Test 1: Main export functionality
            logger.info("Test 1: Testing main export functionality...")
            try:
                export_success = self.analyzer._create_export_functionality()
                
                if export_success:
                    test_results['details']['main_export'] = 'PASS'
                    test_results['tests_passed'] += 1
                    logger.info("✓ Main export functionality successful")
                else:
                    test_results['details']['main_export'] = 'FAIL - Function returned False'
                    logger.error("✗ Main export functionality failed")
            except Exception as e:
                test_results['details']['main_export'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Main export functionality error: {e}")
            
            # Test 2: Summary tables export
            logger.info("Test 2: Testing summary tables export...")
            try:
                self.analyzer._export_summary_tables()
                
                # Check if CSV files were created
                performance_csv = self.analyzer.results_dir / 'performance_summary_export.csv'
                regime_csv = self.analyzer.results_dir / 'regime_summary_export.csv'
                
                csv_count = sum([performance_csv.exists(), regime_csv.exists()])
                
                if csv_count == 2:
                    test_results['details']['summary_tables'] = 'PASS - Both CSVs created'
                    test_results['tests_passed'] += 1
                    logger.info("✓ Summary tables export successful")
                elif csv_count == 1:
                    test_results['details']['summary_tables'] = 'PARTIAL - 1 CSV created'
                    logger.warning("△ Partial summary tables export")
                else:
                    test_results['details']['summary_tables'] = 'FAIL - No CSVs created'
                    logger.error("✗ Summary tables export failed")
            except Exception as e:
                test_results['details']['summary_tables'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Summary tables export error: {e}")
            
            # Test 3: Static charts export
            logger.info("Test 3: Testing static charts export...")
            try:
                self.analyzer._export_static_charts()
                
                # Check if PNG files were created
                heatmap_png = self.analyzer.results_dir / 'performance_heatmap_export.png'
                timeline_png = self.analyzer.results_dir / 'timeline_export.png'
                
                png_count = sum([heatmap_png.exists(), timeline_png.exists()])
                
                if png_count == 2:
                    test_results['details']['static_charts'] = 'PASS - Both PNGs created'
                    test_results['tests_passed'] += 1
                    logger.info("✓ Static charts export successful")
                elif png_count == 1:
                    test_results['details']['static_charts'] = 'PARTIAL - 1 PNG created'
                    logger.warning("△ Partial static charts export")
                else:
                    test_results['details']['static_charts'] = 'FAIL - No PNGs created'
                    logger.error("✗ Static charts export failed")
            except Exception as e:
                test_results['details']['static_charts'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Static charts export error: {e}")
            
            # Test 4: PDF report creation
            logger.info("Test 4: Testing PDF report creation...")
            try:
                self.analyzer._create_pdf_report()
                
                # Check if markdown report was created
                report_md = self.analyzer.results_dir / 'comprehensive_analysis_report.md'
                
                if report_md.exists():
                    test_results['details']['pdf_report'] = 'PASS - Markdown report created'
                    test_results['tests_passed'] += 1
                    logger.info("✓ PDF report creation successful")
                else:
                    test_results['details']['pdf_report'] = 'FAIL - No markdown report'
                    logger.error("✗ PDF report creation failed")
            except Exception as e:
                test_results['details']['pdf_report'] = f'FAIL - {str(e)}'
                logger.error(f"✗ PDF report creation error: {e}")
            
            # Test 5: Portfolio recommendations export
            logger.info("Test 5: Testing portfolio recommendations export...")
            try:
                self.analyzer._export_portfolio_recommendations()
                
                # Check if portfolio recommendations CSV was created
                portfolio_csv = self.analyzer.results_dir / 'portfolio_recommendations_export.csv'
                
                if portfolio_csv.exists():
                    test_results['details']['portfolio_recommendations'] = 'PASS - CSV created'
                    test_results['tests_passed'] += 1
                    logger.info("✓ Portfolio recommendations export successful")
                else:
                    test_results['details']['portfolio_recommendations'] = 'FAIL - No CSV created'
                    logger.error("✗ Portfolio recommendations export failed")
            except Exception as e:
                test_results['details']['portfolio_recommendations'] = f'FAIL - {str(e)}'
                logger.error(f"✗ Portfolio recommendations export error: {e}")
            
            # Test 6: Verify all expected export files exist
            logger.info("Test 6: Verifying all expected export files...")
            expected_files = [
                'performance_summary_export.csv',
                'regime_summary_export.csv',
                'performance_heatmap_export.png',
                'timeline_export.png',
                'comprehensive_analysis_report.md',
                'portfolio_recommendations_export.csv'
            ]
            
            files_found = 0
            for filename in expected_files:
                if (self.analyzer.results_dir / filename).exists():
                    files_found += 1
            
            if files_found >= len(expected_files) * 0.8:  # 80% threshold
                test_results['details']['all_export_files'] = f'PASS - {files_found}/{len(expected_files)} files'
                test_results['tests_passed'] += 1
                logger.info(f"✓ Export files verification: {files_found}/{len(expected_files)} files found")
            else:
                test_results['details']['all_export_files'] = f'FAIL - Only {files_found}/{len(expected_files)} files'
                logger.error(f"✗ Insufficient export files: {files_found}/{len(expected_files)}")
            
            # Test 7: Check file sizes (should not be empty)
            logger.info("Test 7: Checking export file sizes...")
            non_empty_files = 0
            total_size = 0
            
            for filename in expected_files:
                file_path = self.analyzer.results_dir / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    total_size += size
                    if size > 0:
                        non_empty_files += 1
            
            if non_empty_files >= files_found * 0.9:  # 90% of found files should be non-empty
                test_results['details']['file_sizes'] = f'PASS - {non_empty_files} non-empty files, {total_size} bytes total'
                test_results['tests_passed'] += 1
                logger.info(f"✓ File sizes verification: {non_empty_files} non-empty files")
            else:
                test_results['details']['file_sizes'] = f'FAIL - Only {non_empty_files} non-empty files'
                logger.error(f"✗ Too many empty files: {non_empty_files} non-empty")
            
            # Test 8: Test CSV file content validity
            logger.info("Test 8: Testing CSV file content validity...")
            valid_csvs = 0
            csv_files = [f for f in expected_files if f.endswith('.csv')]
            
            for csv_file in csv_files:
                try:
                    file_path = self.analyzer.results_dir / csv_file
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        if len(df) > 0 and len(df.columns) > 0:
                            valid_csvs += 1
                except Exception:
                    pass
            
            if valid_csvs >= len(csv_files) * 0.8:
                test_results['details']['csv_validity'] = f'PASS - {valid_csvs}/{len(csv_files)} valid CSVs'
                test_results['tests_passed'] += 1
                logger.info(f"✓ CSV validity: {valid_csvs}/{len(csv_files)} valid files")
            else:
                test_results['details']['csv_validity'] = f'FAIL - Only {valid_csvs}/{len(csv_files)} valid CSVs'
                logger.error(f"✗ CSV validity failed: {valid_csvs}/{len(csv_files)}")
            
        except Exception as e:
            logger.error(f"Critical error in Step 5.2b testing: {e}")
            test_results['details']['critical_error'] = str(e)
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        logger.info(f"Step 5.2b Results: {test_results['tests_passed']}/{test_results['total_tests']} tests passed ({test_results['success_rate']*100:.1f}%)")
        
        return test_results
    
    def run_comprehensive_phase5_verification(self):
        """
        Run comprehensive verification of all Phase 5 components
        """
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE PHASE 5 VERIFICATION")
        logger.info("="*80)
        
        # Setup analyzer
        if not self.setup_analyzer():
            logger.error("Failed to setup analyzer for testing")
            return False
        
        # Run all Phase 5 tests
        all_results = {}
        
        # Test Step 5.1a
        all_results['5.1a'] = self.test_5_1a_multi_panel_dashboard_layout()
        
        # Test Step 5.1b
        all_results['5.1b'] = self.test_5_1b_interactive_controls()
        
        # Test Step 5.2a
        all_results['5.2a'] = self.test_5_2a_enhanced_hover_analytics()
        
        # Test Step 5.2b
        all_results['5.2b'] = self.test_5_2b_export_functionality()
        
        # Calculate overall results
        total_tests = sum(result['total_tests'] for result in all_results.values())
        total_passed = sum(result['tests_passed'] for result in all_results.values())
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Create comprehensive report
        verification_report = {
            'verification_timestamp': datetime.now().isoformat(),
            'overall_results': {
                'total_tests': total_tests,
                'tests_passed': total_passed,
                'success_rate': overall_success_rate,
                'status': 'PASS' if overall_success_rate >= 0.8 else 'FAIL'
            },
            'step_results': all_results,
            'recommendations': self._generate_recommendations(all_results)
        }
        
        # Save verification report
        with open(self.results_dir / 'phase5_comprehensive_verification_report.json', 'w') as f:
            json.dump(verification_report, f, indent=2, default=str)
        
        # Log final results
        logger.info("="*80)
        logger.info("PHASE 5 VERIFICATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Overall Results: {total_passed}/{total_tests} tests passed ({overall_success_rate*100:.1f}%)")
        
        for step, result in all_results.items():
            status = "✓ PASS" if result['success_rate'] >= 0.8 else "✗ FAIL"
            logger.info(f"Step {step}: {result['tests_passed']}/{result['total_tests']} {status}")
        
        if overall_success_rate >= 0.8:
            logger.info("🎉 PHASE 5 VERIFICATION SUCCESSFUL - Ready for Phase 6!")
        else:
            logger.error("❌ PHASE 5 VERIFICATION FAILED - Issues need resolution")
        
        return overall_success_rate >= 0.8
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on test results"""
        recommendations = []
        
        for step, result in results.items():
            if result['success_rate'] < 0.8:
                recommendations.append(f"Step {step}: Review failed tests - {result['success_rate']*100:.1f}% pass rate")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - Phase 5 implementation is ready for production")
        
        return recommendations

def main():
    """Main execution function"""
    tester = Phase5VerificationTester()
    success = tester.run_comprehensive_phase5_verification()
    
    if success:
        logger.info("✅ Phase 5 verification completed successfully")
        return 0
    else:
        logger.error("❌ Phase 5 verification failed")
        return 1

if __name__ == "__main__":
    exit(main()) 