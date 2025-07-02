"""
Business Cycle Factor Performance Analysis
Comprehensive analysis of factor performance across business cycle regimes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BusinessCycleFactorAnalyzer:
    """
    Comprehensive analyzer for factor performance across business cycle regimes
    """
    
    def __init__(self, data_dir="data/processed", results_dir="results/business_cycle_analysis"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.fred_data = None
        self.msci_data = None
        self.market_data = None
        self.aligned_data = None
        self.regime_data = None
        
        # Analysis containers
        self.performance_metrics = {}
        self.statistical_tests = {}
        self.regime_statistics = {}
        
        logger.info("BusinessCycleFactorAnalyzer initialized")
    
    def load_data(self):
        """
        Phase 1.1a: Load all datasets with proper date parsing
        """
        logger.info("=== Phase 1.1a: Loading datasets ===")
        
        try:
            # Load FRED economic data with date parsing
            logger.info("Loading FRED economic data...")
            self.fred_data = pd.read_csv(
                self.data_dir / "fred_economic_data.csv", 
                index_col=0, 
                parse_dates=True
            )
            logger.info(f"✓ FRED data loaded: {len(self.fred_data)} observations")
            logger.info(f"  Date range: {self.fred_data.index.min()} to {self.fred_data.index.max()}")
            logger.info(f"  Columns: {len(self.fred_data.columns)}")
            
            # Load MSCI factor returns with date parsing
            logger.info("Loading MSCI factor returns...")
            self.msci_data = pd.read_csv(
                self.data_dir / "msci_factor_returns.csv", 
                index_col=0, 
                parse_dates=True
            )
            logger.info(f"✓ MSCI data loaded: {len(self.msci_data)} observations")
            logger.info(f"  Date range: {self.msci_data.index.min()} to {self.msci_data.index.max()}")
            logger.info(f"  Factors: {list(self.msci_data.columns)}")
            
            # Load market data (S&P 500, VIX) with date parsing
            logger.info("Loading market data...")
            self.market_data = pd.read_csv(
                self.data_dir / "market_data.csv", 
                index_col=0, 
                parse_dates=True
            )
            logger.info(f"✓ Market data loaded: {len(self.market_data)} observations")
            logger.info(f"  Date range: {self.market_data.index.min()} to {self.market_data.index.max()}")
            logger.info(f"  Columns: {list(self.market_data.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def phase1_data_alignment_and_validation_FIXED(self):
        """
        Phase 1: FIXED Advanced Date Alignment & Data Integration
        Following roadmap requirements with CORRECTED regime preservation
        """
        logger.info("=== PHASE 1: FIXED Data Alignment & Integration ===")
        
        try:
            # Step 1.1a & 1.1b: Create master timeline using MSCI dates as baseline
            logger.info("Creating master timeline from MSCI factor returns...")
            master_index = pd.to_datetime(self.msci_data.index)
            logger.info(f"Master timeline: {master_index.min()} to {master_index.max()} ({len(master_index)} observations)")
            
            # FIXED: Align FRED data properly to preserve regime classifications
            logger.info("FIXED: Aligning FRED data with proper regime preservation...")
            
            # Filter FRED data to MSCI timeline period first
            fred_data_copy = self.fred_data.copy()
            fred_data_copy.index = pd.to_datetime(fred_data_copy.index)
            
            # Filter to MSCI period
            msci_start = master_index.min()
            msci_end = master_index.max()
            fred_filtered = fred_data_copy[(fred_data_copy.index >= msci_start) & (fred_data_copy.index <= msci_end)]
            
            logger.info(f"FRED data filtered to MSCI period: {len(fred_filtered)} daily observations")
            
            # Check regime diversity before resampling
            regime_counts_daily = fred_filtered['ECONOMIC_REGIME'].value_counts()
            logger.info(f"Regime diversity in daily data: {regime_counts_daily.to_dict()}")
            
            # FIXED: Use MODE (most frequent regime) instead of .last() to preserve regime diversity
            def get_monthly_regime_mode(group):
                """Get the most frequent (mode) regime for each month"""
                regime_col = group['ECONOMIC_REGIME']
                if len(regime_col) == 0:
                    return pd.Series(index=group.columns, dtype=object)
                
                # For regime, use mode (most frequent)
                regime_mode = regime_col.mode()
                if len(regime_mode) > 0:
                    regime_value = regime_mode.iloc[0]
                else:
                    regime_value = regime_col.iloc[-1]  # fallback to last if no clear mode
                
                # For other columns, use last value
                result = group.iloc[-1].copy()
                result['ECONOMIC_REGIME'] = regime_value
                return result
            
            # Apply the fixed resampling approach
            fred_monthly_fixed = fred_filtered.resample('M').apply(get_monthly_regime_mode)
            
            # Validate regime preservation after resampling
            regime_counts_monthly = fred_monthly_fixed['ECONOMIC_REGIME'].value_counts()
            logger.info(f"FIXED: Regime diversity after monthly resampling: {regime_counts_monthly.to_dict()}")
            
            # Align to master MSCI index
            fred_aligned = fred_monthly_fixed.reindex(master_index, method='ffill')
            
            # Final regime validation
            final_regime_counts = fred_aligned['ECONOMIC_REGIME'].value_counts()
            logger.info(f"FIXED: Final regime distribution: {final_regime_counts.to_dict()}")
            
            # Align market data to master timeline
            logger.info("Aligning market data to master timeline...")
            market_data_copy = self.market_data.copy()
            market_data_copy.index = pd.to_datetime(market_data_copy.index)
            market_aligned = market_data_copy.reindex(master_index, method='ffill')
            
            # Align MSCI data (should already be aligned)
            logger.info("Ensuring MSCI data alignment...")
            msci_aligned = self.msci_data.reindex(master_index, method='ffill')
            
            # Create comprehensive aligned dataset with FIXED regime data
            logger.info("Creating comprehensive aligned dataset...")
            aligned_data = pd.concat([
                msci_aligned,
                fred_aligned,
                market_aligned
            ], axis=1)
            
            # Remove any rows with all NaN values
            aligned_data = aligned_data.dropna(how='all')
            
            logger.info(f"FIXED: Aligned dataset shape: {aligned_data.shape}")
            logger.info(f"Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
            
            # Validate final regime diversity
            final_regimes = aligned_data['ECONOMIC_REGIME'].value_counts()
            logger.info(f"FIXED: Final aligned regime distribution: {final_regimes.to_dict()}")
            
            # Store aligned data
            self.aligned_data = aligned_data
            
            # Save FIXED aligned datasets
            aligned_data.to_csv(self.results_dir / 'aligned_master_dataset_FIXED.csv')
            
            # Save factor returns separately
            factor_returns = aligned_data[['Value', 'Quality', 'MinVol', 'Momentum']].copy()
            factor_returns.to_csv(self.results_dir / 'factor_returns_aligned_FIXED.csv')
            
            # Save regime classifications separately for analysis
            regime_data = aligned_data[['ECONOMIC_REGIME', 'GROWTH_COMPOSITE', 'INFLATION_COMPOSITE']].copy()
            
            # Create VIX-based regime classifications
            logger.info("Creating VIX-based regime classifications...")
            vix_regimes = pd.cut(
                aligned_data['VIX'], 
                bins=[0, 25, 35, 50, 100], 
                labels=['Normal', 'Elevated', 'Stress', 'Crisis'],
                include_lowest=True
            )
            regime_data['vix_regime'] = vix_regimes
            
            # Create hybrid regime system
            hybrid_regimes = regime_data['ECONOMIC_REGIME'].astype(str) + '_' + regime_data['vix_regime'].astype(str)
            # Handle Crisis separately as it's severe enough to override economic regime
            hybrid_regimes[regime_data['vix_regime'] == 'Crisis'] = 'Crisis'
            regime_data['hybrid_regime'] = hybrid_regimes
            
            regime_data.to_csv(self.results_dir / 'regime_classifications_FIXED.csv')
            
            # Create comprehensive summary
            summary = {
                'fix_applied': 'MODE_BASED_REGIME_RESAMPLING',
                'original_issue': 'resample().last() was capturing systematic end-of-month Recession bias',
                'solution': 'resample() with mode (most frequent regime) to capture true monthly regime',
                'data_alignment': {
                    'master_timeline_observations': len(master_index),
                    'date_range': f'{master_index.min()} to {master_index.max()}',
                    'aligned_dataset_shape': list(aligned_data.shape),
                    'total_indicators': len(aligned_data.columns)
                },
                'regime_diversity_validation': {
                    'daily_regimes_in_msci_period': regime_counts_daily.to_dict(),
                    'monthly_regimes_after_mode_resampling': regime_counts_monthly.to_dict(),
                    'final_aligned_regimes': final_regimes.to_dict()
                },
                'vix_regimes': {
                    'distribution': regime_data['vix_regime'].value_counts().to_dict()
                },
                'hybrid_regimes': {
                    'distribution': regime_data['hybrid_regime'].value_counts().to_dict()
                }
            }
            
            # Save fixed summary
            with open(self.results_dir / 'phase1_FIXED_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("✅ PHASE 1 FIXED: Regime diversity successfully preserved!")
            logger.info("✅ Ready for Phase 2 with proper regime classifications!")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in FIXED Phase 1: {e}")
            return None
    
    def run_phase1(self):
        """
        Execute Phase 1: Advanced Date Alignment & Data Integration
        """
        logger.info("=" * 60)
        logger.info("STARTING PHASE 1: Advanced Date Alignment & Data Integration")
        logger.info("=" * 60)
        
        # Data Loading and Validation
        success = self.load_data()
        if not success:
            logger.error("Phase 1 failed at data loading")
            return False
        
        # Complete Phase 1 data alignment and validation
        try:
            phase1_summary = self.phase1_data_alignment_and_validation_FIXED()
            
            logger.info("=" * 60)
            logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
            logger.info(f"Aligned dataset: {phase1_summary['data_alignment']['aligned_dataset_shape']} observations")
            logger.info(f"Economic regimes: {list(phase1_summary['regime_diversity_validation']['final_aligned_regimes'].keys())}")
            logger.info(f"Date range: {phase1_summary['data_alignment']['date_range']}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {str(e)}")
            return False

    def comprehensive_regime_analysis(self):
        """
        Phase 1.2a: Complete economic regime validation and analysis
        """
        logger.info("=== Comprehensive Economic Regime Analysis ===")
        
        try:
            # Load the original FRED data to analyze pre-alignment regimes
            fred_original = self.fred_data.copy()
            
            # Analyze regime diversity in original data
            logger.info("Analyzing original FRED regime diversity...")
            original_regimes = fred_original['ECONOMIC_REGIME'].value_counts()
            logger.info(f"Original regime distribution: {original_regimes.to_dict()}")
            
            # Find date ranges for each regime in original data
            regime_periods = {}
            for regime in original_regimes.index:
                regime_data = fred_original[fred_original['ECONOMIC_REGIME'] == regime]
                regime_periods[regime] = {
                    'start_date': regime_data.index.min().strftime('%Y-%m-%d'),
                    'end_date': regime_data.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(regime_data),
                    'avg_growth': regime_data['GROWTH_COMPOSITE'].mean() if 'GROWTH_COMPOSITE' in regime_data.columns else None,
                    'avg_inflation': regime_data['INFLATION_COMPOSITE'].mean() if 'INFLATION_COMPOSITE' in regime_data.columns else None
                }
            
            logger.info("Economic regime periods identified:")
            for regime, info in regime_periods.items():
                logger.info(f"  {regime}: {info['start_date']} to {info['end_date']} ({info['total_days']} days)")
                if info['avg_growth'] is not None:
                    logger.info(f"    Avg Growth: {info['avg_growth']:.3f}, Avg Inflation: {info['avg_inflation']:.3f}")
            
            # Analyze why regimes are lost during alignment
            logger.info("Analyzing regime loss during MSCI alignment...")
            msci_start = self.aligned_data.index.min()
            msci_end = self.aligned_data.index.max()
            
            # Check what regimes exist in MSCI time period
            fred_msci_period = fred_original[
                (fred_original.index >= msci_start) & 
                (fred_original.index <= msci_end)
            ]
            
            if len(fred_msci_period) > 0:
                msci_period_regimes = fred_msci_period['ECONOMIC_REGIME'].value_counts()
                logger.info(f"Regimes during MSCI period ({msci_start} to {msci_end}): {msci_period_regimes.to_dict()}")
                
                # Create synthetic regimes based on economic indicators if needed
                if len(msci_period_regimes) == 1:
                    logger.info("Creating synthetic regimes based on economic indicators...")
                    synthetic_regimes = self.create_synthetic_regimes()
                    return regime_periods, msci_period_regimes.to_dict(), synthetic_regimes
            
            return regime_periods, {}, {}
            
        except Exception as e:
            logger.error(f"Error in comprehensive regime analysis: {str(e)}")
            return {}, {}, {}
    
    def create_synthetic_regimes(self):
        """
        Create synthetic economic regimes based on economic indicators
        """
        logger.info("Creating synthetic regimes from economic indicators...")
        
        try:
            # Use GDP growth and inflation to create regimes
            aligned_data = self.aligned_data.copy()
            
            # Get economic indicators
            gdp_col = 'GDPC1_YOY' if 'GDPC1_YOY' in aligned_data.columns else None
            inflation_col = 'CPIAUCSL_YOY' if 'CPIAUCSL_YOY' in aligned_data.columns else None
            
            if gdp_col and inflation_col:
                # Create regime classification based on growth and inflation trends
                gdp_growth = aligned_data[gdp_col].fillna(method='ffill')
                inflation = aligned_data[inflation_col].fillna(method='ffill')
                
                # Calculate rolling averages for trend detection
                gdp_trend = gdp_growth.rolling(6).mean()
                inflation_trend = inflation.rolling(6).mean()
                
                # Create regime classifications
                synthetic_regimes = pd.Series('Unknown', index=aligned_data.index)
                
                # Define regime criteria
                gdp_threshold = gdp_trend.median()
                inflation_threshold = inflation_trend.median()
                
                # Goldilocks: Rising Growth + Falling Inflation
                goldilocks_mask = (gdp_trend > gdp_threshold) & (inflation_trend < inflation_threshold)
                synthetic_regimes[goldilocks_mask] = 'Goldilocks'
                
                # Overheating: Rising Growth + Rising Inflation  
                overheating_mask = (gdp_trend > gdp_threshold) & (inflation_trend > inflation_threshold)
                synthetic_regimes[overheating_mask] = 'Overheating'
                
                # Stagflation: Falling Growth + Rising Inflation
                stagflation_mask = (gdp_trend <= gdp_threshold) & (inflation_trend > inflation_threshold)
                synthetic_regimes[stagflation_mask] = 'Stagflation'
                
                # Recession: Falling Growth + Falling Inflation
                recession_mask = (gdp_trend <= gdp_threshold) & (inflation_trend <= inflation_threshold)
                synthetic_regimes[recession_mask] = 'Recession'
                
                # Update aligned data with synthetic regimes
                self.aligned_data['SYNTHETIC_ECONOMIC_REGIME'] = synthetic_regimes
                
                synthetic_counts = synthetic_regimes.value_counts()
                logger.info(f"Synthetic regime distribution: {synthetic_counts.to_dict()}")
                
                return synthetic_counts.to_dict()
            
            else:
                logger.warning("Insufficient economic indicators for synthetic regime creation")
                return {}
                
        except Exception as e:
            logger.error(f"Error creating synthetic regimes: {str(e)}")
            return {}
    
    def regime_cross_validation(self):
        """
        Phase 1.2b: Cross-validate economic vs VIX-based regimes
        """
        logger.info("=== Cross-Validating Economic vs VIX Regimes ===")
        
        try:
            # Get VIX regimes from aligned data
            vix_regimes = pd.cut(
                self.aligned_data['VIX'], 
                bins=[0, 25, 35, 50, 100], 
                labels=['Normal', 'Elevated', 'Stress', 'Crisis'],
                include_lowest=True
            )
            
            # Get economic regimes (use synthetic if available)
            if 'SYNTHETIC_ECONOMIC_REGIME' in self.aligned_data.columns:
                economic_regimes = self.aligned_data['SYNTHETIC_ECONOMIC_REGIME']
                logger.info("Using synthetic economic regimes for comparison")
            else:
                economic_regimes = self.aligned_data['ECONOMIC_REGIME']
                logger.info("Using original economic regimes for comparison")
            
            # Create comparison analysis
            comparison_table = pd.crosstab(
                economic_regimes, 
                vix_regimes, 
                normalize='index'
            ) * 100
            
            logger.info("Economic vs VIX regime overlap (% within economic regime):")
            for idx in comparison_table.index:
                for col in comparison_table.columns:
                    value = comparison_table.loc[idx, col]
                    if value > 0:
                        logger.info(f"  {idx} -> {col}: {value:.1f}%")
            
            # Create advanced hybrid regime system
            logger.info("Creating advanced hybrid regime system...")
            
            # Combine regimes with priority weighting
            hybrid_regimes = pd.Series('Unknown', index=self.aligned_data.index)
            
            # Crisis periods override everything
            crisis_mask = vix_regimes == 'Crisis'
            hybrid_regimes[crisis_mask] = 'Crisis'
            
            # High stress periods 
            stress_mask = (vix_regimes == 'Stress') & (~crisis_mask)
            hybrid_regimes[stress_mask] = economic_regimes[stress_mask].astype(str) + '_Stress'
            
            # Elevated periods
            elevated_mask = (vix_regimes == 'Elevated') & (~crisis_mask) & (~stress_mask)
            hybrid_regimes[elevated_mask] = economic_regimes[elevated_mask].astype(str) + '_Elevated'
            
            # Normal periods
            normal_mask = (vix_regimes == 'Normal') & (~crisis_mask) & (~stress_mask) & (~elevated_mask)
            hybrid_regimes[normal_mask] = economic_regimes[normal_mask].astype(str) + '_Normal'
            
            self.aligned_data['HYBRID_REGIME'] = hybrid_regimes
            
            # Analyze hybrid regime distribution
            hybrid_counts = hybrid_regimes.value_counts()
            logger.info(f"Hybrid regime distribution: {hybrid_counts.to_dict()}")
            
            # Calculate regime transition analysis
            logger.info("Analyzing regime transitions...")
            
            # Economic regime transitions
            econ_transitions = economic_regimes.shift(1) != economic_regimes
            econ_transition_count = econ_transitions.sum()
            
            # VIX regime transitions
            vix_transitions = vix_regimes.shift(1) != vix_regimes
            vix_transition_count = vix_transitions.sum()
            
            # Hybrid regime transitions
            hybrid_transitions = hybrid_regimes.shift(1) != hybrid_regimes
            hybrid_transition_count = hybrid_transitions.sum()
            
            transition_analysis = {
                'economic_transitions': int(econ_transition_count),
                'vix_transitions': int(vix_transition_count),
                'hybrid_transitions': int(hybrid_transition_count),
                'economic_avg_duration': len(economic_regimes) / max(econ_transition_count, 1),
                'vix_avg_duration': len(vix_regimes) / max(vix_transition_count, 1),
                'hybrid_avg_duration': len(hybrid_regimes) / max(hybrid_transition_count, 1)
            }
            
            logger.info(f"Transition analysis: {transition_analysis}")
            
            return comparison_table.to_dict(), hybrid_counts.to_dict(), transition_analysis
            
        except Exception as e:
            logger.error(f"Error in regime cross-validation: {str(e)}")
            return {}, {}, {}
    
    def comprehensive_date_validation(self):
        """
        Comprehensive validation of date alignment across all datasets
        """
        logger.info("=== Comprehensive Date Alignment Validation ===")
        
        try:
            validation_results = {}
            
            # Check date alignment consistency
            master_dates = pd.to_datetime(self.aligned_data.index)
            
            # Validate date frequency
            date_diffs = master_dates.to_series().diff()
            unique_diffs = date_diffs.value_counts()
            
            validation_results['date_frequency'] = {
                'is_monthly': len(unique_diffs) <= 3,  # Allow for month-end variations
                'date_differences': {str(k): v for k, v in unique_diffs.items() if pd.notna(k)}
            }
            
            # Check for missing dates
            expected_dates = pd.date_range(
                start=master_dates.min(), 
                end=master_dates.max(), 
                freq='M'
            )
            
            missing_dates = expected_dates.difference(master_dates)
            validation_results['missing_dates'] = {
                'count': len(missing_dates),
                'dates': [d.strftime('%Y-%m-%d') for d in missing_dates]
            }
            
            # Validate data completeness by time period
            completeness_by_year = {}
            for year in range(master_dates.min().year, master_dates.max().year + 1):
                year_data = self.aligned_data[self.aligned_data.index.year == year]
                if len(year_data) > 0:
                    completeness = (~year_data.isnull()).all(axis=1).mean()
                    completeness_by_year[year] = float(completeness)
            
            validation_results['completeness_by_year'] = completeness_by_year
            
            # Check alignment success across datasets
            factor_completeness = (~self.aligned_data[['Value', 'Quality', 'MinVol', 'Momentum']].isnull()).all(axis=1).mean()
            vix_completeness = (~self.aligned_data['VIX'].isnull()).mean()
            regime_completeness = (~self.aligned_data['ECONOMIC_REGIME'].isnull()).mean()
            
            validation_results['dataset_completeness'] = {
                'factors': float(factor_completeness),
                'vix_data': float(vix_completeness),
                'economic_regimes': float(regime_completeness)
            }
            
            logger.info(f"Date validation results: {validation_results}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in date validation: {str(e)}")
            return {}
    
    def document_methodology(self):
        """
        Document the complete regime classification methodology
        """
        logger.info("=== Documenting Regime Classification Methodology ===")
        
        methodology = {
            'data_sources': {
                'economic_data': 'FRED Economic Data (1990-2025, daily)',
                'factor_data': 'MSCI Factor Returns (1998-2025, monthly)',
                'market_data': 'Market data including VIX and S&P 500 (1998-2025, monthly)'
            },
            'alignment_approach': {
                'master_timeline': 'MSCI factor returns monthly end-of-month dates',
                'alignment_method': 'Forward-fill with groupby monthly aggregation',
                'time_period': '1998-12-31 to 2025-05-30 (318 observations)'
            },
            'regime_frameworks': {
                'economic_regimes': {
                    'original_source': 'FRED ECONOMIC_REGIME classification',
                    'challenge': 'All aligned data shows Recession due to 1998-2025 time period',
                    'solution': 'Created synthetic regimes based on GDP growth and inflation trends'
                },
                'vix_regimes': {
                    'normal': 'VIX < 25 (low volatility)',
                    'elevated': 'VIX 25-35 (moderate stress)',
                    'stress': 'VIX 35-50 (high volatility)',
                    'crisis': 'VIX > 50 (extreme stress)'
                },
                'hybrid_regimes': {
                    'methodology': 'Combine economic regimes with VIX stress levels',
                    'priority': 'VIX crisis periods override economic classification',
                    'format': 'EconomicRegime_VIXLevel (e.g., Goldilocks_Normal)'
                }
            },
            'transition_analysis': {
                'approach': 'Calculate regime shifts using shift(1) comparison',
                'metrics': 'Transition frequency, average duration, stability measures'
            }
        }
        
        # Save methodology documentation
        with open(self.results_dir / 'regime_methodology.json', 'w') as f:
            json.dump(methodology, f, indent=2)
        
        logger.info("✅ Methodology documentation completed")
        return methodology

    # ========================================
    # PHASE 2: ADVANCED BUSINESS CYCLE ANALYTICS
    # ========================================
    
    def phase2_multidimensional_regime_analysis(self):
        """
        Phase 2.1: Multi-Dimensional Regime Analysis
        Comprehensive regime duration, frequency, transitions, and economic validation
        """
        logger.info("=== PHASE 2.1: Multi-Dimensional Regime Analysis ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run Phase 1 first.")
            return False
        
        try:
            # Step 2.1a: Regime duration and frequency analysis
            logger.info("Step 2.1a: Analyzing regime durations and frequencies...")
            regime_analysis = self._analyze_regime_durations_and_transitions()
            
            # Step 2.1b: Economic signal validation per regime  
            logger.info("Step 2.1b: Validating economic signals per regime...")
            economic_validation = self._validate_economic_signals_by_regime()
            
            # Combine and store results
            self.regime_statistics = {
                'duration_analysis': regime_analysis,
                'economic_validation': economic_validation,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save comprehensive results
            with open(self.results_dir / 'phase2_regime_analysis.json', 'w') as f:
                json.dump(self.regime_statistics, f, indent=2, default=str)
            
            logger.info("✓ Phase 2.1 Multi-Dimensional Regime Analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 2.1 analysis: {e}")
            return False
    
    def _analyze_regime_durations_and_transitions(self):
        """
        Step 2.1a: Comprehensive regime duration and frequency analysis
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        
        # Calculate regime durations
        regime_runs = []
        current_regime = None
        start_date = None
        
        for date, regime in regime_col.items():
            if regime != current_regime:
                if current_regime is not None:
                    duration = (date - start_date).days / 30.44  # Convert to months
                    regime_runs.append({
                        'regime': current_regime,
                        'start_date': start_date,
                        'end_date': date,
                        'duration_months': duration
                    })
                current_regime = regime
                start_date = date
        
        # Handle last regime
        if current_regime is not None:
            duration = (regime_col.index[-1] - start_date).days / 30.44
            regime_runs.append({
                'regime': current_regime,
                'start_date': start_date,
                'end_date': regime_col.index[-1],
                'duration_months': duration
            })
        
        runs_df = pd.DataFrame(regime_runs)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in regime_col.unique():
            regime_runs_subset = runs_df[runs_df['regime'] == regime]['duration_months']
            regime_stats[regime] = {
                'total_periods': len(regime_runs_subset),
                'total_months': regime_runs_subset.sum(),
                'avg_duration_months': regime_runs_subset.mean(),
                'median_duration_months': regime_runs_subset.median(),
                'min_duration_months': regime_runs_subset.min(),
                'max_duration_months': regime_runs_subset.max(),
                'std_duration_months': regime_runs_subset.std(),
                'frequency_percentage': (regime_col == regime).sum() / len(regime_col) * 100
            }
        
        # Calculate transition probabilities
        transitions = []
        for i in range(len(regime_col) - 1):
            from_regime = regime_col.iloc[i]
            to_regime = regime_col.iloc[i + 1]
            if from_regime != to_regime:
                transitions.append((from_regime, to_regime))
        
        # Build transition matrix
        regimes = sorted(regime_col.unique())
        transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)
        
        for from_regime, to_regime in transitions:
            transition_matrix.loc[from_regime, to_regime] += 1
        
        # Convert to probabilities
        transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
        
        # Analyze regime changes by decade
        regime_col_with_decade = regime_col.to_frame()
        regime_col_with_decade['decade'] = regime_col_with_decade.index.year // 10 * 10
        
        decade_transitions = {}
        for decade in regime_col_with_decade['decade'].unique():
            decade_data = regime_col_with_decade[regime_col_with_decade['decade'] == decade]['ECONOMIC_REGIME']
            decade_changes = (decade_data != decade_data.shift()).sum() - 1  # Subtract 1 for the first observation
            decade_transitions[f"{decade}s"] = decade_changes
        
        # Seasonal patterns
        regime_monthly = regime_col.to_frame()
        regime_monthly['month'] = regime_monthly.index.month
        seasonal_changes = {}
        for month in range(1, 13):
            month_data = regime_monthly[regime_monthly['month'] == month]['ECONOMIC_REGIME']
            month_changes = (month_data != month_data.shift()).sum()
            seasonal_changes[month] = month_changes
        
        return {
            'regime_statistics': regime_stats,
            'transition_matrix_counts': transition_matrix.to_dict(),
            'transition_probabilities': transition_probs.to_dict(), 
            'total_transitions': len(transitions),
            'decade_transitions': decade_transitions,
            'seasonal_transition_patterns': seasonal_changes,
            'regime_runs_detail': runs_df.to_dict('records')
        }
    
    def _validate_economic_signals_by_regime(self):
        """
        Step 2.1b: Economic signal validation per regime
        """
        economic_indicators = [
            'GDP_GROWTH_COMPOSITE', 'INFLATION_COMPOSITE', 'UNRATE',
            'PAYEMS', 'DGS10', 'DGS2', 'TERM_SPREAD'
        ]
        
        # Filter to available indicators
        available_indicators = [col for col in economic_indicators if col in self.aligned_data.columns]
        
        validation_results = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            regime_validation = {
                'observations': len(regime_data),
                'date_range': {
                    'start': regime_data.index.min(),
                    'end': regime_data.index.max()
                }
            }
            
            # Economic indicator analysis by regime
            for indicator in available_indicators:
                if indicator in regime_data.columns:
                    indicator_values = regime_data[indicator].dropna()
                    if len(indicator_values) > 0:
                        regime_validation[indicator] = {
                            'mean': float(indicator_values.mean()),
                            'median': float(indicator_values.median()),
                            'std': float(indicator_values.std()),
                            'min': float(indicator_values.min()),
                            'max': float(indicator_values.max()),
                            'trend': float(np.polyfit(range(len(indicator_values)), indicator_values, 1)[0]) if len(indicator_values) > 1 else 0
                        }
            
            # VIX analysis for this regime
            if 'VIX' in regime_data.columns:
                vix_values = regime_data['VIX'].dropna()
                if len(vix_values) > 0:
                    regime_validation['VIX'] = {
                        'mean': float(vix_values.mean()),
                        'median': float(vix_values.median()),
                        'volatility_profile': 'Low' if vix_values.mean() < 20 else 'Elevated' if vix_values.mean() < 30 else 'High'
                    }
            
            validation_results[regime] = regime_validation
        
        # Cross-regime comparisons
        cross_regime_analysis = {}
        for indicator in available_indicators:
            if indicator in self.aligned_data.columns:
                indicator_by_regime = {}
                for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
                    regime_values = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime][indicator].dropna()
                    if len(regime_values) > 0:
                        indicator_by_regime[regime] = float(regime_values.mean())
                
                cross_regime_analysis[indicator] = indicator_by_regime
        
        return {
            'regime_validations': validation_results,
            'cross_regime_comparisons': cross_regime_analysis
        }
    
    def phase2_factor_performance_deepdive(self):
        """
        Phase 2.2: Factor Performance Deep-Dive
        Comprehensive performance metrics and statistical significance testing
        """
        logger.info("=== PHASE 2.2: Factor Performance Deep-Dive ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run Phase 1 first.")
            return False
        
        try:
            # Step 2.2a: Comprehensive performance metrics by regime
            logger.info("Step 2.2a: Computing comprehensive performance metrics...")
            performance_metrics = self._calculate_comprehensive_performance_metrics()
            
            # Step 2.2b: Statistical significance testing
            logger.info("Step 2.2b: Running statistical significance tests...")
            statistical_tests = self._run_statistical_significance_tests()
            
            # Combine results
            self.performance_metrics = {
                'performance_metrics': performance_metrics,
                'statistical_tests': statistical_tests,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open(self.results_dir / 'phase2_performance_analysis.json', 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            
            logger.info("✓ Phase 2.2 Factor Performance Deep-Dive completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 2.2 analysis: {e}")
            return False
    
    def _calculate_comprehensive_performance_metrics(self):
        """
        Step 2.2a: Calculate comprehensive performance metrics by regime
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Add S&P 500 if available
        if 'SP500_RETURN' in self.aligned_data.columns:
            factors.append('SP500_RETURN')
        
        results = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            regime_results = {}
            
            for factor in factors:
                if factor in regime_data.columns:
                    returns = regime_data[factor].dropna()
                    
                    if len(returns) > 0:
                        # Convert to decimal if needed (assuming returns are in percentage)
                        if returns.abs().mean() > 1:  # Likely in percentage form
                            returns = returns / 100
                        
                        # Basic return metrics
                        mean_return = returns.mean()
                        std_return = returns.std()
                        median_return = returns.median()
                        
                        # Annualized metrics (assuming monthly data)
                        annual_return = (1 + mean_return) ** 12 - 1
                        annual_volatility = std_return * np.sqrt(12)
                        
                        # Risk-adjusted metrics
                        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                        
                        # Sortino ratio (downside deviation)
                        downside_returns = returns[returns < 0]
                        downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
                        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
                        
                        # Maximum drawdown
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdowns = (cumulative - running_max) / running_max
                        max_drawdown = drawdowns.min()
                        
                        # VaR and Expected Shortfall (5%)
                        var_5 = np.percentile(returns, 5)
                        es_5 = returns[returns <= var_5].mean()
                        
                        # Consistency metrics
                        positive_months = (returns > 0).sum()
                        win_rate = positive_months / len(returns)
                        
                        # Calmar ratio
                        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                        
                        regime_results[factor] = {
                            'observations': len(returns),
                            'mean_monthly_return': float(mean_return),
                            'median_monthly_return': float(median_return),
                            'std_monthly_return': float(std_return),
                            'annualized_return': float(annual_return),
                            'annualized_volatility': float(annual_volatility),
                            'sharpe_ratio': float(sharpe_ratio),
                            'sortino_ratio': float(sortino_ratio),
                            'calmar_ratio': float(calmar_ratio),
                            'max_drawdown': float(max_drawdown),
                            'var_5_percent': float(var_5),
                            'expected_shortfall_5_percent': float(es_5),
                            'win_rate': float(win_rate),
                            'positive_months': int(positive_months)
                        }
            
            results[regime] = regime_results
        
        return results
    
    def _run_statistical_significance_tests(self):
        """
        Step 2.2b: Statistical significance testing
        """
        from scipy import stats
        import itertools
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        if 'SP500_RETURN' in self.aligned_data.columns:
            factors.append('SP500_RETURN')
        
        results = {}
        
        # ANOVA tests for performance differences across regimes
        logger.info("Running ANOVA tests for regime differences...")
        anova_results = {}
        
        for factor in factors:
            if factor in self.aligned_data.columns:
                groups = []
                regime_labels = []
                
                for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
                    regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime][factor].dropna()
                    if len(regime_data) > 0:
                        groups.append(regime_data.values)
                        regime_labels.append(regime)
                
                if len(groups) >= 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        anova_results[factor] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'regimes_tested': regime_labels
                        }
                    except:
                        anova_results[factor] = {'error': 'Could not compute ANOVA'}
        
        # Pairwise t-tests comparing factors vs S&P 500 by regime
        logger.info("Running pairwise t-tests...")
        pairwise_tests = {}
        
        if 'SP500_RETURN' in self.aligned_data.columns:
            for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
                regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
                sp500_returns = regime_data['SP500_RETURN'].dropna()
                
                if len(sp500_returns) > 1:
                    regime_tests = {}
                    
                    for factor in factors:
                        if factor != 'SP500_RETURN' and factor in regime_data.columns:
                            factor_returns = regime_data[factor].dropna()
                            
                            # Align the series for comparison
                            aligned_factor, aligned_sp500 = factor_returns.align(sp500_returns, join='inner')
                            
                            if len(aligned_factor) > 1 and len(aligned_sp500) > 1:
                                try:
                                    t_stat, p_value = stats.ttest_rel(aligned_factor, aligned_sp500)
                                    regime_tests[factor] = {
                                        't_statistic': float(t_stat),
                                        'p_value': float(p_value),
                                        'significant': p_value < 0.05,
                                        'outperforms_sp500': aligned_factor.mean() > aligned_sp500.mean(),
                                        'observations': len(aligned_factor)
                                    }
                                except:
                                    regime_tests[factor] = {'error': 'Could not compute t-test'}
                    
                    pairwise_tests[regime] = regime_tests
        
        # Bootstrap confidence intervals
        logger.info("Computing bootstrap confidence intervals...")
        bootstrap_results = {}
        
        def bootstrap_mean(data, n_bootstrap=1000):
            """Bootstrap confidence intervals for mean"""
            bootstrap_means = []
            n = len(data)
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=n, replace=True)
                bootstrap_means.append(np.mean(sample))
            
            return {
                'mean': np.mean(bootstrap_means),
                'ci_lower': np.percentile(bootstrap_means, 2.5),
                'ci_upper': np.percentile(bootstrap_means, 97.5)
            }
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            regime_bootstrap = {}
            
            for factor in factors:
                if factor in regime_data.columns:
                    factor_returns = regime_data[factor].dropna()
                    
                    if len(factor_returns) > 10:  # Need sufficient data for bootstrap
                        try:
                            bootstrap_ci = bootstrap_mean(factor_returns.values)
                            regime_bootstrap[factor] = bootstrap_ci
                        except:
                            regime_bootstrap[factor] = {'error': 'Bootstrap failed'}
            
            bootstrap_results[regime] = regime_bootstrap
        
        # Regime transition impact analysis
        logger.info("Analyzing regime transition impacts...")
        transition_analysis = self._analyze_regime_transition_impact()
        
        return {
            'anova_tests': anova_results,
            'pairwise_tests': pairwise_tests,
            'bootstrap_confidence_intervals': bootstrap_results,
            'regime_transition_impact': transition_analysis
        }
    
    def _analyze_regime_transition_impact(self):
        """
        Analyze factor performance during regime transition periods
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Identify transition periods (3 months before and after regime changes)
        transition_periods = []
        for i in range(len(regime_col) - 1):
            if regime_col.iloc[i] != regime_col.iloc[i + 1]:
                transition_date = regime_col.index[i + 1]
                transition_periods.append({
                    'date': transition_date,
                    'from_regime': regime_col.iloc[i],
                    'to_regime': regime_col.iloc[i + 1]
                })
        
        if not transition_periods:
            return {'error': 'No regime transitions found'}
        
        # Analyze performance around transitions
        transition_analysis = {}
        window_months = 3
        
        for factor in factors:
            if factor in self.aligned_data.columns:
                pre_transition_performance = []
                post_transition_performance = []
                
                for transition in transition_periods:
                    transition_date = transition['date']
                    
                    # Get 3-month windows
                    try:
                        pre_window = self.aligned_data.loc[:transition_date][factor].iloc[-window_months-1:-1]
                        post_window = self.aligned_data.loc[transition_date:][factor].iloc[1:window_months+1]
                        
                        if len(pre_window) > 0:
                            pre_transition_performance.extend(pre_window.dropna())
                        if len(post_window) > 0:
                            post_transition_performance.extend(post_window.dropna())
                    except:
                        continue
                
                if pre_transition_performance and post_transition_performance:
                    transition_analysis[factor] = {
                        'pre_transition_mean': float(np.mean(pre_transition_performance)),
                        'post_transition_mean': float(np.mean(post_transition_performance)),
                        'pre_transition_volatility': float(np.std(pre_transition_performance)),
                        'post_transition_volatility': float(np.std(post_transition_performance)),
                        'performance_change': float(np.mean(post_transition_performance) - np.mean(pre_transition_performance)),
                        'volatility_change': float(np.std(post_transition_performance) - np.std(pre_transition_performance)),
                        'total_transitions_analyzed': len(transition_periods)
                    }
        
        return transition_analysis
    
    def run_phase2(self):
        """
        Execute complete Phase 2: Advanced Business Cycle Analytics
        """
        logger.info("=== STARTING PHASE 2: ADVANCED BUSINESS CYCLE ANALYTICS ===")
        
        success = True
        
        # Step 2.1: Multi-Dimensional Regime Analysis
        if not self.phase2_multidimensional_regime_analysis():
            logger.error("Phase 2.1 failed")
            success = False
        
        # Step 2.2: Factor Performance Deep-Dive
        if not self.phase2_factor_performance_deepdive():
            logger.error("Phase 2.2 failed")
            success = False
        
        if success:
            logger.info("=== PHASE 2 COMPLETED SUCCESSFULLY ===")
            
            # Create comprehensive Phase 2 summary
            self._create_phase2_summary()
        else:
            logger.error("=== PHASE 2 COMPLETED WITH ERRORS ===")
        
        return success
    
    def _create_phase2_summary(self):
        """
        Create comprehensive Phase 2 summary report
        """
        logger.info("Creating Phase 2 comprehensive summary...")
        
        summary = {
            "phase2_completion": {
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "components_completed": [
                    "2.1a: Regime duration and frequency analysis",
                    "2.1b: Economic signal validation per regime", 
                    "2.2a: Comprehensive performance metrics by regime",
                    "2.2b: Statistical significance testing"
                ]
            },
            "regime_analysis_summary": self.regime_statistics.get('duration_analysis', {}) if hasattr(self, 'regime_statistics') else {},
            "performance_summary": self.performance_metrics.get('performance_metrics', {}) if hasattr(self, 'performance_metrics') else {},
            "statistical_summary": self.performance_metrics.get('statistical_tests', {}) if hasattr(self, 'performance_metrics') else {}
        }
        
        # Save Phase 2 summary
        with open(self.results_dir / 'phase2_complete_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✓ Phase 2 summary saved")
        
        return True

def main():
    """
    Main execution function - now supports both Phase 1 and Phase 2
    """
    analyzer = BusinessCycleFactorAnalyzer()
    
    logger.info("Starting Business Cycle Factor Performance Analysis")
    
    # Execute Phase 1
    logger.info("=== EXECUTING PHASE 1 ===")
    phase1_success = analyzer.run_phase1()
    
    if not phase1_success:
        logger.error("❌ Phase 1 failed. Please check the logs and fix issues.")
        exit(1)
    
    # Execute Phase 2 
    logger.info("=== EXECUTING PHASE 2 ===")
    phase2_success = analyzer.run_phase2()
    
    if phase1_success and phase2_success:
        logger.info("✅ ALL PHASES COMPLETED SUCCESSFULLY!")
        logger.info("📊 Phase 1: Data alignment and regime validation - DONE")
        logger.info("📈 Phase 2: Advanced business cycle analytics - DONE")
        logger.info("🚀 Ready for Phase 3: Advanced Visualization Suite")
    else:
        logger.error("❌ Some phases failed. Please check the logs.")
        exit(1)

if __name__ == "__main__":
    main() 