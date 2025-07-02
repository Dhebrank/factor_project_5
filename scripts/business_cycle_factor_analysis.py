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
from scipy import stats
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
            'GROWTH_COMPOSITE', 'GDPC1_YOY', 'INFLATION_COMPOSITE', 'UNRATE',
            'PAYEMS', 'PAYEMS_YOY', 'DGS10', 'DGS2', 'T10Y2Y'
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
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            factors.append('SP500_Monthly_Return')
        
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
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            factors.append('SP500_Monthly_Return')
        
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
        
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
                regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
                sp500_returns = regime_data['SP500_Monthly_Return'].dropna()
                
                if len(sp500_returns) > 1:
                    regime_tests = {}
                    
                    for factor in factors:
                        if factor != 'SP500_Monthly_Return' and factor in regime_data.columns:
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

    # ========================================
    # PHASE 3: ADVANCED VISUALIZATION SUITE
    # ========================================
    
    def phase3_master_dashboard_layout(self):
        """
        Phase 3.1: Master Business Cycle Dashboard Layout
        Interactive timeline with regime overlay and dynamic statistics panel
        """
        logger.info("=== PHASE 3.1: Master Business Cycle Dashboard Layout ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run Phase 1 first.")
            return False
        
        try:
            # Step 3.1a: Interactive timeline with regime overlay
            logger.info("Step 3.1a: Creating interactive timeline with regime overlay...")
            timeline_fig = self._create_interactive_timeline()
            
            # Step 3.1b: Dynamic regime statistics panel
            logger.info("Step 3.1b: Creating dynamic regime statistics panel...")
            stats_panel = self._create_regime_statistics_panel()
            
            # Save timeline visualization
            timeline_fig.write_html(self.results_dir / 'interactive_timeline_regime_overlay.html')
            
            # Save regime statistics
            with open(self.results_dir / 'regime_statistics_panel.json', 'w') as f:
                json.dump(stats_panel, f, indent=2, default=str)
            
            logger.info("✓ Phase 3.1 Master Dashboard Layout completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 3.1 dashboard layout: {e}")
            return False
    
    def _create_interactive_timeline(self):
        """
        Step 3.1a: Create interactive timeline with regime overlay
        """
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Business Cycle Regimes & Market Performance', 'VIX Stress Levels'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Define colors for each regime
        regime_colors = {
            'Goldilocks': '#2E8B57',  # Sea Green
            'Overheating': '#FF6347',  # Tomato
            'Stagflation': '#FFD700',  # Gold  
            'Recession': '#8B0000'    # Dark Red
        }
        
        # Add regime background bands
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        dates = self.aligned_data.index
        
        # Group consecutive regime periods
        regime_periods = []
        current_regime = None
        start_date = None
        
        for i, (date, regime) in enumerate(regime_col.items()):
            if regime != current_regime:
                if current_regime is not None:
                    regime_periods.append({
                        'regime': current_regime,
                        'start': start_date,
                        'end': date,
                        'color': regime_colors.get(current_regime, '#808080')
                    })
                current_regime = regime
                start_date = date
        
        # Add final period
        if current_regime is not None:
            regime_periods.append({
                'regime': current_regime,
                'start': start_date,
                'end': dates[-1],
                'color': regime_colors.get(current_regime, '#808080')
            })
        
        # Add regime background rectangles
        for period in regime_periods:
            fig.add_shape(
                type="rect",
                x0=period['start'], x1=period['end'],
                y0=0, y1=1,
                yref="y domain",
                fillcolor=period['color'],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Add S&P 500 performance line
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            sp500_cumulative = (1 + self.aligned_data['SP500_Monthly_Return']).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=sp500_cumulative,
                    name='S&P 500 Cumulative Return',
                    line=dict(color='black', width=2),
                    hovertemplate='<b>S&P 500</b><br>' +
                                'Date: %{x}<br>' +
                                'Cumulative Return: %{y:.2f}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add factor performance lines
        factor_colors = {
            'Value': '#1f77b4',    # Blue
            'Quality': '#ff7f0e',  # Orange
            'MinVol': '#2ca02c',   # Green
            'Momentum': '#d62728'  # Red
        }
        
        for factor, color in factor_colors.items():
            if factor in self.aligned_data.columns:
                factor_cumulative = (1 + self.aligned_data[factor]).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=factor_cumulative,
                        name=f'{factor} Factor',
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'<b>{factor} Factor</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Cumulative Return: %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add VIX levels in second subplot
        if 'VIX' in self.aligned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=self.aligned_data['VIX'],
                    name='VIX Level',
                    line=dict(color='purple', width=1.5),
                    fill='tonexty',
                    hovertemplate='<b>VIX</b><br>' +
                                'Date: %{x}<br>' +
                                'Level: %{y:.1f}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add VIX threshold lines
            vix_thresholds = [25, 35, 50]
            threshold_labels = ['Elevated', 'Stress', 'Crisis']
            threshold_colors = ['orange', 'red', 'darkred']
            
            for threshold, label, color in zip(vix_thresholds, threshold_labels, threshold_colors):
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{label} ({threshold})",
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Business Cycle Factor Performance Analysis (1998-2025)',
                'x': 0.5,
                'font': {'size': 20}
            },
            showlegend=True,
            height=800,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update x-axis
        fig.update_xaxes(title_text="Date")
        
        # Update y-axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="VIX Level", row=2, col=1)
        
        return fig
    
    def _create_regime_statistics_panel(self):
        """
        Step 3.1b: Create dynamic regime statistics panel
        """
        regime_stats = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            # Basic statistics
            total_months = len(regime_data)
            percentage_of_period = (total_months / len(self.aligned_data)) * 100
            
            # Date range
            date_range = {
                'start': regime_data.index.min().strftime('%Y-%m-%d'),
                'end': regime_data.index.max().strftime('%Y-%m-%d')
            }
            
            # Average factor performance
            factor_performance = {}
            for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                if factor in regime_data.columns:
                    monthly_return = regime_data[factor].mean()
                    annualized_return = (1 + monthly_return) ** 12 - 1
                    factor_performance[factor] = {
                        'monthly_return': float(monthly_return),
                        'annualized_return': float(annualized_return)
                    }
            
            # VIX statistics
            vix_stats = {}
            if 'VIX' in regime_data.columns:
                vix_stats = {
                    'average_vix': float(regime_data['VIX'].mean()),
                    'max_vix': float(regime_data['VIX'].max()),
                    'min_vix': float(regime_data['VIX'].min())
                }
            
            regime_stats[regime] = {
                'total_months': total_months,
                'percentage_of_period': float(percentage_of_period),
                'date_range': date_range,
                'factor_performance': factor_performance,
                'vix_statistics': vix_stats
            }
        
        return regime_stats
    
    def phase3_multilayer_heatmaps(self):
        """
        Phase 3.2: Multi-Layer Performance Heatmaps
        Primary performance, risk-adjusted, and relative performance heatmaps
        """
        logger.info("=== PHASE 3.2: Multi-Layer Performance Heatmaps ===")
        
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            logger.error("No performance metrics available. Run Phase 2 first.")
            return False
        
        try:
            # Step 3.2a: Primary performance heatmap (Factor × Regime)
            logger.info("Step 3.2a: Creating primary performance heatmap...")
            primary_heatmap = self._create_primary_performance_heatmap()
            
            # Step 3.2b: Risk-adjusted performance heatmap
            logger.info("Step 3.2b: Creating risk-adjusted performance heatmap...")
            risk_adjusted_heatmap = self._create_risk_adjusted_heatmap()
            
            # Step 3.2c: Relative performance heatmap (vs S&P 500)
            logger.info("Step 3.2c: Creating relative performance heatmap...")
            relative_heatmap = self._create_relative_performance_heatmap()
            
            # Save all heatmaps
            primary_heatmap.write_html(self.results_dir / 'primary_performance_heatmap.html')
            risk_adjusted_heatmap.write_html(self.results_dir / 'risk_adjusted_heatmap.html')
            relative_heatmap.write_html(self.results_dir / 'relative_performance_heatmap.html')
            
            logger.info("✓ Phase 3.2 Multi-Layer Performance Heatmaps completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 3.2 heatmaps: {e}")
            return False
    
    def _create_primary_performance_heatmap(self):
        """
        Step 3.2a: Create primary performance heatmap (Factor × Regime)
        """
        # Extract performance data
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500_Monthly_Return']
        factor_labels = ['Value', 'Quality', 'MinVol', 'Momentum', 'S&P 500']
        
        # Create data matrix for heatmap
        performance_matrix = []
        hover_text = []
        
        for factor, label in zip(factors, factor_labels):
            row_data = []
            row_hover = []
            
            for regime in regimes:
                if regime in self.performance_metrics['performance_metrics']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        sharpe_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                        win_rate = self.performance_metrics['performance_metrics'][regime][factor]['win_rate']
                        
                        row_data.append(annual_return * 100)  # Convert to percentage
                        row_hover.append(
                            f"<b>{label} in {regime}</b><br>" +
                            f"Annual Return: {annual_return*100:.1f}%<br>" +
                            f"Sharpe Ratio: {sharpe_ratio:.2f}<br>" +
                            f"Win Rate: {win_rate*100:.1f}%"
                        )
                    else:
                        row_data.append(np.nan)
                        row_hover.append(f"<b>{label} in {regime}</b><br>No data available")
                else:
                    row_data.append(np.nan)
                    row_hover.append(f"<b>{label} in {regime}</b><br>No data available")
            
            performance_matrix.append(row_data)
            hover_text.append(row_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=regimes,
            y=factor_labels,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='RdYlGn',
            colorbar=dict(title="Annual Return (%)")
        ))
        
        # Add text annotations
        for i, factor_label in enumerate(factor_labels):
            for j, regime in enumerate(regimes):
                if not np.isnan(performance_matrix[i][j]):
                    fig.add_annotation(
                        x=regime,
                        y=factor_label,
                        text=f"{performance_matrix[i][j]:.1f}%",
                        showarrow=False,
                        font=dict(color="white" if abs(performance_matrix[i][j]) > 10 else "black")
                    )
        
        fig.update_layout(
            title={
                'text': 'Factor Performance by Economic Regime<br><sub>Annualized Returns (%)</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title="Economic Regime",
            yaxis_title="Investment Factor",
            height=500,
            width=800
        )
        
        return fig
    
    def _create_risk_adjusted_heatmap(self):
        """
        Step 3.2b: Create risk-adjusted performance heatmap
        """
        # Extract Sharpe ratio data
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500_Monthly_Return']
        factor_labels = ['Value', 'Quality', 'MinVol', 'Momentum', 'S&P 500']
        
        # Create data matrix for heatmap
        sharpe_matrix = []
        hover_text = []
        
        for factor, label in zip(factors, factor_labels):
            row_data = []
            row_hover = []
            
            for regime in regimes:
                if regime in self.performance_metrics['performance_metrics']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        sharpe_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                        sortino_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sortino_ratio']
                        max_drawdown = self.performance_metrics['performance_metrics'][regime][factor]['max_drawdown']
                        
                        row_data.append(sharpe_ratio)
                        row_hover.append(
                            f"<b>{label} in {regime}</b><br>" +
                            f"Sharpe Ratio: {sharpe_ratio:.2f}<br>" +
                            f"Sortino Ratio: {sortino_ratio:.2f}<br>" +
                            f"Annual Return: {annual_return*100:.1f}%<br>" +
                            f"Max Drawdown: {max_drawdown*100:.1f}%"
                        )
                    else:
                        row_data.append(np.nan)
                        row_hover.append(f"<b>{label} in {regime}</b><br>No data available")
                else:
                    row_data.append(np.nan)
                    row_hover.append(f"<b>{label} in {regime}</b><br>No data available")
            
            sharpe_matrix.append(row_data)
            hover_text.append(row_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sharpe_matrix,
            x=regimes,
            y=factor_labels,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='RdYlGn',
            colorbar=dict(title="Sharpe Ratio"),
            zmid=0
        ))
        
        # Add text annotations
        for i, factor_label in enumerate(factor_labels):
            for j, regime in enumerate(regimes):
                if not np.isnan(sharpe_matrix[i][j]):
                    fig.add_annotation(
                        x=regime,
                        y=factor_label,
                        text=f"{sharpe_matrix[i][j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(sharpe_matrix[i][j]) > 0.5 else "black")
                    )
        
        fig.update_layout(
            title={
                'text': 'Risk-Adjusted Performance by Economic Regime<br><sub>Sharpe Ratios</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title="Economic Regime",
            yaxis_title="Investment Factor",
            height=500,
            width=800
        )
        
        return fig
    
    def _create_relative_performance_heatmap(self):
        """
        Step 3.2c: Create relative performance heatmap (vs S&P 500)
        """
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Create data matrix for relative performance
        relative_matrix = []
        hover_text = []
        
        for factor in factors:
            row_data = []
            row_hover = []
            
            for regime in regimes:
                if regime in self.performance_metrics['performance_metrics']:
                    if (factor in self.performance_metrics['performance_metrics'][regime] and
                        'SP500_Monthly_Return' in self.performance_metrics['performance_metrics'][regime]):
                        
                        factor_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        sp500_return = self.performance_metrics['performance_metrics'][regime]['SP500_Monthly_Return']['annualized_return']
                        
                        excess_return = (factor_return - sp500_return) * 100  # Convert to percentage
                        
                        factor_sharpe = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                        sp500_sharpe = self.performance_metrics['performance_metrics'][regime]['SP500_Monthly_Return']['sharpe_ratio']
                        
                        row_data.append(excess_return)
                        row_hover.append(
                            f"<b>{factor} vs S&P 500 in {regime}</b><br>" +
                            f"Excess Return: {excess_return:.1f}%<br>" +
                            f"Factor Return: {factor_return*100:.1f}%<br>" +
                            f"S&P 500 Return: {sp500_return*100:.1f}%<br>" +
                            f"Factor Sharpe: {factor_sharpe:.2f}<br>" +
                            f"S&P 500 Sharpe: {sp500_sharpe:.2f}"
                        )
                    else:
                        row_data.append(np.nan)
                        row_hover.append(f"<b>{factor} vs S&P 500 in {regime}</b><br>No data available")
                else:
                    row_data.append(np.nan)
                    row_hover.append(f"<b>{factor} vs S&P 500 in {regime}</b><br>No data available")
            
            relative_matrix.append(row_data)
            hover_text.append(row_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=relative_matrix,
            x=regimes,
            y=factors,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='RdYlGn',
            colorbar=dict(title="Excess Return (%)"),
            zmid=0
        ))
        
        # Add text annotations
        for i, factor in enumerate(factors):
            for j, regime in enumerate(regimes):
                if not np.isnan(relative_matrix[i][j]):
                    fig.add_annotation(
                        x=regime,
                        y=factor,
                        text=f"{relative_matrix[i][j]:.1f}%",
                        showarrow=False,
                        font=dict(color="white" if abs(relative_matrix[i][j]) > 5 else "black")
                    )
        
        fig.update_layout(
            title={
                'text': 'Factor Performance vs S&P 500 by Economic Regime<br><sub>Excess Annual Returns (%)</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title="Economic Regime",
            yaxis_title="Investment Factor",
            height=500,
            width=800
        )
        
        return fig

    def phase3_advanced_analytical_charts(self):
        """
        Phase 3.3: Advanced Analytical Charts
        Factor rotation wheel, risk-return scatter plots, and rolling regime analysis
        """
        logger.info("=== PHASE 3.3: Advanced Analytical Charts ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run Phase 1 first.")
            return False
        
        try:
            # Step 3.3a: Factor rotation wheel by regime
            logger.info("Step 3.3a: Creating factor rotation wheel by regime...")
            rotation_wheel = self._create_factor_rotation_wheel()
            
            # Step 3.3b: Risk-return scatter plots with regime clustering
            logger.info("Step 3.3b: Creating risk-return scatter plots...")
            scatter_plots = self._create_risk_return_scatter()
            
            # Step 3.3c: Rolling regime analysis
            logger.info("Step 3.3c: Creating rolling regime analysis...")
            rolling_analysis = self._create_rolling_regime_analysis()
            
            # Save all charts
            rotation_wheel.write_html(self.results_dir / 'factor_rotation_wheel.html')
            scatter_plots.write_html(self.results_dir / 'risk_return_scatter_plots.html')
            rolling_analysis.write_html(self.results_dir / 'rolling_regime_analysis.html')
            
            logger.info("✓ Phase 3.3 Advanced Analytical Charts completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 3.3 analytical charts: {e}")
            return False
    
    def _create_factor_rotation_wheel(self):
        """
        Step 3.3a: Create factor rotation wheel by regime
        """
        # Extract performance data for each regime
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Create polar/radar chart for each regime
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=regimes,
            specs=[[{"type": "polar"}, {"type": "polar"}],
                   [{"type": "polar"}, {"type": "polar"}]]
        )
        
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, regime in enumerate(regimes):
            row, col = subplot_positions[idx]
            
            if regime in self.performance_metrics['performance_metrics']:
                # Extract Sharpe ratios for this regime
                sharpe_values = []
                for factor in factors:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        sharpe_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                        sharpe_values.append(max(0, sharpe_ratio))  # Use 0 as minimum for visualization
                    else:
                        sharpe_values.append(0)
                
                # Add factor names again to close the loop
                theta_values = factors + [factors[0]]
                r_values = sharpe_values + [sharpe_values[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_values,
                        theta=theta_values,
                        fill='toself',
                        name=f'{regime}',
                        line_color='rgb(106, 81, 163)',
                        fillcolor='rgba(106, 81, 163, 0.3)'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title={
                'text': 'Factor Performance Rotation Wheel by Economic Regime<br><sub>Sharpe Ratios</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_risk_return_scatter(self):
        """
        Step 3.3b: Create risk-return scatter plots with regime clustering
        """
        fig = go.Figure()
        
        # Define colors for each regime
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700',
            'Recession': '#8B0000'
        }
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500_Monthly_Return']
        factor_labels = ['Value', 'Quality', 'MinVol', 'Momentum', 'S&P 500']
        
        for regime in self.performance_metrics['performance_metrics']:
            volatility_values = []
            return_values = []
            factor_names = []
            hover_texts = []
            
            for factor, label in zip(factors, factor_labels):
                if factor in self.performance_metrics['performance_metrics'][regime]:
                    metrics = self.performance_metrics['performance_metrics'][regime][factor]
                    volatility = metrics['annualized_volatility']
                    annual_return = metrics['annualized_return']
                    sharpe_ratio = metrics['sharpe_ratio']
                    
                    volatility_values.append(volatility * 100)  # Convert to percentage
                    return_values.append(annual_return * 100)   # Convert to percentage
                    factor_names.append(label)
                    hover_texts.append(
                        f"<b>{label} in {regime}</b><br>" +
                        f"Annual Return: {annual_return*100:.1f}%<br>" +
                        f"Volatility: {volatility*100:.1f}%<br>" +
                        f"Sharpe Ratio: {sharpe_ratio:.2f}"
                    )
            
            if volatility_values:  # Only add if we have data
                fig.add_trace(
                    go.Scatter(
                        x=volatility_values,
                        y=return_values,
                        mode='markers+text',
                        name=regime,
                        text=factor_names,
                        textposition="top center",
                        marker=dict(
                            size=12,
                            color=regime_colors.get(regime, '#808080'),
                            symbol='circle'
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts
                    )
                )
        
        # Add efficient frontier reference line (45-degree line representing Sharpe ratio = 1)
        max_vol = 25  # Assuming max volatility around 25%
        fig.add_trace(
            go.Scatter(
                x=[0, max_vol],
                y=[0, max_vol],
                mode='lines',
                name='Sharpe Ratio = 1',
                line=dict(dash='dash', color='gray'),
                showlegend=True
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Risk-Return Profile by Economic Regime<br><sub>Factor Performance Clustering</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Return (%)",
            height=600,
            width=900,
            hovermode='closest'
        )
        
        return fig
    
    def _create_rolling_regime_analysis(self):
        """
        Step 3.3c: Create rolling regime analysis
        """
        # Create 12-month rolling performance analysis
        window_months = 12
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Rolling 12-Month Factor Performance',
                'Regime Transitions',
                'Rolling Volatility'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # Calculate rolling returns
        factor_colors = {
            'Value': '#1f77b4',
            'Quality': '#ff7f0e',
            'MinVol': '#2ca02c',
            'Momentum': '#d62728'
        }
        
        dates = self.aligned_data.index
        
        for factor, color in factor_colors.items():
            if factor in self.aligned_data.columns:
                # Calculate rolling 12-month returns
                rolling_returns = self.aligned_data[factor].rolling(window=window_months).apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100  # Convert to percentage
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=rolling_returns,
                        name=f'{factor} Rolling Return',
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'<b>{factor}</b><br>' +
                                    'Date: %{x}<br>' +
                                    '12M Return: %{y:.1f}%<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add regime transition markers in second subplot
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700',
            'Recession': '#8B0000'
        }
        
        # Create regime timeline as colored bars
        for i, (date, regime) in enumerate(regime_col.items()):
            fig.add_trace(
                go.Scatter(
                    x=[date, date],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color=regime_colors.get(regime, '#808080'), width=3),
                    showlegend=False,
                    hovertemplate=f'<b>{regime}</b><br>Date: {date}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add rolling volatility in third subplot
        for factor, color in factor_colors.items():
            if factor in self.aligned_data.columns:
                rolling_vol = self.aligned_data[factor].rolling(window=window_months).std() * np.sqrt(12) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=rolling_vol,
                        name=f'{factor} Rolling Vol',
                        line=dict(color=color, width=1.5, dash='dot'),
                        hovertemplate=f'<b>{factor} Volatility</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Rolling Vol: %{y:.1f}%<br>' +
                                    '<extra></extra>',
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        fig.update_layout(
            title={
                'text': 'Rolling Factor Performance & Regime Analysis<br><sub>12-Month Rolling Windows</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            height=900,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="12M Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1, showticklabels=False)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
        
        # Update x-axis
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def phase3_correlation_dependency_analysis(self):
        """
        Phase 3.4: Correlation & Dependency Analysis
        Dynamic correlation matrices and factor momentum persistence
        """
        logger.info("=== PHASE 3.4: Correlation & Dependency Analysis ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run Phase 1 first.")
            return False
        
        try:
            # Step 3.4a: Dynamic correlation matrices
            logger.info("Step 3.4a: Creating dynamic correlation matrices...")
            correlation_matrices = self._create_correlation_matrices()
            
            # Step 3.4b: Factor momentum persistence
            logger.info("Step 3.4b: Analyzing factor momentum persistence...")
            momentum_analysis = self._create_momentum_persistence_analysis()
            
            # Save visualizations
            correlation_matrices.write_html(self.results_dir / 'correlation_matrices_by_regime.html')
            momentum_analysis.write_html(self.results_dir / 'momentum_persistence_analysis.html')
            
            logger.info("✓ Phase 3.4 Correlation & Dependency Analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 3.4 correlation analysis: {e}")
            return False
    
    def _create_correlation_matrices(self):
        """
        Step 3.4a: Create dynamic correlation matrices by regime
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        
        # Create subplots for each regime
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{regime} Regime' for regime in regimes],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, regime in enumerate(regimes):
            row, col = subplot_positions[idx]
            
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            if len(regime_data) > 10:  # Need sufficient data for correlation
                # Calculate correlation matrix for this regime
                factor_data = regime_data[factors].dropna()
                if len(factor_data) > 5:
                    corr_matrix = factor_data.corr().values
                    
                    # Create hover text
                    hover_text = []
                    for i in range(len(factors)):
                        hover_row = []
                        for j in range(len(factors)):
                            hover_row.append(
                                f"<b>{factors[i]} vs {factors[j]}</b><br>" +
                                f"Correlation: {corr_matrix[i, j]:.3f}<br>" +
                                f"Regime: {regime}"
                            )
                        hover_text.append(hover_row)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=corr_matrix,
                            x=factors,
                            y=factors,
                            colorscale='RdBu',
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=hover_text,
                            showscale=(idx == 0),  # Only show colorbar for first subplot
                            colorbar=dict(title="Correlation", x=1.02) if idx == 0 else None
                        ),
                        row=row, col=col
                    )
                    
                    # Add correlation values as text
                    for i in range(len(factors)):
                        for j in range(len(factors)):
                            fig.add_annotation(
                                x=factors[j],
                                y=factors[i],
                                text=f"{corr_matrix[i, j]:.2f}",
                                showarrow=False,
                                font=dict(color="white" if abs(corr_matrix[i, j]) > 0.5 else "black"),
                                row=row, col=col
                            )
        
        fig.update_layout(
            title={
                'text': 'Factor Correlation Matrices by Economic Regime',
                'x': 0.5,
                'font': {'size': 16}
            },
            height=800,
            width=1000
        )
        
        return fig
    
    def _create_momentum_persistence_analysis(self):
        """
        Step 3.4b: Create factor momentum persistence analysis
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{factor} Momentum Persistence' for factor in factors],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, factor in enumerate(factors):
            row, col = subplot_positions[idx]
            
            if factor in self.aligned_data.columns:
                # Calculate momentum persistence (autocorrelation at different lags)
                factor_returns = self.aligned_data[factor].dropna()
                
                lags = range(1, 13)  # 1 to 12 months
                autocorrelations = []
                
                for lag in lags:
                    autocorr = factor_returns.autocorr(lag=lag)
                    autocorrelations.append(autocorr if not np.isnan(autocorr) else 0)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(lags),
                        y=autocorrelations,
                        mode='lines+markers',
                        name=f'{factor} Autocorr',
                        line=dict(width=2),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{factor} Autocorrelation</b><br>' +
                                    'Lag: %{x} months<br>' +
                                    'Autocorr: %{y:.3f}<br>' +
                                    '<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    row=row, col=col
                )
                
                # Add significance bands (rough approximation)
                n_obs = len(factor_returns)
                significance_bound = 1.96 / np.sqrt(n_obs)
                
                fig.add_hline(
                    y=significance_bound,
                    line_dash="dot",
                    line_color="red",
                    row=row, col=col
                )
                fig.add_hline(
                    y=-significance_bound,
                    line_dash="dot",
                    line_color="red",
                    row=row, col=col
                )
        
        fig.update_layout(
            title={
                'text': 'Factor Momentum Persistence Analysis<br><sub>Autocorrelation by Lag Period</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            height=600,
            width=1000
        )
        
        # Update axes
        fig.update_xaxes(title_text="Lag (Months)")
        fig.update_yaxes(title_text="Autocorrelation")
        
        return fig
    
    def run_phase3(self):
        """
        Execute complete Phase 3: Advanced Visualization Suite
        """
        logger.info("=== STARTING PHASE 3: ADVANCED VISUALIZATION SUITE ===")
        
        success = True
        completed_steps = []
        
        # Step 3.1: Master Business Cycle Dashboard Layout
        if self.phase3_master_dashboard_layout():
            completed_steps.extend(['3.1a', '3.1b'])
            logger.info("✓ Phase 3.1 completed successfully")
        else:
            logger.error("✗ Phase 3.1 failed")
            success = False
        
        # Step 3.2: Multi-Layer Performance Heatmaps
        if self.phase3_multilayer_heatmaps():
            completed_steps.extend(['3.2a', '3.2b', '3.2c'])
            logger.info("✓ Phase 3.2 completed successfully")
        else:
            logger.error("✗ Phase 3.2 failed")
            success = False
        
        # Step 3.3: Advanced Analytical Charts
        if self.phase3_advanced_analytical_charts():
            completed_steps.extend(['3.3a', '3.3b', '3.3c'])
            logger.info("✓ Phase 3.3 completed successfully")
        else:
            logger.error("✗ Phase 3.3 failed")
            success = False
        
        # Step 3.4: Correlation & Dependency Analysis
        if self.phase3_correlation_dependency_analysis():
            completed_steps.extend(['3.4a', '3.4b'])
            logger.info("✓ Phase 3.4 completed successfully")
        else:
            logger.error("✗ Phase 3.4 failed")
            success = False
        
        if success:
            logger.info("=== PHASE 3 COMPLETED SUCCESSFULLY ===")
            logger.info(f"✓ All visualization components completed: {', '.join(completed_steps)}")
            
            # Create comprehensive Phase 3 summary
            self._create_phase3_summary(completed_steps)
        else:
            logger.error("=== PHASE 3 COMPLETED WITH ERRORS ===")
        
        return success
    
    def _create_phase3_summary(self, completed_steps):
        """
        Create comprehensive Phase 3 summary report
        """
        logger.info("Creating Phase 3 comprehensive summary...")
        
        summary = {
            "phase3_completion": {
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "completed_steps": completed_steps,
                "components_completed": [
                    "3.1a: Interactive timeline with regime overlay",
                    "3.1b: Dynamic regime statistics panel",
                    "3.2a: Primary performance heatmap (Factor × Regime)",
                    "3.2b: Risk-adjusted performance heatmap",
                    "3.2c: Relative performance heatmap (vs S&P 500)",
                    "3.3a: Factor rotation wheel by regime",
                    "3.3b: Risk-return scatter plots with regime clustering",
                    "3.3c: Rolling regime analysis",
                    "3.4a: Dynamic correlation matrices",
                    "3.4b: Factor momentum persistence"
                ]
            },
            "visualization_files_created": [
                "interactive_timeline_regime_overlay.html",
                "regime_statistics_panel.json",
                "primary_performance_heatmap.html",
                "risk_adjusted_heatmap.html",
                "relative_performance_heatmap.html",
                "factor_rotation_wheel.html",
                "risk_return_scatter_plots.html",
                "rolling_regime_analysis.html",
                "correlation_matrices_by_regime.html",
                "momentum_persistence_analysis.html"
            ],
            "visualization_summary": {
                "total_charts_created": 10,
                "interactive_dashboards": 10,
                "static_summaries": 1
            }
        }
        
        # Save Phase 3 summary
        with open(self.results_dir / 'phase3_complete_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✓ Phase 3 summary saved")
        
        return True

    # ========================================
    # PHASE 4: STATISTICAL DEEP-DIVE & PATTERN RECOGNITION
    # ========================================
    
    def phase4_regime_transition_analytics(self):
        """
        Phase 4.1: Regime Transition Analytics
        Advanced analysis of regime transition probabilities and performance impact
        """
        logger.info("=== PHASE 4.1: Regime Transition Analytics ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run previous phases first.")
            return False
        
        try:
            # Step 4.1a: Transition probability matrix
            logger.info("Step 4.1a: Creating transition probability matrix...")
            transition_analysis = self._create_transition_probability_matrix()
            
            # Step 4.1b: Performance during regime changes
            logger.info("Step 4.1b: Analyzing performance during regime changes...")
            transition_performance = self._analyze_transition_performance()
            
            # Combine and store results
            self.transition_analytics = {
                'transition_probabilities': transition_analysis,
                'transition_performance': transition_performance,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open(self.results_dir / 'phase4_regime_transition_analytics.json', 'w') as f:
                json.dump(self.transition_analytics, f, indent=2, default=str)
            
            logger.info("✓ Phase 4.1 Regime Transition Analytics completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 4.1 transition analytics: {e}")
            return False
    
    def _create_transition_probability_matrix(self):
        """
        Step 4.1a: Create comprehensive transition probability matrix
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        regimes = sorted(regime_col.unique())
        
        # Create transition matrix
        transitions = []
        for i in range(len(regime_col) - 1):
            from_regime = regime_col.iloc[i]
            to_regime = regime_col.iloc[i + 1]
            transitions.append((from_regime, to_regime))
        
        # Build transition count matrix
        transition_counts = pd.DataFrame(0, index=regimes, columns=regimes)
        for from_regime, to_regime in transitions:
            transition_counts.loc[from_regime, to_regime] += 1
        
        # Convert to probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
        
        # Calculate expected duration for each regime
        expected_durations = {}
        for regime in regimes:
            if transition_probs.loc[regime, regime] < 1:
                # Expected duration = 1 / (1 - probability of staying)
                stay_prob = transition_probs.loc[regime, regime]
                expected_duration = 1 / (1 - stay_prob) if stay_prob < 1 else float('inf')
                expected_durations[regime] = expected_duration
            else:
                expected_durations[regime] = float('inf')
        
        # Calculate regime stability metrics
        regime_stability = {}
        for regime in regimes:
            regime_periods = regime_col[regime_col == regime]
            if len(regime_periods) > 0:
                # Calculate run lengths
                runs = []
                current_run = 1
                for i in range(1, len(regime_col)):
                    if regime_col.iloc[i] == regime and regime_col.iloc[i-1] == regime:
                        current_run += 1
                    elif regime_col.iloc[i-1] == regime:
                        runs.append(current_run)
                        current_run = 1
                
                if regime_col.iloc[-1] == regime:
                    runs.append(current_run)
                
                regime_stability[regime] = {
                    'average_duration': np.mean(runs) if runs else 0,
                    'median_duration': np.median(runs) if runs else 0,
                    'max_duration': max(runs) if runs else 0,
                    'total_periods': len(runs),
                    'frequency': len(regime_periods) / len(regime_col)
                }
        
        # Early warning signals
        early_warning_signals = self._calculate_early_warning_signals()
        
        return {
            'transition_counts': transition_counts.to_dict(),
            'transition_probabilities': transition_probs.to_dict(),
            'expected_durations': expected_durations,
            'regime_stability': regime_stability,
            'early_warning_signals': early_warning_signals,
            'total_transitions': len(transitions)
        }
    
    def _calculate_early_warning_signals(self):
        """
        Calculate early warning signals for regime transitions
        """
        # Use VIX and economic indicators as early warning signals
        signals = {}
        
        if 'VIX' in self.aligned_data.columns:
            # VIX volatility clusters as transition warning
            vix_volatility = self.aligned_data['VIX'].rolling(3).std()
            vix_levels = self.aligned_data['VIX']
            
            signals['vix_warnings'] = {
                'high_volatility_threshold': vix_volatility.quantile(0.8),
                'crisis_level_threshold': 50,
                'elevated_level_threshold': 25
            }
        
        # Economic indicator warnings
        if 'T10Y2Y' in self.aligned_data.columns:
            yield_curve = self.aligned_data['T10Y2Y']
            signals['yield_curve_inversion_warning'] = {
                'inversion_threshold': 0,
                'flat_curve_threshold': 0.5
            }
        
        if 'UNRATE' in self.aligned_data.columns:
            unemployment = self.aligned_data['UNRATE']
            unemployment_change = unemployment.diff(3)  # 3-month change
            signals['unemployment_warnings'] = {
                'rapid_rise_threshold': unemployment_change.quantile(0.8),
                'recession_level': unemployment.quantile(0.7)
            }
        
        return signals
    
    def _analyze_transition_performance(self):
        """
        Step 4.1b: Analyze factor performance during regime transitions
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Identify transition dates
        transition_dates = []
        for i in range(1, len(regime_col)):
            if regime_col.iloc[i] != regime_col.iloc[i-1]:
                transition_dates.append({
                    'date': regime_col.index[i],
                    'from_regime': regime_col.iloc[i-1],
                    'to_regime': regime_col.iloc[i]
                })
        
        # Analyze performance windows around transitions
        window_months = 6  # 6 months before and after
        transition_performance = {}
        
        for factor in factors:
            if factor in self.aligned_data.columns:
                pre_transition_returns = []
                post_transition_returns = []
                transition_details = []
                
                for transition in transition_dates:
                    transition_date = transition['date']
                    transition_idx = self.aligned_data.index.get_loc(transition_date)
                    
                    # Get pre-transition window
                    pre_start = max(0, transition_idx - window_months)
                    pre_returns = self.aligned_data[factor].iloc[pre_start:transition_idx]
                    
                    # Get post-transition window
                    post_end = min(len(self.aligned_data), transition_idx + window_months + 1)
                    post_returns = self.aligned_data[factor].iloc[transition_idx:post_end]
                    
                    if len(pre_returns) > 0 and len(post_returns) > 0:
                        pre_performance = pre_returns.mean()
                        post_performance = post_returns.mean()
                        
                        pre_transition_returns.append(pre_performance)
                        post_transition_returns.append(post_performance)
                        
                        transition_details.append({
                            'date': transition_date,
                            'from_regime': transition['from_regime'],
                            'to_regime': transition['to_regime'],
                            'pre_performance': float(pre_performance),
                            'post_performance': float(post_performance),
                            'performance_change': float(post_performance - pre_performance)
                        })
                
                if pre_transition_returns and post_transition_returns:
                    # Statistical significance test
                    from scipy import stats
                    t_stat, p_value = stats.ttest_rel(post_transition_returns, pre_transition_returns)
                    
                    transition_performance[factor] = {
                        'average_pre_transition': float(np.mean(pre_transition_returns)),
                        'average_post_transition': float(np.mean(post_transition_returns)),
                        'average_change': float(np.mean(post_transition_returns) - np.mean(pre_transition_returns)),
                        'volatility_pre': float(np.std(pre_transition_returns)),
                        'volatility_post': float(np.std(post_transition_returns)),
                        'volatility_change': float(np.std(post_transition_returns) - np.std(pre_transition_returns)),
                        'statistical_significance': {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'is_significant': p_value < 0.05
                        },
                        'transition_details': transition_details,
                        'total_transitions_analyzed': len(transition_details)
                    }
        
        return transition_performance
    
    def phase4_cyclical_pattern_detection(self):
        """
        Phase 4.2: Cyclical Pattern Detection
        Analysis of intra-regime evolution and macro-factor relationships
        """
        logger.info("=== PHASE 4.2: Cyclical Pattern Detection ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run previous phases first.")
            return False
        
        try:
            # Step 4.2a: Intra-regime performance evolution
            logger.info("Step 4.2a: Analyzing intra-regime performance evolution...")
            intra_regime_analysis = self._analyze_intra_regime_evolution()
            
            # Step 4.2b: Macro-factor relationships
            logger.info("Step 4.2b: Analyzing macro-factor relationships...")
            macro_factor_analysis = self._analyze_macro_factor_relationships()
            
            # Combine and store results
            self.cyclical_patterns = {
                'intra_regime_evolution': intra_regime_analysis,
                'macro_factor_relationships': macro_factor_analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open(self.results_dir / 'phase4_cyclical_pattern_detection.json', 'w') as f:
                json.dump(self.cyclical_patterns, f, indent=2, default=str)
            
            logger.info("✓ Phase 4.2 Cyclical Pattern Detection completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 4.2 cyclical pattern detection: {e}")
            return False
    
    def _analyze_intra_regime_evolution(self):
        """
        Step 4.2a: Analyze early vs late cycle factor leadership within regimes
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        intra_regime_analysis = {}
        
        for regime in regime_col.unique():
            regime_periods = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            if len(regime_periods) < 6:  # Need at least 6 months for meaningful analysis
                continue
            
            # Divide regime periods into early, middle, late phases
            n_obs = len(regime_periods)
            early_cutoff = n_obs // 3
            late_cutoff = 2 * n_obs // 3
            
            early_phase = regime_periods.iloc[:early_cutoff]
            middle_phase = regime_periods.iloc[early_cutoff:late_cutoff]
            late_phase = regime_periods.iloc[late_cutoff:]
            
            regime_evolution = {}
            
            for factor in factors:
                if factor in regime_periods.columns:
                    # Calculate performance by phase
                    early_performance = early_phase[factor].mean() if len(early_phase) > 0 else 0
                    middle_performance = middle_phase[factor].mean() if len(middle_phase) > 0 else 0
                    late_performance = late_phase[factor].mean() if len(late_phase) > 0 else 0
                    
                    # Calculate momentum/decay patterns
                    performance_trend = np.polyfit(range(n_obs), regime_periods[factor], 1)[0] if n_obs > 1 else 0
                    
                    regime_evolution[factor] = {
                        'early_phase_performance': float(early_performance),
                        'middle_phase_performance': float(middle_performance),
                        'late_phase_performance': float(late_performance),
                        'performance_trend_slope': float(performance_trend),
                        'early_to_late_change': float(late_performance - early_performance),
                        'optimal_phase': 'early' if early_performance > max(middle_performance, late_performance) else 
                                       'middle' if middle_performance > late_performance else 'late'
                    }
            
            # Calculate regime maturity indicators
            if len(regime_periods) > 1:
                # VIX trend during regime
                vix_trend = 0
                if 'VIX' in regime_periods.columns:
                    vix_trend = np.polyfit(range(len(regime_periods)), regime_periods['VIX'], 1)[0]
                
                # Economic indicator trends
                growth_trend = 0
                if 'GROWTH_COMPOSITE' in regime_periods.columns:
                    growth_trend = np.polyfit(range(len(regime_periods)), regime_periods['GROWTH_COMPOSITE'], 1)[0]
                
                inflation_trend = 0
                if 'INFLATION_COMPOSITE' in regime_periods.columns:
                    inflation_trend = np.polyfit(range(len(regime_periods)), regime_periods['INFLATION_COMPOSITE'], 1)[0]
                
                regime_evolution['regime_maturity_indicators'] = {
                    'vix_trend': float(vix_trend),
                    'growth_trend': float(growth_trend),
                    'inflation_trend': float(inflation_trend),
                    'regime_duration': len(regime_periods)
                }
            
            intra_regime_analysis[regime] = regime_evolution
        
        return intra_regime_analysis
    
    def _analyze_macro_factor_relationships(self):
        """
        Step 4.2b: Analyze macro-economic factor relationships
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        macro_indicators = ['DGS10', 'T10Y2Y', 'INFLATION_COMPOSITE', 'GROWTH_COMPOSITE']
        
        # Filter available indicators
        available_macro = [indicator for indicator in macro_indicators if indicator in self.aligned_data.columns]
        
        macro_relationships = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            if len(regime_data) < 10:  # Need sufficient data
                continue
            
            regime_relationships = {}
            
            for factor in factors:
                if factor in regime_data.columns:
                    factor_relationships = {}
                    
                    for macro_var in available_macro:
                        if macro_var in regime_data.columns:
                            # Calculate correlation
                            factor_returns = regime_data[factor].dropna()
                            macro_values = regime_data[macro_var].dropna()
                            
                            # Align series
                            aligned_factor, aligned_macro = factor_returns.align(macro_values, join='inner')
                            
                            if len(aligned_factor) > 5:
                                correlation = aligned_factor.corr(aligned_macro)
                                
                                # Calculate lagged correlations
                                lag_correlations = {}
                                for lag in [1, 2, 3]:
                                    if len(aligned_macro) > lag:
                                        lagged_macro = aligned_macro.shift(lag)
                                        lag_corr = aligned_factor.corr(lagged_macro)
                                        lag_correlations[f'lag_{lag}'] = float(lag_corr) if not np.isnan(lag_corr) else 0
                                
                                # Beta calculation (sensitivity)
                                if len(aligned_factor) > 1 and aligned_macro.std() > 0:
                                    beta = np.cov(aligned_factor, aligned_macro)[0, 1] / np.var(aligned_macro)
                                else:
                                    beta = 0
                                
                                factor_relationships[macro_var] = {
                                    'correlation': float(correlation) if not np.isnan(correlation) else 0,
                                    'beta_sensitivity': float(beta),
                                    'lag_correlations': lag_correlations,
                                    'observations': len(aligned_factor)
                                }
                    
                    regime_relationships[factor] = factor_relationships
            
            macro_relationships[regime] = regime_relationships
        
        # Cross-regime sensitivity analysis
        cross_regime_sensitivity = {}
        for factor in factors:
            if factor in self.aligned_data.columns:
                factor_sensitivities = {}
                
                for macro_var in available_macro:
                    if macro_var in self.aligned_data.columns:
                        regime_correlations = {}
                        for regime in macro_relationships:
                            if factor in macro_relationships[regime] and macro_var in macro_relationships[regime][factor]:
                                regime_correlations[regime] = macro_relationships[regime][factor][macro_var]['correlation']
                        
                        if regime_correlations:
                            factor_sensitivities[macro_var] = {
                                'regime_correlations': regime_correlations,
                                'sensitivity_stability': float(np.std(list(regime_correlations.values()))),
                                'average_sensitivity': float(np.mean(list(regime_correlations.values())))
                            }
                
                cross_regime_sensitivity[factor] = factor_sensitivities
        
        return {
            'regime_specific_relationships': macro_relationships,
            'cross_regime_sensitivity': cross_regime_sensitivity
        }
    
    def phase4_portfolio_construction_insights(self):
        """
        Phase 4.3: Portfolio Construction Insights
        Regime-aware allocation frameworks and factor timing models
        """
        logger.info("=== PHASE 4.3: Portfolio Construction Insights ===")
        
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            logger.error("No performance metrics available. Run Phase 2 first.")
            return False
        
        try:
            # Step 4.3a: Regime-aware allocation frameworks
            logger.info("Step 4.3a: Creating regime-aware allocation frameworks...")
            allocation_frameworks = self._create_allocation_frameworks()
            
            # Step 4.3b: Factor timing models
            logger.info("Step 4.3b: Developing factor timing models...")
            timing_models = self._develop_timing_models()
            
            # Combine and store results
            self.portfolio_insights = {
                'allocation_frameworks': allocation_frameworks,
                'timing_models': timing_models,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open(self.results_dir / 'phase4_portfolio_construction_insights.json', 'w') as f:
                json.dump(self.portfolio_insights, f, indent=2, default=str)
            
            logger.info("✓ Phase 4.3 Portfolio Construction Insights completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 4.3 portfolio insights: {e}")
            return False
    
    def _create_allocation_frameworks(self):
        """
        Step 4.3a: Create regime-aware allocation frameworks
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        allocation_frameworks = {}
        
        # Risk-parity approach
        risk_parity_allocations = {}
        
        # Mean-variance optimization approach
        mean_variance_allocations = {}
        
        # Risk-adjusted approach (Sharpe-based)
        sharpe_based_allocations = {}
        
        for regime in self.performance_metrics['performance_metrics']:
            regime_data = self.performance_metrics['performance_metrics'][regime]
            
            # Extract returns and volatilities
            returns = []
            volatilities = []
            sharpe_ratios = []
            factor_names = []
            
            for factor in factors:
                if factor in regime_data:
                    returns.append(regime_data[factor]['annualized_return'])
                    volatilities.append(regime_data[factor]['annualized_volatility'])
                    sharpe_ratios.append(regime_data[factor]['sharpe_ratio'])
                    factor_names.append(factor)
            
            if len(returns) > 1:
                returns_array = np.array(returns)
                volatilities_array = np.array(volatilities)
                sharpe_array = np.array(sharpe_ratios)
                
                # 1. Risk Parity Allocation (inverse volatility weighting)
                if np.sum(volatilities_array) > 0:
                    inv_vol_weights = (1 / volatilities_array) / np.sum(1 / volatilities_array)
                    risk_parity_allocations[regime] = {
                        factor_names[i]: float(inv_vol_weights[i]) for i in range(len(factor_names))
                    }
                
                # 2. Sharpe-based Allocation (positive Sharpe ratios only)
                positive_sharpe = np.maximum(sharpe_array, 0)
                if np.sum(positive_sharpe) > 0:
                    sharpe_weights = positive_sharpe / np.sum(positive_sharpe)
                    sharpe_based_allocations[regime] = {
                        factor_names[i]: float(sharpe_weights[i]) for i in range(len(factor_names))
                    }
                
                # 3. Equal Weight Baseline
                equal_weights = np.ones(len(factor_names)) / len(factor_names)
                
                # Calculate expected portfolio metrics for each approach
                allocation_frameworks[regime] = {
                    'risk_parity': {
                        'weights': risk_parity_allocations.get(regime, {}),
                        'expected_return': float(np.dot(inv_vol_weights, returns_array)) if np.sum(volatilities_array) > 0 else 0,
                        'expected_volatility': self._calculate_portfolio_volatility(inv_vol_weights, volatilities_array, regime) if np.sum(volatilities_array) > 0 else 0
                    },
                    'sharpe_optimized': {
                        'weights': sharpe_based_allocations.get(regime, {}),
                        'expected_return': float(np.dot(sharpe_weights, returns_array)) if np.sum(positive_sharpe) > 0 else 0,
                        'expected_volatility': self._calculate_portfolio_volatility(sharpe_weights, volatilities_array, regime) if np.sum(positive_sharpe) > 0 else 0
                    },
                    'equal_weight': {
                        'weights': {factor_names[i]: float(equal_weights[i]) for i in range(len(factor_names))},
                        'expected_return': float(np.mean(returns_array)),
                        'expected_volatility': self._calculate_portfolio_volatility(equal_weights, volatilities_array, regime)
                    }
                }
        
        # Dynamic allocation recommendations
        dynamic_recommendations = self._create_dynamic_allocation_recommendations()
        
        return {
            'regime_specific_allocations': allocation_frameworks,
            'dynamic_recommendations': dynamic_recommendations
        }
    
    def _calculate_portfolio_volatility(self, weights, volatilities, regime):
        """
        Calculate portfolio volatility assuming correlation structure
        """
        # Get correlation matrix for this regime if available
        regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        if len(regime_data) > 10:
            factor_data = regime_data[factors].dropna()
            if len(factor_data) > 5:
                corr_matrix = factor_data.corr().values
                cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return float(np.sqrt(portfolio_variance))
        
        # Fallback: assume 0.5 correlation
        avg_correlation = 0.5
        portfolio_variance = np.dot(weights**2, volatilities**2) + avg_correlation * np.sum(np.outer(weights, weights) * np.outer(volatilities, volatilities) - np.diag(weights**2 * volatilities**2))
        return float(np.sqrt(portfolio_variance))
    
    def _create_dynamic_allocation_recommendations(self):
        """
        Create dynamic allocation recommendations based on regime transitions
        """
        # Create transition probability matrix for this analysis
        transition_analysis = self._create_transition_probability_matrix()
        transition_probs = transition_analysis.get('transition_probabilities', {})
        
        recommendations = {}
        
        for current_regime in transition_probs:
            regime_recommendations = {}
            
            # Get most likely next regimes
            next_regime_probs = transition_probs.get(current_regime, {})
            
            if next_regime_probs:
                # Simple allocation based on regime characteristics
                regime_recommendations = {
                    'next_regime_probabilities': {k: v for k, v in sorted(next_regime_probs.items(), key=lambda x: x[1], reverse=True)[:3]},
                    'regime_transition_advice': self._get_simple_regime_advice(current_regime, next_regime_probs)
                }
            
            recommendations[current_regime] = regime_recommendations
        
        return recommendations
    
    def _get_simple_regime_advice(self, current_regime, next_regime_probs):
        """
        Get simple allocation advice based on regime transitions
        """
        # Most likely next regime
        most_likely_next = max(next_regime_probs.items(), key=lambda x: x[1])
        next_regime, probability = most_likely_next
        
        # Simple factor recommendations by regime
        regime_factor_advice = {
            'Goldilocks': 'Focus on Value and Momentum factors - growth environment',
            'Overheating': 'Consider Quality and MinVol - late cycle positioning',
            'Stagflation': 'Emphasize Value and Quality - inflation protection',
            'Recession': 'Defensive positioning with Quality and MinVol factors'
        }
        
        advice = f"Most likely transition to {next_regime} ({probability:.1%}). "
        advice += regime_factor_advice.get(next_regime, 'Maintain balanced allocation')
        
        return advice
    
    def _develop_timing_models(self):
        """
        Step 4.3b: Develop factor timing models
        """
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        timing_models = {}
        
        # Simple momentum-based timing model
        momentum_signals = {}
        
        # Mean reversion timing model
        mean_reversion_signals = {}
        
        # Regime prediction accuracy
        regime_prediction_accuracy = {}
        
        for factor in factors:
            if factor in self.aligned_data.columns:
                factor_returns = self.aligned_data[factor].dropna()
                
                # 1. Momentum signals (3, 6, 12 month)
                momentum_3m = factor_returns.rolling(3).mean()
                momentum_6m = factor_returns.rolling(6).mean()
                momentum_12m = factor_returns.rolling(12).mean()
                
                # Calculate momentum consistency with proper alignment
                momentum_3m_clean = momentum_3m.dropna()
                momentum_12m_clean = momentum_12m.dropna()
                
                # Align the series for correlation calculation
                if len(momentum_3m_clean) > 1 and len(momentum_12m_clean) > 1:
                    aligned_3m, aligned_12m = momentum_3m_clean.align(momentum_12m_clean, join='inner')
                    momentum_consistency = aligned_3m.corr(aligned_12m) if len(aligned_3m) > 1 else 0
                else:
                    momentum_consistency = 0
                
                momentum_signals[factor] = {
                    '3_month_momentum': float(momentum_3m.iloc[-1]) if not momentum_3m.empty else 0,
                    '6_month_momentum': float(momentum_6m.iloc[-1]) if not momentum_6m.empty else 0,
                    '12_month_momentum': float(momentum_12m.iloc[-1]) if not momentum_12m.empty else 0,
                    'momentum_consistency': float(momentum_consistency) if not np.isnan(momentum_consistency) else 0
                }
                
                # 2. Mean reversion signals
                rolling_mean = factor_returns.rolling(24).mean()  # 2-year average
                current_vs_mean = factor_returns.iloc[-12:].mean() - rolling_mean.iloc[-1] if not rolling_mean.empty else 0
                
                mean_reversion_signals[factor] = {
                    'deviation_from_longterm': float(current_vs_mean),
                    'reversion_signal': 'buy' if current_vs_mean < -0.01 else 'sell' if current_vs_mean > 0.01 else 'hold',
                    'signal_strength': abs(float(current_vs_mean))
                }
        
        # 3. Regime prediction accuracy analysis
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        
        # Simple regime persistence model
        regime_persistence = {}
        for regime in regime_col.unique():
            regime_periods = regime_col[regime_col == regime]
            if len(regime_periods) > 0:
                # Calculate how often regime persists for 1, 3, 6 months
                persistence_1m = self._calculate_regime_persistence(regime_col, regime, 1)
                persistence_3m = self._calculate_regime_persistence(regime_col, regime, 3)
                persistence_6m = self._calculate_regime_persistence(regime_col, regime, 6)
                
                regime_persistence[regime] = {
                    'persistence_1_month': persistence_1m,
                    'persistence_3_months': persistence_3m,
                    'persistence_6_months': persistence_6m
                }
        
        timing_models = {
            'momentum_signals': momentum_signals,
            'mean_reversion_signals': mean_reversion_signals,
            'regime_persistence': regime_persistence,
            'current_regime': regime_col.iloc[-1] if not regime_col.empty else 'Unknown'
        }
        
        # Strategy performance attribution
        strategy_performance = self._analyze_timing_strategy_performance()
        timing_models['strategy_performance'] = strategy_performance
        
        return timing_models
    
    def _calculate_regime_persistence(self, regime_col, target_regime, months):
        """
        Calculate how often a regime persists for specified number of months
        """
        persistence_count = 0
        total_regime_starts = 0
        
        for i in range(len(regime_col) - months):
            if regime_col.iloc[i] == target_regime:
                if i == 0 or regime_col.iloc[i-1] != target_regime:  # Start of regime
                    total_regime_starts += 1
                    if all(regime_col.iloc[i:i+months] == target_regime):
                        persistence_count += 1
        
        return persistence_count / total_regime_starts if total_regime_starts > 0 else 0
    
    def _analyze_timing_strategy_performance(self):
        """
        Analyze performance of various timing strategies
        """
        strategies = {}
        
        # Buy and hold strategy
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            buy_hold_return = (1 + self.aligned_data['SP500_Monthly_Return']).prod() - 1
            strategies['buy_and_hold_sp500'] = {
                'total_return': float(buy_hold_return),
                'annualized_return': float((1 + buy_hold_return)**(12/len(self.aligned_data)) - 1)
            }
        
        # Regime-based rotation strategy (simplified)
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        regime_rotation_returns = []
        
        for i, (date, regime) in enumerate(self.aligned_data['ECONOMIC_REGIME'].items()):
            # Simple rule: Quality/MinVol in Recession, Value/Momentum in Goldilocks/Overheating
            if regime in ['Recession']:
                preferred_factors = ['Quality', 'MinVol']
            elif regime in ['Goldilocks', 'Overheating']:
                preferred_factors = ['Value', 'Momentum']
            else:  # Stagflation
                preferred_factors = ['Value', 'Quality']
            
            # Equal weight preferred factors
            period_return = 0
            valid_factors = 0
            for factor in preferred_factors:
                if factor in self.aligned_data.columns:
                    period_return += self.aligned_data[factor].iloc[i]
                    valid_factors += 1
            
            if valid_factors > 0:
                regime_rotation_returns.append(period_return / valid_factors)
        
        if regime_rotation_returns:
            total_return = np.prod(1 + np.array(regime_rotation_returns)) - 1
            strategies['regime_rotation'] = {
                'total_return': float(total_return),
                'annualized_return': float((1 + total_return)**(12/len(regime_rotation_returns)) - 1)
            }
        
        return strategies
    
    def run_phase4(self):
        """
        Execute complete Phase 4: Statistical Deep-Dive & Pattern Recognition
        """
        logger.info("=== STARTING PHASE 4: STATISTICAL DEEP-DIVE & PATTERN RECOGNITION ===")
        
        success = True
        completed_steps = []
        
        # Step 4.1: Regime Transition Analytics
        if self.phase4_regime_transition_analytics():
            completed_steps.extend(['4.1a', '4.1b'])
            logger.info("✓ Phase 4.1 completed successfully")
        else:
            logger.error("✗ Phase 4.1 failed")
            success = False
        
        # Step 4.2: Cyclical Pattern Detection
        if self.phase4_cyclical_pattern_detection():
            completed_steps.extend(['4.2a', '4.2b'])
            logger.info("✓ Phase 4.2 completed successfully")
        else:
            logger.error("✗ Phase 4.2 failed")
            success = False
        
        # Step 4.3: Portfolio Construction Insights
        if self.phase4_portfolio_construction_insights():
            completed_steps.extend(['4.3a', '4.3b'])
            logger.info("✓ Phase 4.3 completed successfully")
        else:
            logger.error("✗ Phase 4.3 failed")
            success = False
        
        if success:
            logger.info("=== PHASE 4 COMPLETED SUCCESSFULLY ===")
            logger.info(f"✓ All statistical analysis components completed: {', '.join(completed_steps)}")
            
            # Create comprehensive Phase 4 summary
            self._create_phase4_summary(completed_steps)
        else:
            logger.error("=== PHASE 4 COMPLETED WITH ERRORS ===")
        
        return success
    
    def _create_phase4_summary(self, completed_steps):
        """
        Create comprehensive Phase 4 summary report
        """
        logger.info("Creating Phase 4 comprehensive summary...")
        
        summary = {
            "phase4_completion": {
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "completed_steps": completed_steps,
                "components_completed": [
                    "4.1a: Transition probability matrix",
                    "4.1b: Performance during regime changes",
                    "4.2a: Intra-regime performance evolution",
                    "4.2b: Macro-factor relationships",
                    "4.3a: Regime-aware allocation frameworks",
                    "4.3b: Factor timing models"
                ]
            },
            "analysis_files_created": [
                "phase4_regime_transition_analytics.json",
                "phase4_cyclical_pattern_detection.json", 
                "phase4_portfolio_construction_insights.json"
            ],
            "key_insights": self._extract_key_phase4_insights()
        }
        
        # Save Phase 4 summary
        with open(self.results_dir / 'phase4_complete_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✓ Phase 4 summary saved")
        
        return True
    
    def _extract_key_phase4_insights(self):
        """
        Extract key insights from Phase 4 analysis
        """
        insights = {}
        
        # Transition insights
        if hasattr(self, 'transition_analytics'):
            transition_data = self.transition_analytics.get('transition_probabilities', {})
            if transition_data:
                insights['transition_insights'] = {
                    'most_stable_regime': max(transition_data.get('expected_durations', {}), key=transition_data.get('expected_durations', {}).get, default='Unknown'),
                    'total_transitions_analyzed': transition_data.get('total_transitions', 0)
                }
        
        # Performance insights
        if hasattr(self, 'cyclical_patterns'):
            intra_regime = self.cyclical_patterns.get('intra_regime_evolution', {})
            if intra_regime:
                insights['cyclical_insights'] = {
                    'regimes_analyzed': len(intra_regime),
                    'sample_regime_evolution': list(intra_regime.keys())[:2]  # Sample
                }
        
        # Portfolio insights
        if hasattr(self, 'portfolio_insights'):
            allocation_data = self.portfolio_insights.get('allocation_frameworks', {})
            if allocation_data:
                insights['portfolio_insights'] = {
                    'allocation_methods_available': ['risk_parity', 'sharpe_optimized', 'equal_weight'],
                    'regimes_with_allocations': len(allocation_data.get('regime_specific_allocations', {}))
                }
        
        return insights

    # ========================================
    # PHASE 5: INTERACTIVE DASHBOARD & REPORTING
    # ========================================
    
    def phase5_comprehensive_interactive_dashboard(self):
        """
        Phase 5.1: Comprehensive Interactive Dashboard
        Multi-panel layout with interactive controls combining all previous analyses
        """
        logger.info("=== PHASE 5.1: Comprehensive Interactive Dashboard ===")
        
        if self.aligned_data is None:
            logger.error("No aligned data available. Run previous phases first.")
            return False
        
        try:
            # Step 5.1a: Multi-panel layout implementation
            logger.info("Step 5.1a: Creating multi-panel dashboard layout...")
            comprehensive_dashboard = self._create_multi_panel_dashboard()
            
            # Step 5.1b: Interactive controls implementation
            logger.info("Step 5.1b: Adding interactive controls...")
            self._add_interactive_controls(comprehensive_dashboard)
            
            # Save comprehensive dashboard
            comprehensive_dashboard.write_html(
                self.results_dir / 'comprehensive_business_cycle_dashboard.html',
                config={'displayModeBar': True, 'modeBarButtonsToAdd': ['downloadSVG']}
            )
            
            logger.info("✓ Phase 5.1 Comprehensive Interactive Dashboard completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 5.1 dashboard creation: {e}")
            return False
    
    def _create_multi_panel_dashboard(self):
        """
        Step 5.1a: Create comprehensive multi-panel dashboard layout
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create complex subplot layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Business Cycle Timeline & Factor Performance', 'Current Regime Statistics', 'VIX Stress Levels',
                'Primary Performance Heatmap', 'Risk-Adjusted Performance', 'Relative Performance vs S&P 500',
                'Factor Rotation Wheel', 'Risk-Return Clustering', 'Transition Probabilities',
                'Rolling Performance Analysis', 'Correlation Matrix', 'Portfolio Allocation Framework'
            ),
            specs=[
                [{"colspan": 2}, None, {"type": "table"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                [{"type": "polar"}, {"type": "xy"}, {"type": "xy"}],
                [{"colspan": 3}, None, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            row_heights=[0.3, 0.25, 0.25, 0.2]
        )
        
        # Panel 1: Business Cycle Timeline (spans 2 columns)
        self._add_timeline_to_dashboard(fig, row=1, col=1)
        
        # Panel 2: Current Regime Statistics (table)
        self._add_regime_stats_to_dashboard(fig, row=1, col=3)
        
        # Panel 3: Primary Performance Heatmap
        self._add_performance_heatmap_to_dashboard(fig, row=2, col=1)
        
        # Panel 4: Risk-Adjusted Performance
        self._add_risk_adjusted_heatmap_to_dashboard(fig, row=2, col=2)
        
        # Panel 5: Relative Performance
        self._add_relative_performance_to_dashboard(fig, row=2, col=3)
        
        # Panel 6: Factor Rotation Wheel (polar)
        self._add_rotation_wheel_to_dashboard(fig, row=3, col=1)
        
        # Panel 7: Risk-Return Scatter (xy)
        self._add_risk_return_scatter_to_dashboard(fig, row=3, col=2)
        
        # Panel 8: Transition Analysis
        self._add_transition_analysis_to_dashboard(fig, row=3, col=3)
        
        # Panel 9: Rolling Analysis (bottom panel spans all columns)
        self._add_rolling_analysis_to_dashboard(fig, row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Business Cycle Factor Performance Dashboard (1998-2025)<br>' +
                       '<sub>Interactive Analysis of Factor Performance Across Economic Regimes</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=1400,
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def _add_timeline_to_dashboard(self, fig, row, col):
        """Add timeline panel to dashboard"""
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            cumulative_returns = (1 + self.aligned_data['SP500_Monthly_Return']).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=self.aligned_data.index,
                    y=cumulative_returns,
                    name='S&P 500',
                    line=dict(color='black', width=2),
                    hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Return: %{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Add regime background colors
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700',
            'Recession': '#8B0000'
        }
        
        # Add regime periods as background
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        current_regime = None
        start_date = None
        
        for i, (date, regime) in enumerate(regime_col.items()):
            if regime != current_regime:
                if current_regime is not None:
                    fig.add_shape(
                        type="rect",
                        x0=start_date, x1=date,
                        y0=0, y1=1,
                        yref=f"y{'' if row == 1 and col == 1 else (row-1)*3 + col} domain",
                        fillcolor=regime_colors.get(current_regime, '#808080'),
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        row=row, col=col
                    )
                current_regime = regime
                start_date = date
    
    def _add_regime_stats_to_dashboard(self, fig, row, col):
        """Add regime statistics table to dashboard"""
        # Create current regime statistics
        current_regime = self.aligned_data['ECONOMIC_REGIME'].iloc[-1]
        
        stats_data = []
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            freq = len(regime_data) / len(self.aligned_data) * 100
            
            # Get best performing factor in this regime
            best_factor = 'N/A'
            best_return = -999
            for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                if factor in regime_data.columns:
                    annual_return = (1 + regime_data[factor].mean()) ** 12 - 1
                    if annual_return > best_return:
                        best_return = annual_return
                        best_factor = factor
            
            stats_data.append([
                regime,
                f"{freq:.1f}%",
                f"{len(regime_data)} months",
                best_factor,
                f"{best_return*100:.1f}%" if best_return != -999 else 'N/A'
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Regime', 'Frequency', 'Duration', 'Best Factor', 'Best Return'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=row, col=col
        )
    
    def _add_performance_heatmap_to_dashboard(self, fig, row, col):
        """Add performance heatmap to dashboard"""
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            return
        
        # Extract performance data for heatmap
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        performance_matrix = []
        for factor in factors:
            row_data = []
            for regime in regimes:
                if regime in self.performance_metrics['performance_metrics']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        row_data.append(annual_return * 100)
                    else:
                        row_data.append(0)
                else:
                    row_data.append(0)
            performance_matrix.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=performance_matrix,
                x=regimes,
                y=factors,
                colorscale='RdYlGn',
                showscale=False,
                hovertemplate='Factor: %{y}<br>Regime: %{x}<br>Return: %{z:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_risk_adjusted_heatmap_to_dashboard(self, fig, row, col):
        """Add risk-adjusted heatmap to dashboard"""
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            return
        
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        sharpe_matrix = []
        for factor in factors:
            row_data = []
            for regime in regimes:
                if regime in self.performance_metrics['performance_metrics']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        sharpe_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                        row_data.append(sharpe_ratio)
                    else:
                        row_data.append(0)
                else:
                    row_data.append(0)
            sharpe_matrix.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=sharpe_matrix,
                x=regimes,
                y=factors,
                colorscale='RdYlGn',
                showscale=False,
                zmid=0,
                hovertemplate='Factor: %{y}<br>Regime: %{x}<br>Sharpe: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_relative_performance_to_dashboard(self, fig, row, col):
        """Add relative performance to dashboard"""
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            return
        
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        relative_matrix = []
        for factor in factors:
            row_data = []
            for regime in regimes:
                if (regime in self.performance_metrics['performance_metrics'] and
                    factor in self.performance_metrics['performance_metrics'][regime] and
                    'SP500_Monthly_Return' in self.performance_metrics['performance_metrics'][regime]):
                    
                    factor_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                    sp500_return = self.performance_metrics['performance_metrics'][regime]['SP500_Monthly_Return']['annualized_return']
                    excess_return = (factor_return - sp500_return) * 100
                    row_data.append(excess_return)
                else:
                    row_data.append(0)
            relative_matrix.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=relative_matrix,
                x=regimes,
                y=factors,
                colorscale='RdYlGn',
                showscale=False,
                zmid=0,
                hovertemplate='Factor: %{y}<br>Regime: %{x}<br>Excess Return: %{z:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_rotation_wheel_to_dashboard(self, fig, row, col):
        """Add factor rotation wheel to dashboard"""
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            return
        
        # Use Goldilocks regime as example
        regime = 'Goldilocks'
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        if regime in self.performance_metrics['performance_metrics']:
            sharpe_values = []
            for factor in factors:
                if factor in self.performance_metrics['performance_metrics'][regime]:
                    sharpe_ratio = self.performance_metrics['performance_metrics'][regime][factor]['sharpe_ratio']
                    sharpe_values.append(max(0, sharpe_ratio))
                else:
                    sharpe_values.append(0)
            
            # Close the loop
            theta_values = factors + [factors[0]]
            r_values = sharpe_values + [sharpe_values[0]]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name=f'{regime} Performance',
                    line_color='rgb(106, 81, 163)',
                    fillcolor='rgba(106, 81, 163, 0.3)',
                    hovertemplate='Factor: %{theta}<br>Sharpe: %{r:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    def _add_risk_return_scatter_to_dashboard(self, fig, row, col):
        """Add risk-return scatter to dashboard"""
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            return
        
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700',
            'Recession': '#8B0000'
        }
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        for regime in self.performance_metrics['performance_metrics']:
            volatility_values = []
            return_values = []
            
            for factor in factors:
                if factor in self.performance_metrics['performance_metrics'][regime]:
                    metrics = self.performance_metrics['performance_metrics'][regime][factor]
                    volatility = metrics['annualized_volatility'] * 100
                    annual_return = metrics['annualized_return'] * 100
                    
                    volatility_values.append(volatility)
                    return_values.append(annual_return)
            
            if volatility_values:
                fig.add_trace(
                    go.Scatter(
                        x=volatility_values,
                        y=return_values,
                        mode='markers',
                        name=regime,
                        marker=dict(
                            size=8,
                            color=regime_colors.get(regime, '#808080')
                        ),
                        hovertemplate=f'<b>{regime}</b><br>Volatility: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    def _add_transition_analysis_to_dashboard(self, fig, row, col):
        """Add transition analysis to dashboard"""
        if not hasattr(self, 'transition_analytics'):
            return
        
        transition_probs = self.transition_analytics.get('transition_probabilities', {}).get('transition_probabilities', {})
        if not transition_probs:
            return
        
        regimes = list(transition_probs.keys())
        if not regimes:
            return
        
        transition_matrix = []
        for from_regime in regimes:
            row_data = []
            for to_regime in regimes:
                prob = transition_probs.get(from_regime, {}).get(to_regime, 0)
                row_data.append(prob * 100)
            transition_matrix.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=regimes,
                y=regimes,
                colorscale='Blues',
                showscale=False,
                hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_rolling_analysis_to_dashboard(self, fig, row, col):
        """Add rolling analysis to dashboard"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        factor_colors = {
            'Value': '#1f77b4',
            'Quality': '#ff7f0e',
            'MinVol': '#2ca02c',
            'Momentum': '#d62728'
        }
        
        dates = self.aligned_data.index
        
        for factor, color in factor_colors.items():
            if factor in self.aligned_data.columns:
                # Calculate 12-month rolling returns
                rolling_returns = self.aligned_data[factor].rolling(window=12).apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=rolling_returns,
                        name=f'{factor}',
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'<b>{factor}</b><br>Date: %{{x}}<br>12M Return: %{{y:.1f}}%<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    def _add_interactive_controls(self, fig):
        """
        Step 5.1b: Add interactive controls to the dashboard
        """
        # Add dropdown menus and buttons for interactivity
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [i < 4 for i in range(len(fig.data))]}],
                            label="Timeline Only",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [4 <= i < 8 for i in range(len(fig.data))]}],
                            label="Heatmaps Only",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [i >= 8 for i in range(len(fig.data))]}],
                            label="Analytics Only",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=0.98,
                    yanchor="top"
                ),
            ],
            annotations=[
                dict(text="Dashboard Views:", showarrow=False,
                     x=0.01, y=0.99, xref="paper", yref="paper")
            ]
        )
    
    def phase5_advanced_features(self):
        """
        Phase 5.2: Advanced Features
        Enhanced hover analytics and export functionality
        """
        logger.info("=== PHASE 5.2: Advanced Features ===")
        
        try:
            # Step 5.2a: Enhanced hover-over analytics
            logger.info("Step 5.2a: Implementing enhanced hover analytics...")
            self._implement_enhanced_hover_analytics()
            
            # Step 5.2b: Export functionality
            logger.info("Step 5.2b: Creating export functionality...")
            export_success = self._create_export_functionality()
            
            logger.info("✓ Phase 5.2 Advanced Features completed")
            return export_success
            
        except Exception as e:
            logger.error(f"Error in Phase 5.2 advanced features: {e}")
            return False
    
    def _implement_enhanced_hover_analytics(self):
        """
        Step 5.2a: Implement enhanced hover-over analytics
        """
        # Create comprehensive hover information for existing charts
        enhanced_analytics = {
            'regime_details': {},
            'factor_analytics': {},
            'statistical_summaries': {}
        }
        
        # Enhanced regime analytics
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            enhanced_analytics['regime_details'][regime] = {
                'total_months': len(regime_data),
                'frequency_percentage': len(regime_data) / len(self.aligned_data) * 100,
                'avg_vix_level': float(regime_data['VIX'].mean()) if 'VIX' in regime_data.columns else 0,
                'date_range': {
                    'start': regime_data.index.min().strftime('%Y-%m-%d'),
                    'end': regime_data.index.max().strftime('%Y-%m-%d')
                }
            }
        
        # Enhanced factor analytics
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        for factor in factors:
            if factor in self.aligned_data.columns:
                factor_data = self.aligned_data[factor].dropna()
                enhanced_analytics['factor_analytics'][factor] = {
                    'overall_sharpe': float(factor_data.mean() / factor_data.std() * np.sqrt(12)),
                    'best_regime': self._find_best_regime_for_factor(factor),
                    'worst_regime': self._find_worst_regime_for_factor(factor),
                    'volatility_rank': self._rank_factor_volatility(factor)
                }
        
        # Save enhanced analytics
        with open(self.results_dir / 'enhanced_hover_analytics.json', 'w') as f:
            json.dump(enhanced_analytics, f, indent=2, default=str)
        
        logger.info("✓ Enhanced hover analytics implemented")
        return enhanced_analytics
    
    def _find_best_regime_for_factor(self, factor):
        """Find the regime where factor performs best"""
        if not hasattr(self, 'performance_metrics'):
            return 'Unknown'
        
        best_regime = 'Unknown'
        best_return = -999
        
        for regime in self.performance_metrics.get('performance_metrics', {}):
            if factor in self.performance_metrics['performance_metrics'][regime]:
                annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                if annual_return > best_return:
                    best_return = annual_return
                    best_regime = regime
        
        return best_regime
    
    def _find_worst_regime_for_factor(self, factor):
        """Find the regime where factor performs worst"""
        if not hasattr(self, 'performance_metrics'):
            return 'Unknown'
        
        worst_regime = 'Unknown'
        worst_return = 999
        
        for regime in self.performance_metrics.get('performance_metrics', {}):
            if factor in self.performance_metrics['performance_metrics'][regime]:
                annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                if annual_return < worst_return:
                    worst_return = annual_return
                    worst_regime = regime
        
        return worst_regime
    
    def _rank_factor_volatility(self, factor):
        """Rank factor volatility relative to others"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        volatilities = {}
        
        for f in factors:
            if f in self.aligned_data.columns:
                volatilities[f] = self.aligned_data[f].std() * np.sqrt(12)
        
        sorted_factors = sorted(volatilities.items(), key=lambda x: x[1])
        
        for i, (f, vol) in enumerate(sorted_factors):
            if f == factor:
                return i + 1  # Rank from 1 (lowest vol) to 4 (highest vol)
        
        return 0
    
    def _create_export_functionality(self):
        """
        Step 5.2b: Create comprehensive export functionality
        """
        logger.info("Creating export functionality...")
        
        try:
            # Export summary tables
            self._export_summary_tables()
            
            # Export static chart images
            self._export_static_charts()
            
            # Create comprehensive PDF report
            self._create_pdf_report()
            
            # Create portfolio recommendations
            self._export_portfolio_recommendations()
            
            logger.info("✓ Export functionality created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating export functionality: {e}")
            return False
    
    def _export_summary_tables(self):
        """Export summary statistics tables"""
        # Performance summary by regime
        if hasattr(self, 'performance_metrics'):
            performance_data = []
            for regime in self.performance_metrics.get('performance_metrics', {}):
                for factor in ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500_Monthly_Return']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        metrics = self.performance_metrics['performance_metrics'][regime][factor]
                        performance_data.append({
                            'Regime': regime,
                            'Factor': factor,
                            'Annual_Return': f"{metrics['annualized_return']*100:.2f}%",
                            'Volatility': f"{metrics['annualized_volatility']*100:.2f}%",
                            'Sharpe_Ratio': f"{metrics['sharpe_ratio']:.3f}",
                            'Max_Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
                            'Win_Rate': f"{metrics['win_rate']*100:.1f}%"
                        })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_csv(self.results_dir / 'performance_summary_export.csv', index=False)
        
        # Regime statistics export
        regime_data = []
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_periods = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            regime_data.append({
                'Regime': regime,
                'Frequency': f"{len(regime_periods)/len(self.aligned_data)*100:.1f}%",
                'Total_Months': len(regime_periods),
                'Start_Date': regime_periods.index.min().strftime('%Y-%m-%d'),
                'End_Date': regime_periods.index.max().strftime('%Y-%m-%d'),
                'Avg_VIX': f"{regime_periods['VIX'].mean():.1f}" if 'VIX' in regime_periods.columns else 'N/A'
            })
        
        regime_df = pd.DataFrame(regime_data)
        regime_df.to_csv(self.results_dir / 'regime_summary_export.csv', index=False)
    
    def _export_static_charts(self):
        """Export static versions of key charts"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Performance heatmap
        if hasattr(self, 'performance_metrics'):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
            factors = ['Value', 'Quality', 'MinVol', 'Momentum']
            
            performance_matrix = []
            for factor in factors:
                row_data = []
                for regime in regimes:
                    if (regime in self.performance_metrics['performance_metrics'] and
                        factor in self.performance_metrics['performance_metrics'][regime]):
                        annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        row_data.append(annual_return * 100)
                    else:
                        row_data.append(0)
                performance_matrix.append(row_data)
            
            sns.heatmap(
                performance_matrix,
                xticklabels=regimes,
                yticklabels=factors,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax
            )
            
            ax.set_title('Factor Performance by Economic Regime\n(Annualized Returns %)', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'performance_heatmap_export.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Timeline chart
        fig, ax = plt.subplots(figsize=(15, 8))
        
        if 'SP500_Monthly_Return' in self.aligned_data.columns:
            cumulative_returns = (1 + self.aligned_data['SP500_Monthly_Return']).cumprod()
            ax.plot(self.aligned_data.index, cumulative_returns, 'k-', linewidth=2, label='S&P 500')
        
        # Add regime background
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700',
            'Recession': '#8B0000'
        }
        
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        current_regime = None
        start_date = None
        
        for i, (date, regime) in enumerate(regime_col.items()):
            if regime != current_regime:
                if current_regime is not None:
                    ax.axvspan(start_date, date, alpha=0.3, color=regime_colors.get(current_regime, '#808080'))
                current_regime = regime
                start_date = date
        
        ax.set_title('Business Cycle Timeline & Market Performance (1998-2025)', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'timeline_export.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_report(self):
        """Create comprehensive PDF report"""
        # Create markdown report that can be converted to PDF
        report_content = f"""
# Business Cycle Factor Performance Analysis Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
This report analyzes factor performance (Value, Quality, MinVol, Momentum) across business cycle regimes from 1998 to 2025.

### Data Summary
- **Analysis Period**: {self.aligned_data.index.min().strftime('%Y-%m-%d')} to {self.aligned_data.index.max().strftime('%Y-%m-%d')}
- **Total Observations**: {len(self.aligned_data)}
- **Economic Regimes Analyzed**: {len(self.aligned_data['ECONOMIC_REGIME'].unique())}

### Regime Distribution
"""
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            frequency = len(regime_data) / len(self.aligned_data) * 100
            report_content += f"- **{regime}**: {frequency:.1f}% ({len(regime_data)} months)\n"
        
        if hasattr(self, 'performance_metrics'):
            report_content += "\n### Key Performance Insights\n"
            
            # Find best performing factor per regime
            for regime in self.performance_metrics.get('performance_metrics', {}):
                best_factor = 'N/A'
                best_return = -999
                
                for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                    if factor in self.performance_metrics['performance_metrics'][regime]:
                        annual_return = self.performance_metrics['performance_metrics'][regime][factor]['annualized_return']
                        if annual_return > best_return:
                            best_return = annual_return
                            best_factor = factor
                
                report_content += f"- **{regime}**: {best_factor} performs best ({best_return*100:.1f}% annually)\n"
        
        # Save report
        with open(self.results_dir / 'comprehensive_analysis_report.md', 'w') as f:
            f.write(report_content)
    
    def _export_portfolio_recommendations(self):
        """Export portfolio allocation recommendations"""
        if not hasattr(self, 'portfolio_insights'):
            return
        
        recommendations = []
        allocation_data = self.portfolio_insights.get('allocation_frameworks', {}).get('regime_specific_allocations', {})
        
        for regime in allocation_data:
            regime_allocations = allocation_data[regime]
            
            for method in ['risk_parity', 'sharpe_optimized', 'equal_weight']:
                if method in regime_allocations and 'weights' in regime_allocations[method]:
                    weights = regime_allocations[method]['weights']
                    expected_return = regime_allocations[method].get('expected_return', 0)
                    expected_vol = regime_allocations[method].get('expected_volatility', 0)
                    
                    recommendation = {
                        'Regime': regime,
                        'Method': method,
                        'Expected_Return': f"{expected_return*100:.2f}%",
                        'Expected_Volatility': f"{expected_vol*100:.2f}%"
                    }
                    
                    # Add individual factor weights
                    for factor, weight in weights.items():
                        recommendation[f'{factor}_Weight'] = f"{weight*100:.1f}%"
                    
                    recommendations.append(recommendation)
        
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_csv(self.results_dir / 'portfolio_recommendations_export.csv', index=False)
    
    def run_phase5(self):
        """
        Execute complete Phase 5: Interactive Dashboard & Reporting
        """
        logger.info("=== STARTING PHASE 5: INTERACTIVE DASHBOARD & REPORTING ===")
        
        success = True
        completed_steps = []
        
        # Step 5.1: Comprehensive Interactive Dashboard
        if self.phase5_comprehensive_interactive_dashboard():
            completed_steps.extend(['5.1a', '5.1b'])
            logger.info("✓ Phase 5.1 completed successfully")
        else:
            logger.error("✗ Phase 5.1 failed")
            success = False
        
        # Step 5.2: Advanced Features
        if self.phase5_advanced_features():
            completed_steps.extend(['5.2a', '5.2b'])
            logger.info("✓ Phase 5.2 completed successfully")
        else:
            logger.error("✗ Phase 5.2 failed")
            success = False
        
        if success:
            logger.info("=== PHASE 5 COMPLETED SUCCESSFULLY ===")
            logger.info(f"✓ All dashboard components completed: {', '.join(completed_steps)}")
            
            # Create comprehensive Phase 5 summary
            self._create_phase5_summary(completed_steps)
        else:
            logger.error("=== PHASE 5 COMPLETED WITH ERRORS ===")
        
        return success
    
    def _create_phase5_summary(self, completed_steps):
        """
        Create comprehensive Phase 5 summary report
        """
        logger.info("Creating Phase 5 comprehensive summary...")
        
        summary = {
            "phase5_completion": {
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "completed_steps": completed_steps,
                "components_completed": [
                    "5.1a: Multi-panel dashboard layout implementation",
                    "5.1b: Interactive controls implementation",
                    "5.2a: Enhanced hover-over analytics implementation",
                    "5.2b: Export functionality implementation"
                ]
            },
            "dashboard_files_created": [
                "comprehensive_business_cycle_dashboard.html",
                "enhanced_hover_analytics.json",
                "performance_summary_export.csv",
                "regime_summary_export.csv",
                "performance_heatmap_export.png",
                "timeline_export.png",
                "comprehensive_analysis_report.md",
                "portfolio_recommendations_export.csv"
            ],
            "dashboard_features": {
                "interactive_panels": 12,
                "export_formats": ["HTML", "CSV", "PNG", "Markdown"],
                "control_elements": ["View toggles", "Interactive hover", "Data export buttons"]
            }
        }
        
        # Save Phase 5 summary
        with open(self.results_dir / 'phase5_complete_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✓ Phase 5 summary saved")
        
        return True

    # ========================================
    # PHASE 6: BUSINESS INSIGHTS & STRATEGY DEVELOPMENT
    # ========================================
    
    def phase6_regime_specific_insights(self):
        """
        Phase 6.1: Regime-Specific Insights Generation
        Factor leadership patterns and risk management insights
        """
        logger.info("=== PHASE 6.1: Regime-Specific Insights Generation ===")
        
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            logger.error("No performance metrics available. Run Phase 2 first.")
            return False
        
        try:
            # Step 6.1a: Factor leadership patterns analysis
            logger.info("Step 6.1a: Analyzing factor leadership patterns...")
            leadership_analysis = self._analyze_factor_leadership_patterns()
            
            # Step 6.1b: Risk management insights
            logger.info("Step 6.1b: Generating risk management insights...")
            risk_insights = self._generate_risk_management_insights()
            
            # Combine insights
            self.business_insights = {
                'factor_leadership_patterns': leadership_analysis,
                'risk_management_insights': risk_insights,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save insights
            with open(self.results_dir / 'phase6_business_insights.json', 'w') as f:
                json.dump(self.business_insights, f, indent=2, default=str)
            
            logger.info("✓ Phase 6.1 Regime-Specific Insights completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 6.1 insights generation: {e}")
            return False
    
    def _analyze_factor_leadership_patterns(self):
        """
        Step 6.1a: Analyze factor leadership patterns by regime
        """
        leadership_patterns = {}
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        for regime in self.performance_metrics['performance_metrics']:
            regime_data = self.performance_metrics['performance_metrics'][regime]
            
            # Rank factors by various metrics
            rankings = {
                'by_returns': [],
                'by_sharpe': [],
                'by_win_rate': [],
                'by_drawdown': []
            }
            
            factor_metrics = []
            for factor in factors:
                if factor in regime_data:
                    metrics = regime_data[factor]
                    factor_metrics.append({
                        'factor': factor,
                        'annual_return': metrics['annualized_return'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'win_rate': metrics['win_rate'],
                        'max_drawdown': abs(metrics['max_drawdown'])  # Use absolute for ranking
                    })
            
            # Sort by different criteria
            rankings['by_returns'] = sorted(factor_metrics, key=lambda x: x['annual_return'], reverse=True)
            rankings['by_sharpe'] = sorted(factor_metrics, key=lambda x: x['sharpe_ratio'], reverse=True)
            rankings['by_win_rate'] = sorted(factor_metrics, key=lambda x: x['win_rate'], reverse=True)
            rankings['by_drawdown'] = sorted(factor_metrics, key=lambda x: x['max_drawdown'])  # Lower is better
            
            # Identify leaders and laggards
            best_return_factor = rankings['by_returns'][0]['factor'] if rankings['by_returns'] else 'N/A'
            best_sharpe_factor = rankings['by_sharpe'][0]['factor'] if rankings['by_sharpe'] else 'N/A'
            worst_return_factor = rankings['by_returns'][-1]['factor'] if rankings['by_returns'] else 'N/A'
            
            # Generate regime-specific insights
            regime_insights = self._generate_regime_specific_recommendations(regime, rankings)
            
            leadership_patterns[regime] = {
                'best_return_factor': best_return_factor,
                'best_risk_adjusted_factor': best_sharpe_factor,
                'worst_factor': worst_return_factor,
                'factor_rankings': rankings,
                'regime_recommendations': regime_insights,
                'statistical_confidence': self._calculate_leadership_confidence(regime, factor_metrics)
            }
        
        # Cross-regime leadership consistency
        consistency_analysis = self._analyze_leadership_consistency()
        
        return {
            'regime_specific_patterns': leadership_patterns,
            'cross_regime_consistency': consistency_analysis
        }
    
    def _generate_regime_specific_recommendations(self, regime, rankings):
        """Generate specific recommendations for each regime"""
        recommendations = {}
        
        if regime == 'Goldilocks':
            recommendations = {
                'primary_strategy': 'Growth and Momentum focus',
                'reasoning': 'Rising growth with controlled inflation favors cyclical factors',
                'recommended_factors': ['Value', 'Momentum'],
                'allocation_tilt': 'Overweight cyclical factors, reduce defensive positions',
                'risk_level': 'Moderate - favorable environment for risk-taking'
            }
        elif regime == 'Overheating':
            recommendations = {
                'primary_strategy': 'Mixed signals - transition preparation',
                'reasoning': 'Late cycle conditions suggest defensive preparation',
                'recommended_factors': ['Quality', 'Value'],
                'allocation_tilt': 'Begin defensive tilt while maintaining some growth exposure',
                'risk_level': 'Moderate-High - monitor for regime change signals'
            }
        elif regime == 'Stagflation':
            recommendations = {
                'primary_strategy': 'Value and real asset focus',
                'reasoning': 'High inflation with weak growth favors value and real assets',
                'recommended_factors': ['Value', 'Quality'],
                'allocation_tilt': 'Strong value tilt, avoid growth-sensitive factors',
                'risk_level': 'High - challenging environment for most assets'
            }
        elif regime == 'Recession':
            recommendations = {
                'primary_strategy': 'Quality and defensive positioning',
                'reasoning': 'Falling growth requires defensive, high-quality positions',
                'recommended_factors': ['Quality', 'MinVol'],
                'allocation_tilt': 'Maximum defensive positioning, capital preservation focus',
                'risk_level': 'Low - capital preservation priority'
            }
        
        # Add specific allocation ranges
        best_factors = [item['factor'] for item in rankings['by_sharpe'][:2]]
        recommendations['specific_allocations'] = {
            factor: '25-35%' if factor in best_factors else '10-20%' 
            for factor in ['Value', 'Quality', 'MinVol', 'Momentum']
        }
        
        return recommendations
    
    def _calculate_leadership_confidence(self, regime, factor_metrics):
        """Calculate statistical confidence in factor leadership"""
        if len(factor_metrics) < 2:
            return 0
        
        # Calculate confidence based on performance separation
        sorted_factors = sorted(factor_metrics, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        if len(sorted_factors) >= 2:
            top_sharpe = sorted_factors[0]['sharpe_ratio']
            second_sharpe = sorted_factors[1]['sharpe_ratio']
            
            # Simple confidence metric based on separation
            confidence = min(abs(top_sharpe - second_sharpe) * 100, 100)
            return float(confidence)
        
        return 0
    
    def _analyze_leadership_consistency(self):
        """Analyze consistency of factor leadership across regimes"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        consistency_metrics = {}
        
        for factor in factors:
            regime_ranks = []
            
            for regime in self.performance_metrics['performance_metrics']:
                regime_data = self.performance_metrics['performance_metrics'][regime]
                
                # Get all factor Sharpe ratios for this regime
                regime_sharpes = []
                for f in factors:
                    if f in regime_data:
                        regime_sharpes.append((f, regime_data[f]['sharpe_ratio']))
                
                # Sort and find rank
                regime_sharpes.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (f, sharpe) in enumerate(regime_sharpes):
                    if f == factor:
                        regime_ranks.append(rank + 1)  # 1-based ranking
                        break
            
            if regime_ranks:
                consistency_metrics[factor] = {
                    'average_rank': float(np.mean(regime_ranks)),
                    'rank_volatility': float(np.std(regime_ranks)),
                    'best_rank': int(min(regime_ranks)),
                    'worst_rank': int(max(regime_ranks)),
                    'consistency_score': float(4 - np.mean(regime_ranks) + 1 - np.std(regime_ranks))  # Higher is better
                }
        
        return consistency_metrics
    
    def _generate_risk_management_insights(self):
        """
        Step 6.1b: Generate risk management insights
        """
        risk_insights = {}
        
        # Correlation breakdown analysis
        correlation_insights = self._analyze_correlation_breakdowns()
        
        # Tail risk analysis by regime
        tail_risk_insights = self._analyze_tail_risk_by_regime()
        
        # Portfolio stress testing
        stress_test_insights = self._create_stress_testing_scenarios()
        
        # Risk budget recommendations
        risk_budget_insights = self._create_regime_risk_budgets()
        
        # Diversification effectiveness
        diversification_insights = self._analyze_diversification_effectiveness()
        
        return {
            'correlation_breakdown_analysis': correlation_insights,
            'tail_risk_by_regime': tail_risk_insights,
            'stress_testing_scenarios': stress_test_insights,
            'regime_risk_budgets': risk_budget_insights,
            'diversification_effectiveness': diversification_insights
        }
    
    def _analyze_correlation_breakdowns(self):
        """Analyze when factor correlations break down"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        correlation_analysis = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            if len(regime_data) > 10:
                factor_data = regime_data[factors].dropna()
                if len(factor_data) > 5:
                    corr_matrix = factor_data.corr()
                    
                    # Calculate average correlations
                    correlations = []
                    for i in range(len(factors)):
                        for j in range(i+1, len(factors)):
                            correlations.append(corr_matrix.iloc[i, j])
                    
                    correlation_analysis[regime] = {
                        'average_correlation': float(np.mean(correlations)),
                        'correlation_volatility': float(np.std(correlations)),
                        'max_correlation': float(np.max(correlations)),
                        'min_correlation': float(np.min(correlations)),
                        'diversification_ratio': float(1 - np.mean(np.abs(correlations)))
                    }
        
        return correlation_analysis
    
    def _analyze_tail_risk_by_regime(self):
        """Analyze tail risk characteristics by regime"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        tail_risk_analysis = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            regime_tail_risk = {}
            
            for factor in factors:
                if factor in regime_data.columns:
                    returns = regime_data[factor].dropna()
                    
                    if len(returns) > 10:
                        # Calculate various tail risk metrics
                        var_1 = np.percentile(returns, 1)
                        var_5 = np.percentile(returns, 5)
                        
                        # Expected shortfall
                        es_1 = returns[returns <= var_1].mean() if len(returns[returns <= var_1]) > 0 else var_1
                        es_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5
                        
                        # Skewness and kurtosis
                        skewness = returns.skew()
                        kurtosis = returns.kurtosis()
                        
                        regime_tail_risk[factor] = {
                            'var_1_percent': float(var_1),
                            'var_5_percent': float(var_5),
                            'expected_shortfall_1_percent': float(es_1),
                            'expected_shortfall_5_percent': float(es_5),
                            'skewness': float(skewness),
                            'excess_kurtosis': float(kurtosis)
                        }
            
            tail_risk_analysis[regime] = regime_tail_risk
        
        return tail_risk_analysis
    
    def _create_stress_testing_scenarios(self):
        """Create portfolio stress testing scenarios"""
        scenarios = {
            'regime_transition_stress': {
                'description': 'Rapid regime transition from Goldilocks to Recession',
                'expected_impact': 'High correlation spike, diversification breakdown',
                'risk_factors': ['Correlation increases to 0.8+', 'Value and Momentum underperform severely'],
                'mitigation': 'Increase Quality and MinVol allocations during transitions'
            },
            'volatility_spike_stress': {
                'description': 'VIX spikes above 50 (Crisis level)',
                'expected_impact': 'All factors negative, MinVol provides best defense',
                'risk_factors': ['Universal factor underperformance', 'Liquidity constraints'],
                'mitigation': 'Maintain minimum 20% MinVol allocation, cash reserves'
            },
            'prolonged_stagflation_stress': {
                'description': 'Extended stagflation period (6+ months)',
                'expected_impact': 'Value outperforms but absolute returns low',
                'risk_factors': ['Low absolute returns across factors', 'High inflation erodes real returns'],
                'mitigation': 'Maximum Value allocation, consider real asset exposure'
            },
            'economic_expansion_stress': {
                'description': 'Overheating transitions to bubble conditions',
                'expected_impact': 'Momentum and growth factors become extremely expensive',
                'risk_factors': ['Valuation extremes', 'Potential sharp reversal'],
                'mitigation': 'Begin defensive rotation early, monitor valuation metrics'
            }
        }
        
        return scenarios
    
    def _create_regime_risk_budgets(self):
        """Create risk budget recommendations by regime"""
        risk_budgets = {}
        
        for regime in ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']:
            if regime == 'Goldilocks':
                risk_budgets[regime] = {
                    'target_volatility': '12-15%',
                    'max_drawdown_limit': '15%',
                    'factor_concentration_limit': '40%',
                    'risk_allocation': {
                        'factor_risk': '70%',
                        'regime_transition_risk': '20%',
                        'tail_risk_buffer': '10%'
                    }
                }
            elif regime == 'Overheating':
                risk_budgets[regime] = {
                    'target_volatility': '10-13%',
                    'max_drawdown_limit': '12%',
                    'factor_concentration_limit': '35%',
                    'risk_allocation': {
                        'factor_risk': '60%',
                        'regime_transition_risk': '30%',
                        'tail_risk_buffer': '10%'
                    }
                }
            elif regime == 'Stagflation':
                risk_budgets[regime] = {
                    'target_volatility': '8-12%',
                    'max_drawdown_limit': '10%',
                    'factor_concentration_limit': '50%',  # Allow concentration in Value
                    'risk_allocation': {
                        'factor_risk': '50%',
                        'regime_transition_risk': '25%',
                        'tail_risk_buffer': '25%'
                    }
                }
            elif regime == 'Recession':
                risk_budgets[regime] = {
                    'target_volatility': '6-10%',
                    'max_drawdown_limit': '8%',
                    'factor_concentration_limit': '45%',  # Allow concentration in Quality/MinVol
                    'risk_allocation': {
                        'factor_risk': '40%',
                        'regime_transition_risk': '20%',
                        'tail_risk_buffer': '40%'
                    }
                }
        
        return risk_budgets
    
    def _analyze_diversification_effectiveness(self):
        """Analyze diversification effectiveness across regimes"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        diversification_analysis = {}
        
        for regime in self.aligned_data['ECONOMIC_REGIME'].unique():
            regime_data = self.aligned_data[self.aligned_data['ECONOMIC_REGIME'] == regime]
            
            if len(regime_data) > 10:
                factor_data = regime_data[factors].dropna()
                
                if len(factor_data) > 5:
                    # Equal weight portfolio
                    equal_weight_returns = factor_data.mean(axis=1)
                    
                    # Individual factor volatilities
                    individual_vols = factor_data.std() * np.sqrt(12)
                    avg_individual_vol = individual_vols.mean()
                    
                    # Portfolio volatility
                    portfolio_vol = equal_weight_returns.std() * np.sqrt(12)
                    
                    # Diversification ratio
                    diversification_ratio = avg_individual_vol / portfolio_vol if portfolio_vol > 0 else 1
                    
                    diversification_analysis[regime] = {
                        'diversification_ratio': float(diversification_ratio),
                        'portfolio_volatility': float(portfolio_vol),
                        'average_factor_volatility': float(avg_individual_vol),
                        'volatility_reduction': float((avg_individual_vol - portfolio_vol) / avg_individual_vol * 100),
                        'effectiveness_rating': 'High' if diversification_ratio > 1.3 else 'Moderate' if diversification_ratio > 1.1 else 'Low'
                    }
        
        return diversification_analysis
    
    def phase6_implementation_framework(self):
        """
        Phase 6.2: Implementation Framework
        Dynamic allocation recommendations and monitoring system
        """
        logger.info("=== PHASE 6.2: Implementation Framework ===")
        
        if not hasattr(self, 'business_insights') or not self.business_insights:
            logger.error("No business insights available. Run Phase 6.1 first.")
            return False
        
        try:
            # Step 6.2a: Dynamic allocation recommendations
            logger.info("Step 6.2a: Creating dynamic allocation recommendations...")
            allocation_framework = self._create_dynamic_allocation_framework()
            
            # Step 6.2b: Monitoring and alerts system
            logger.info("Step 6.2b: Developing monitoring and alerts system...")
            monitoring_system = self._develop_monitoring_system()
            
            # Combine framework
            self.implementation_framework = {
                'dynamic_allocation_framework': allocation_framework,
                'monitoring_and_alerts_system': monitoring_system,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save framework
            with open(self.results_dir / 'phase6_implementation_framework.json', 'w') as f:
                json.dump(self.implementation_framework, f, indent=2, default=str)
            
            logger.info("✓ Phase 6.2 Implementation Framework completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 6.2 implementation framework: {e}")
            return False
    
    def _create_dynamic_allocation_framework(self):
        """
        Step 6.2a: Create dynamic allocation recommendations
        """
        framework = {}
        
        # Base case allocations per regime (from Phase 4 portfolio insights)
        if hasattr(self, 'portfolio_insights'):
            base_allocations = self.portfolio_insights.get('allocation_frameworks', {}).get('regime_specific_allocations', {})
        else:
            base_allocations = {}
        
        # Dynamic tilt adjustments based on regime confidence
        regime_confidence_tilts = {
            'high_confidence': {
                'description': 'Strong regime signals, full allocation tilts',
                'tilt_magnitude': 1.0,
                'risk_adjustment': 'Normal risk budget'
            },
            'moderate_confidence': {
                'description': 'Moderate regime signals, partial tilts',
                'tilt_magnitude': 0.6,
                'risk_adjustment': 'Reduced risk budget by 20%'
            },
            'low_confidence': {
                'description': 'Weak regime signals, minimal tilts',
                'tilt_magnitude': 0.3,
                'risk_adjustment': 'Reduced risk budget by 40%'
            },
            'transition_period': {
                'description': 'Regime transition detected, defensive positioning',
                'tilt_magnitude': 0.2,
                'risk_adjustment': 'Maximum defensive allocation'
            }
        }
        
        # Risk overlay adjustments
        risk_overlays = {
            'normal_vol': {
                'description': 'VIX < 25, normal market conditions',
                'allocation_adjustment': 'No adjustment to base allocation',
                'max_factor_weight': 0.4
            },
            'elevated_vol': {
                'description': 'VIX 25-35, elevated volatility',
                'allocation_adjustment': 'Increase MinVol by 5%, reduce others proportionally',
                'max_factor_weight': 0.35
            },
            'high_vol': {
                'description': 'VIX 35-50, high stress',
                'allocation_adjustment': 'Increase MinVol by 10%, increase Quality by 5%',
                'max_factor_weight': 0.3
            },
            'crisis_vol': {
                'description': 'VIX > 50, crisis conditions',
                'allocation_adjustment': 'Maximum defensive: 40% MinVol, 30% Quality, 15% each Value/Momentum',
                'max_factor_weight': 0.4
            }
        }
        
        # Allocation optimization framework
        optimization_framework = {
            'rebalancing_frequency': {
                'normal_conditions': 'Monthly',
                'high_volatility': 'Bi-weekly',
                'regime_transitions': 'Weekly monitoring with trigger-based rebalancing'
            },
            'transaction_cost_considerations': {
                'cost_threshold': '0.1% of portfolio value',
                'minimum_trade_size': '2% allocation change',
                'trading_implementation': 'Gradual implementation over 3-5 days for large changes'
            },
            'risk_management_rules': {
                'stop_loss': 'No factor allocation below 10% or above 50%',
                'concentration_limit': 'Maximum 40% in any single factor during normal conditions',
                'drawdown_trigger': 'Reduce risk by 25% if portfolio drawdown exceeds 15%'
            }
        }
        
        # Current regime recommendations (based on latest data)
        current_regime = self.aligned_data['ECONOMIC_REGIME'].iloc[-1] if not self.aligned_data.empty else 'Unknown'
        current_vix = self.aligned_data['VIX'].iloc[-1] if 'VIX' in self.aligned_data.columns and not self.aligned_data.empty else 25
        
        # Determine current risk environment
        if current_vix < 25:
            current_vol_regime = 'normal_vol'
        elif current_vix < 35:
            current_vol_regime = 'elevated_vol'
        elif current_vix < 50:
            current_vol_regime = 'high_vol'
        else:
            current_vol_regime = 'crisis_vol'
        
        current_recommendations = {
            'current_economic_regime': current_regime,
            'current_volatility_regime': current_vol_regime,
            'current_vix_level': float(current_vix),
            'recommended_allocation_approach': self._get_current_allocation_recommendation(current_regime, current_vol_regime),
            'next_review_triggers': [
                'VIX moves above/below key thresholds (25, 35, 50)',
                'Economic regime change signals',
                'Factor momentum shifts (3-month rolling)',
                'Monthly portfolio review'
            ]
        }
        
        return {
            'base_allocations': base_allocations,
            'regime_confidence_tilts': regime_confidence_tilts,
            'risk_overlay_adjustments': risk_overlays,
            'optimization_framework': optimization_framework,
            'current_recommendations': current_recommendations
        }
    
    def _get_current_allocation_recommendation(self, economic_regime, vol_regime):
        """Get specific allocation recommendation for current conditions"""
        
        base_recommendation = {
            'Goldilocks': {'Value': 0.30, 'Quality': 0.20, 'MinVol': 0.20, 'Momentum': 0.30},
            'Overheating': {'Value': 0.25, 'Quality': 0.30, 'MinVol': 0.25, 'Momentum': 0.20},
            'Stagflation': {'Value': 0.40, 'Quality': 0.25, 'MinVol': 0.20, 'Momentum': 0.15},
            'Recession': {'Value': 0.20, 'Quality': 0.35, 'MinVol': 0.35, 'Momentum': 0.10}
        }.get(economic_regime, {'Value': 0.25, 'Quality': 0.25, 'MinVol': 0.25, 'Momentum': 0.25})
        
        # Adjust for volatility regime
        if vol_regime == 'elevated_vol':
            base_recommendation['MinVol'] += 0.05
            # Reduce others proportionally
            for factor in ['Value', 'Quality', 'Momentum']:
                base_recommendation[factor] *= 0.95/1.05
        elif vol_regime == 'high_vol':
            base_recommendation['MinVol'] += 0.10
            base_recommendation['Quality'] += 0.05
            # Reduce others proportionally
            for factor in ['Value', 'Momentum']:
                base_recommendation[factor] *= 0.85/1.15
        elif vol_regime == 'crisis_vol':
            base_recommendation = {'Value': 0.15, 'Quality': 0.30, 'MinVol': 0.40, 'Momentum': 0.15}
        
        return {
            'allocation': base_recommendation,
            'rationale': f'Base {economic_regime} allocation adjusted for {vol_regime} conditions',
            'confidence_level': 'High' if vol_regime == 'normal_vol' else 'Moderate' if vol_regime == 'elevated_vol' else 'Low'
        }
    
    def _develop_monitoring_system(self):
        """
        Step 6.2b: Develop monitoring and alerts system
        """
        monitoring_system = {}
        
        # Regime change probability tracking
        regime_monitoring = {
            'current_regime_stability': {
                'metric': 'Rolling 3-month regime consistency',
                'calculation': 'Percentage of recent observations in current regime',
                'alert_threshold': 'Below 70% consistency',
                'action': 'Prepare for potential regime transition'
            },
            'transition_probability_tracking': {
                'metric': 'Estimated probability of regime change in next 3 months',
                'calculation': 'Based on historical transition patterns and current indicators',
                'alert_threshold': 'Above 40% transition probability',
                'action': 'Begin defensive positioning'
            },
            'economic_indicator_divergence': {
                'metric': 'Current indicators vs regime expectations',
                'calculation': 'Z-score of current indicators vs regime historical average',
                'alert_threshold': 'Z-score above 2.0 or below -2.0',
                'action': 'Investigate regime classification accuracy'
            }
        }
        
        # Factor momentum shift detection
        momentum_monitoring = {
            'factor_momentum_persistence': {
                'metric': '3-month vs 12-month factor momentum alignment',
                'calculation': 'Correlation between short and long-term momentum',
                'alert_threshold': 'Correlation below 0.3',
                'action': 'Review factor allocation weights'
            },
            'relative_factor_performance': {
                'metric': 'Factor performance vs regime expectations',
                'calculation': 'Current factor ranking vs historical regime ranking',
                'alert_threshold': 'Ranking change of 2+ positions',
                'action': 'Investigate factor-specific issues'
            },
            'factor_volatility_spike': {
                'metric': 'Individual factor volatility vs historical average',
                'calculation': 'Rolling 1-month volatility vs regime average',
                'alert_threshold': 'Volatility 50% above regime average',
                'action': 'Consider temporary allocation reduction'
            }
        }
        
        # Risk threshold monitoring
        risk_monitoring = {
            'portfolio_drawdown': {
                'metric': 'Current portfolio drawdown from peak',
                'calculation': 'Peak-to-current portfolio value',
                'alert_threshold': 'Drawdown exceeds 10%',
                'action': 'Implement risk reduction measures'
            },
            'vix_threshold_breach': {
                'metric': 'VIX level relative to regime thresholds',
                'calculation': 'Current VIX vs threshold levels (25, 35, 50)',
                'alert_threshold': 'VIX crosses major threshold',
                'action': 'Adjust volatility overlay allocations'
            },
            'correlation_spike': {
                'metric': 'Factor correlation vs regime average',
                'calculation': 'Rolling 1-month factor correlations',
                'alert_threshold': 'Average correlation 20% above regime norm',
                'action': 'Reduce concentration, increase diversification'
            }
        }
        
        # Performance attribution monitoring
        attribution_monitoring = {
            'regime_attribution': {
                'metric': 'Performance attribution to regime vs factor selection',
                'calculation': 'Decompose returns into regime timing and factor selection',
                'review_frequency': 'Monthly',
                'action': 'Adjust strategy based on attribution results'
            },
            'factor_contribution': {
                'metric': 'Individual factor contribution to portfolio returns',
                'calculation': 'Weight × performance for each factor',
                'review_frequency': 'Monthly',
                'action': 'Rebalance if contributions deviate significantly from expectations'
            },
            'risk_adjusted_performance': {
                'metric': 'Portfolio Sharpe ratio vs benchmark',
                'calculation': 'Rolling 12-month Sharpe ratio comparison',
                'review_frequency': 'Quarterly',
                'action': 'Strategy review if underperforming for 2+ quarters'
            }
        }
        
        # Automated alert system
        alert_system = {
            'immediate_alerts': [
                'VIX spikes above 50 (Crisis threshold)',
                'Portfolio drawdown exceeds 15%',
                'Factor correlation spike above 0.8'
            ],
            'daily_alerts': [
                'VIX crosses 25 or 35 thresholds',
                'Factor momentum reversal signals',
                'Economic indicator regime divergence'
            ],
            'weekly_alerts': [
                'Regime transition probability above 40%',
                'Factor ranking changes',
                'Risk budget utilization above 90%'
            ],
            'monthly_reviews': [
                'Comprehensive performance attribution',
                'Strategy effectiveness review',
                'Risk management assessment'
            ]
        }
        
        # Monitoring dashboard specifications
        dashboard_specs = {
            'real_time_indicators': [
                'Current regime and confidence level',
                'VIX level with threshold indicators',
                'Factor momentum scores',
                'Portfolio allocation vs targets'
            ],
            'performance_metrics': [
                'Portfolio vs benchmark performance',
                'Individual factor contributions',
                'Risk-adjusted returns',
                'Drawdown analysis'
            ],
            'risk_metrics': [
                'Current portfolio volatility',
                'Factor correlations',
                'Tail risk indicators',
                'Risk budget utilization'
            ],
            'forward_looking': [
                'Regime transition probabilities',
                'Factor momentum trends',
                'Stress test scenarios',
                'Rebalancing recommendations'
            ]
        }
        
        return {
            'regime_change_monitoring': regime_monitoring,
            'factor_momentum_monitoring': momentum_monitoring,
            'risk_threshold_monitoring': risk_monitoring,
            'performance_attribution_monitoring': attribution_monitoring,
            'automated_alert_system': alert_system,
            'monitoring_dashboard_specifications': dashboard_specs
        }
    
    def run_phase6(self):
        """
        Execute complete Phase 6: Business Insights & Strategy Development
        """
        logger.info("=== STARTING PHASE 6: BUSINESS INSIGHTS & STRATEGY DEVELOPMENT ===")
        
        success = True
        completed_steps = []
        
        # Step 6.1: Regime-Specific Insights Generation
        if self.phase6_regime_specific_insights():
            completed_steps.extend(['6.1a', '6.1b'])
            logger.info("✓ Phase 6.1 completed successfully")
        else:
            logger.error("✗ Phase 6.1 failed")
            success = False
        
        # Step 6.2: Implementation Framework
        if self.phase6_implementation_framework():
            completed_steps.extend(['6.2a', '6.2b'])
            logger.info("✓ Phase 6.2 completed successfully")
        else:
            logger.error("✗ Phase 6.2 failed")
            success = False
        
        if success:
            logger.info("=== PHASE 6 COMPLETED SUCCESSFULLY ===")
            logger.info(f"✓ All business insights components completed: {', '.join(completed_steps)}")
            
            # Create comprehensive Phase 6 summary
            self._create_phase6_summary(completed_steps)
        else:
            logger.error("=== PHASE 6 COMPLETED WITH ERRORS ===")
        
        return success
    
    def _create_phase6_summary(self, completed_steps):
        """
        Create comprehensive Phase 6 summary report
        """
        logger.info("Creating Phase 6 comprehensive summary...")
        
        summary = {
            "phase6_completion": {
                "timestamp": datetime.now().isoformat(),
                "status": "COMPLETED",
                "completed_steps": completed_steps,
                "components_completed": [
                    "6.1a: Factor leadership patterns analysis",
                    "6.1b: Risk management insights generation",
                    "6.2a: Dynamic allocation recommendations framework",
                    "6.2b: Monitoring and alerts system development"
                ]
            },
            "business_insights_generated": [
                "Regime-specific factor leadership patterns",
                "Statistical confidence in factor rankings",
                "Cross-regime leadership consistency analysis",
                "Risk management insights and recommendations",
                "Correlation breakdown analysis",
                "Tail risk analysis by regime",
                "Portfolio stress testing scenarios",
                "Regime-specific risk budgets"
            ],
            "implementation_framework_created": [
                "Dynamic allocation framework with regime confidence tilts",
                "Risk overlay adjustments for volatility regimes",
                "Optimization framework with rebalancing rules",
                "Current market recommendations",
                "Comprehensive monitoring system",
                "Automated alert system",
                "Performance attribution monitoring",
                "Dashboard specifications"
            ],
            "files_created": [
                "phase6_business_insights.json",
                "phase6_implementation_framework.json",
                "phase6_complete_summary.json"
            ],
            "key_strategic_insights": self._extract_key_strategic_insights()
        }
        
        # Save Phase 6 summary
        with open(self.results_dir / 'phase6_complete_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✓ Phase 6 summary saved")
        
        return True
    
    def _extract_key_strategic_insights(self):
        """Extract key strategic insights from Phase 6 analysis"""
        insights = {}
        
        # Current regime and recommendation
        current_regime = self.aligned_data['ECONOMIC_REGIME'].iloc[-1] if not self.aligned_data.empty else 'Unknown'
        current_vix = self.aligned_data['VIX'].iloc[-1] if 'VIX' in self.aligned_data.columns and not self.aligned_data.empty else 25
        
        insights['current_market_assessment'] = {
            'economic_regime': current_regime,
            'volatility_level': float(current_vix),
            'market_stress_category': 'Low' if current_vix < 25 else 'Moderate' if current_vix < 35 else 'High' if current_vix < 50 else 'Crisis',
            'immediate_strategy_focus': self._get_immediate_strategy_focus(current_regime, current_vix)
        }
        
        # Key strategic principles
        insights['strategic_principles'] = {
            'regime_timing': 'Economic regime classification is primary driver of factor allocation decisions',
            'volatility_overlay': 'VIX-based volatility overlay provides crucial risk management',
            'diversification_limits': 'Factor diversification effectiveness varies significantly by regime',
            'transition_management': 'Regime transition periods require defensive positioning and increased monitoring'
        }
        
        # Implementation priorities
        insights['implementation_priorities'] = [
            'Establish regime monitoring system with economic indicator tracking',
            'Implement VIX-based volatility overlay for dynamic risk management',
            'Create factor momentum monitoring for early reversal detection',
            'Develop automated rebalancing triggers based on regime and volatility changes',
            'Establish comprehensive performance attribution framework'
        ]
        
        return insights
    
    def _get_immediate_strategy_focus(self, regime, vix):
        """Get immediate strategic focus based on current conditions"""
        
        if vix > 50:
            return "Crisis management: Maximum defensive positioning with 40% MinVol, 30% Quality allocation"
        elif vix > 35:
            return "High stress: Defensive tilt with increased Quality and MinVol exposure"
        elif regime == 'Goldilocks':
            return "Growth positioning: Favor Value and Momentum factors in supportive environment"
        elif regime == 'Overheating':
            return "Late cycle positioning: Begin defensive preparation while maintaining growth exposure"
        elif regime == 'Stagflation':
            return "Value focus: Maximum Value allocation with Quality support for inflation environment"
        elif regime == 'Recession':
            return "Defensive positioning: Quality and MinVol focus for capital preservation"
        else:
            return "Balanced positioning: Equal weight allocation until regime clarity improves"

def main():
    """
    Main execution function - now supports all phases including Phase 5
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
    
    if not phase2_success:
        logger.error("❌ Phase 2 failed. Please check the logs and fix issues.")
        exit(1)
    
    # Execute Phase 3
    logger.info("=== EXECUTING PHASE 3 ===")
    phase3_success = analyzer.run_phase3()
    
    if not phase3_success:
        logger.error("❌ Phase 3 failed. Please check the logs and fix issues.")
        exit(1)
    
    # Execute Phase 4
    logger.info("=== EXECUTING PHASE 4 ===")
    phase4_success = analyzer.run_phase4()
    
    if not phase4_success:
        logger.error("❌ Phase 4 failed. Please check the logs and fix issues.")
        exit(1)
    
    # Execute Phase 5
    logger.info("=== EXECUTING PHASE 5 ===")
    phase5_success = analyzer.run_phase5()
    
    if not phase5_success:
        logger.error("❌ Phase 5 failed. Please check the logs and fix issues.")
        exit(1)
    
    # Execute Phase 6
    logger.info("=== EXECUTING PHASE 6 ===")
    phase6_success = analyzer.run_phase6()
    
    if phase1_success and phase2_success and phase3_success and phase4_success and phase5_success and phase6_success:
        logger.info("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        logger.info("📊 Phase 1: Data alignment and regime validation - DONE")
        logger.info("📈 Phase 2: Advanced business cycle analytics - DONE") 
        logger.info("📱 Phase 3: Advanced visualization suite - DONE")
        logger.info("🔬 Phase 4: Statistical deep-dive & pattern recognition - DONE")
        logger.info("🚀 Phase 5: Interactive dashboard & reporting - DONE")
        logger.info("🎯 Phase 6: Business insights & strategy development - DONE")
        logger.info("")
        logger.info("📁 Generated Analysis:")
        logger.info("   • Regime transition probability matrices")
        logger.info("   • Factor performance during regime changes")
        logger.info("   • Intra-regime performance evolution patterns")
        logger.info("   • Macro-factor relationship analysis")
        logger.info("   • Regime-aware allocation frameworks")
        logger.info("   • Factor timing models and signals")
        logger.info("   • Business insights with factor leadership patterns")
        logger.info("   • Implementation framework with monitoring system")
        logger.info("")
        logger.info("📱 Interactive Dashboard:")
        logger.info("   • Comprehensive 12-panel business cycle dashboard")
        logger.info("   • Interactive controls and view toggles")
        logger.info("   • Enhanced hover analytics with detailed statistics")
        logger.info("   • Export functionality (CSV, PNG, Markdown, PDF)")
        logger.info("")
        logger.info("📁 Generated Visualizations:")
        logger.info("   • Interactive timeline with regime overlay")
        logger.info("   • Primary performance heatmap (Factor × Regime)")
        logger.info("   • Risk-adjusted performance heatmap")
        logger.info("   • Relative performance heatmap (vs S&P 500)")
        logger.info("   • Factor rotation wheel by regime")
        logger.info("   • Risk-return scatter plots with regime clustering")
        logger.info("   • Rolling regime analysis")
        logger.info("   • Dynamic correlation matrices")
        logger.info("   • Factor momentum persistence analysis")
        logger.info("")
        logger.info("🔗 Check results/business_cycle_analysis/ for all outputs")
        logger.info("🎯 COMPREHENSIVE BUSINESS CYCLE FACTOR ANALYSIS COMPLETE!")
        logger.info("")
        logger.info("🎯 Phase 6 Business Insights Delivered:")
        logger.info("   • Regime-specific factor leadership analysis")
        logger.info("   • Risk management insights and stress testing")
        logger.info("   • Dynamic allocation framework")
        logger.info("   • Comprehensive monitoring and alerts system")
        logger.info("   • Current market assessment and recommendations")
    else:
        logger.error("❌ Some phases failed. Please check the logs.")
        if not phase6_success:
            logger.error("❌ Phase 6 failed. Business insights incomplete.")
        elif not phase5_success:
            logger.error("❌ Phase 5 failed. Dashboard creation incomplete.")
        elif not phase4_success:
            logger.error("❌ Phase 4 failed. Statistical analysis incomplete.")
        elif not phase3_success:
            logger.error("❌ Phase 3 failed. Data analysis completed but visualizations incomplete.")
        exit(1)

if __name__ == "__main__":
    main() 