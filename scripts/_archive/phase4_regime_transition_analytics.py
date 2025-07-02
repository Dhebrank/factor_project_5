"""
Phase 4: Statistical Deep-Dive & Pattern Recognition
Step 4.1: Regime Transition Analytics
Advanced regime transition modeling and early warning systems
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
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimeTransitionAnalyzer:
    """
    Advanced regime transition analytics and forecasting
    """
    
    def __init__(self, data_dir="data/processed", results_dir="results/business_cycle_analysis"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load aligned data
        self.aligned_data = pd.read_csv(
            self.results_dir / 'aligned_master_dataset_FIXED.csv',
            index_col=0,
            parse_dates=True
        )
        
        # Load performance metrics from Phase 2
        with open(self.results_dir / 'phase2_performance_analysis.json', 'r') as f:
            self.performance_metrics = json.load(f)
        
        # Initialize containers
        self.transition_data = None
        self.transition_probabilities = None
        self.early_warning_signals = None
        
        logger.info("RegimeTransitionAnalyzer initialized")
    
    def phase4_1a_transition_probability_matrix(self):
        """
        Phase 4.1a: Transition Probability Matrix
        Calculate transition frequencies, expected durations, and early warning signals
        """
        logger.info("=== PHASE 4.1a: Transition Probability Matrix ===")
        
        try:
            # Step 4.1a.1: Calculate historical transition frequencies
            logger.info("Step 4.1a.1: Calculating historical regime transition frequencies...")
            transition_frequencies = self._calculate_transition_frequencies()
            
            # Step 4.1a.2: Build expected regime duration models
            logger.info("Step 4.1a.2: Building expected regime duration models...")
            duration_models = self._build_duration_models()
            
            # Step 4.1a.3: Develop early warning signal analysis
            logger.info("Step 4.1a.3: Developing early warning signal analysis...")
            early_warning_signals = self._develop_early_warning_signals()
            
            # Step 4.1a.4: Create transition probability heatmap
            logger.info("Step 4.1a.4: Creating transition probability heatmap...")
            transition_heatmap = self._create_transition_probability_heatmap(transition_frequencies)
            
            # Step 4.1a.5: Add confidence intervals for transition probabilities
            logger.info("Step 4.1a.5: Adding confidence intervals for transition probabilities...")
            confidence_intervals = self._calculate_transition_confidence_intervals(transition_frequencies)
            
            # Step 4.1a.6: Include regime persistence analysis
            logger.info("Step 4.1a.6: Including regime persistence analysis...")
            persistence_analysis = self._analyze_regime_persistence()
            
            # Combine all results
            self.transition_probabilities = {
                'transition_frequencies': transition_frequencies,
                'duration_models': duration_models,
                'early_warning_signals': early_warning_signals,
                'confidence_intervals': confidence_intervals,
                'persistence_analysis': persistence_analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            with open(self.results_dir / 'phase4_transition_probabilities.json', 'w') as f:
                json.dump(self.transition_probabilities, f, indent=2, default=str)
            
            # Save interactive heatmap
            transition_heatmap.write_html(self.results_dir / 'transition_probability_heatmap.html')
            
            logger.info("✓ Phase 4.1a Transition Probability Matrix completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Phase 4.1a: {e}")
            return False
    
    def _calculate_transition_frequencies(self):
        """
        Step 4.1a.1: Calculate historical regime transition frequencies
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        
        # Identify all transitions
        transitions = []
        for i in range(len(regime_col) - 1):
            from_regime = regime_col.iloc[i]
            to_regime = regime_col.iloc[i + 1]
            if from_regime != to_regime:
                transitions.append({
                    'from_regime': from_regime,
                    'to_regime': to_regime,
                    'date': regime_col.index[i + 1],
                    'from_date': regime_col.index[i]
                })
        
        # Build transition count matrix
        regimes = sorted(regime_col.unique())
        transition_counts = pd.DataFrame(0, index=regimes, columns=regimes)
        
        for transition in transitions:
            from_regime = transition['from_regime']
            to_regime = transition['to_regime']
            transition_counts.loc[from_regime, to_regime] += 1
        
        # Calculate transition probabilities
        # Add 1 to avoid division by zero (Laplace smoothing)
        row_sums = transition_counts.sum(axis=1) + len(regimes)
        transition_probabilities = (transition_counts + 1).div(row_sums, axis=0)
        
        # Calculate regime holding periods
        regime_periods = []
        current_regime = None
        start_date = None
        
        for date, regime in regime_col.items():
            if regime != current_regime:
                if current_regime is not None:
                    duration_months = len(regime_col[(regime_col.index >= start_date) & 
                                                   (regime_col.index < date)])
                    regime_periods.append({
                        'regime': current_regime,
                        'start_date': start_date,
                        'end_date': date,
                        'duration_months': duration_months
                    })
                current_regime = regime
                start_date = date
        
        # Handle final period
        if current_regime is not None:
            duration_months = len(regime_col[regime_col.index >= start_date])
            regime_periods.append({
                'regime': current_regime,
                'start_date': start_date,
                'end_date': regime_col.index[-1],
                'duration_months': duration_months
            })
        
        return {
            'transition_counts': transition_counts.to_dict(),
            'transition_probabilities': transition_probabilities.to_dict(),
            'transitions_list': transitions,
            'regime_periods': regime_periods,
            'total_transitions': len(transitions)
        }
    
    def _build_duration_models(self):
        """
        Step 4.1a.2: Build expected regime duration models
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        
        # Calculate duration statistics for each regime
        duration_stats = {}
        
        for regime in regime_col.unique():
            regime_periods = [p for p in self.transition_probabilities['regime_periods'] 
                            if p['regime'] == regime]
            
            if regime_periods:
                durations = [p['duration_months'] for p in regime_periods]
                
                # Fit exponential distribution for duration modeling
                if len(durations) > 1:
                    # MLE estimator for exponential distribution
                    lambda_param = 1.0 / np.mean(durations)
                    
                    # Calculate survival probabilities
                    months = np.arange(1, 25)  # 1-24 months
                    survival_probs = np.exp(-lambda_param * months)
                    
                    duration_stats[regime] = {
                        'count': len(durations),
                        'mean_duration': float(np.mean(durations)),
                        'median_duration': float(np.median(durations)),
                        'std_duration': float(np.std(durations)),
                        'min_duration': int(min(durations)),
                        'max_duration': int(max(durations)),
                        'lambda_parameter': float(lambda_param),
                        'survival_probabilities': {int(m): float(p) for m, p in zip(months, survival_probs)},
                        'expected_remaining_life': float(1 / lambda_param)
                    }
        
        return duration_stats
    
    def _develop_early_warning_signals(self):
        """
        Step 4.1a.3: Develop early warning signal analysis
        """
        # Economic indicators that might predict regime changes
        indicators = [
            'GROWTH_COMPOSITE', 'INFLATION_COMPOSITE', 'DGS10', 'DGS2', 
            'T10Y2Y', 'VIX', 'UNRATE'
        ]
        
        # Filter to available indicators
        available_indicators = [ind for ind in indicators if ind in self.aligned_data.columns]
        
        early_warning_analysis = {}
        
        for indicator in available_indicators:
            indicator_data = self.aligned_data[indicator].dropna()
            
            if len(indicator_data) > 0:
                # Calculate rolling statistics
                rolling_mean = indicator_data.rolling(6).mean()  # 6-month average
                rolling_std = indicator_data.rolling(6).std()    # 6-month volatility
                
                # Z-score for anomaly detection
                z_scores = (indicator_data - rolling_mean) / rolling_std
                
                # Identify extreme values (potential regime change signals)
                extreme_threshold = 2.0  # 2 standard deviations
                extreme_dates = z_scores[abs(z_scores) > extreme_threshold].index
                
                # Check if extreme values precede regime changes
                regime_changes = []
                regime_col = self.aligned_data['ECONOMIC_REGIME']
                
                for i in range(len(regime_col) - 1):
                    if regime_col.iloc[i] != regime_col.iloc[i + 1]:
                        regime_changes.append(regime_col.index[i + 1])
                
                # Calculate lead times between extreme values and regime changes
                lead_times = []
                for change_date in regime_changes:
                    # Look for extreme values in the 6 months before regime change
                    preceding_extremes = extreme_dates[
                        (extreme_dates < change_date) & 
                        (extreme_dates >= change_date - pd.DateOffset(months=6))
                    ]
                    
                    if len(preceding_extremes) > 0:
                        lead_time = (change_date - preceding_extremes[-1]).days
                        lead_times.append(lead_time)
                
                early_warning_analysis[indicator] = {
                    'extreme_values_count': len(extreme_dates),
                    'regime_changes_count': len(regime_changes),
                    'successful_predictions': len(lead_times),
                    'prediction_accuracy': len(lead_times) / len(regime_changes) if regime_changes else 0,
                    'average_lead_time_days': float(np.mean(lead_times)) if lead_times else 0,
                    'median_lead_time_days': float(np.median(lead_times)) if lead_times else 0
                }
        
        return early_warning_analysis
    
    def _create_transition_probability_heatmap(self, transition_frequencies):
        """
        Step 4.1a.4: Create transition probability heatmap
        """
        # Convert transition probabilities to DataFrame for visualization
        probs_df = pd.DataFrame(transition_frequencies['transition_probabilities'])
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=probs_df.values,
            x=probs_df.columns,
            y=probs_df.index,
            colorscale='Blues',
            hovertemplate='<b>%{y} → %{x}</b><br>' +
                         'Probability: %{z:.3f}<br>' +
                         '<extra></extra>',
            colorbar=dict(title="Transition Probability")
        ))
        
        # Add probability values as text
        for i, from_regime in enumerate(probs_df.index):
            for j, to_regime in enumerate(probs_df.columns):
                prob = probs_df.iloc[i, j]
                fig.add_annotation(
                    x=to_regime,
                    y=from_regime,
                    text=f"{prob:.3f}",
                    showarrow=False,
                    font=dict(color="white" if prob > 0.5 else "black")
                )
        
        fig.update_layout(
            title={
                'text': 'Regime Transition Probability Matrix<br><sub>Historical Transition Frequencies (1998-2025)</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=500,
            width=700
        )
        
        return fig
    
    def _calculate_transition_confidence_intervals(self, transition_frequencies):
        """
        Step 4.1a.5: Calculate confidence intervals for transition probabilities
        """
        counts_df = pd.DataFrame(transition_frequencies['transition_counts'])
        probs_df = pd.DataFrame(transition_frequencies['transition_probabilities'])
        
        confidence_intervals = {}
        
        for from_regime in counts_df.index:
            confidence_intervals[from_regime] = {}
            total_transitions = counts_df.loc[from_regime].sum()
            
            for to_regime in counts_df.columns:
                count = counts_df.loc[from_regime, to_regime]
                prob = probs_df.loc[from_regime, to_regime]
                
                if total_transitions > 0:
                    # Wilson score interval for binomial proportions
                    n = total_transitions
                    p = count / n if n > 0 else 0
                    z = 1.96  # 95% confidence
                    
                    if n > 0:
                        denominator = 1 + z**2 / n
                        center = (p + z**2 / (2*n)) / denominator
                        half_width = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
                        
                        lower_bound = max(0, center - half_width)
                        upper_bound = min(1, center + half_width)
                    else:
                        lower_bound = upper_bound = 0
                    
                    confidence_intervals[from_regime][to_regime] = {
                        'point_estimate': float(prob),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'sample_size': int(total_transitions)
                    }
        
        return confidence_intervals
    
    def _analyze_regime_persistence(self):
        """
        Step 4.1a.6: Analyze regime persistence
        """
        regime_col = self.aligned_data['ECONOMIC_REGIME']
        
        persistence_analysis = {}
        
        for regime in regime_col.unique():
            # Calculate average time spent in regime
            regime_mask = regime_col == regime
            regime_periods = []
            
            current_length = 0
            for is_regime in regime_mask:
                if is_regime:
                    current_length += 1
                else:
                    if current_length > 0:
                        regime_periods.append(current_length)
                    current_length = 0
            
            # Handle last period
            if current_length > 0:
                regime_periods.append(current_length)
            
            if regime_periods:
                persistence_analysis[regime] = {
                    'average_duration': float(np.mean(regime_periods)),
                    'median_duration': float(np.median(regime_periods)),
                    'total_periods': len(regime_periods),
                    'total_months': int(sum(regime_periods)),
                    'longest_period': int(max(regime_periods)),
                    'shortest_period': int(min(regime_periods)),
                    'probability_of_persistence': float(regime_mask.sum() / len(regime_mask))
                }
        
        return persistence_analysis
    
    def run_phase4_1a(self):
        """
        Execute Phase 4.1a: Transition Probability Matrix
        """
        logger.info("=== STARTING PHASE 4.1a: TRANSITION PROBABILITY MATRIX ===")
        
        success = self.phase4_1a_transition_probability_matrix()
        
        if success:
            logger.info("✅ PHASE 4.1a COMPLETED SUCCESSFULLY")
            return True
        else:
            logger.error("❌ PHASE 4.1a FAILED")
            return False

def main():
    """
    Main execution function for Phase 4.1a
    """
    analyzer = RegimeTransitionAnalyzer()
    
    logger.info("Starting Phase 4: Statistical Deep-Dive & Pattern Recognition")
    logger.info("Executing Phase 4.1a: Transition Probability Matrix")
    
    success = analyzer.run_phase4_1a()
    
    if success:
        logger.info("🎉 Phase 4.1a completed successfully!")
        logger.info("📊 Generated: transition_probability_heatmap.html")
        logger.info("📄 Generated: phase4_transition_probabilities.json")
        logger.info("🚀 Ready for Phase 4.1b: Performance during regime changes")
    else:
        logger.error("❌ Phase 4.1a failed. Please check logs.")

if __name__ == "__main__":
    main() 