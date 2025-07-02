"""
Comprehensive Individual Substep Demos for Phase 2
Demonstrates each substep with detailed examples and outputs
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

class Phase2SubstepDemos:
    """
    Individual demos for each Phase 2 substep
    """
    
    def __init__(self):
        self.analyzer = BusinessCycleFactorAnalyzer()
        self.results_dir = Path("results/business_cycle_analysis")
        
        # Initialize analyzer
        self.analyzer.load_data()
        self.analyzer.run_phase1()
        
        logger.info("Phase 2 Substep Demos initialized")
    
    def demo_step_2_1a_requirement_1_average_regime_length(self):
        """
        DEMO: Step 2.1a Req 1 - Calculate average regime length by type
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 1 - Average Regime Length Calculation")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📊 AVERAGE REGIME DURATION BY TYPE:")
        print("-" * 50)
        
        for regime, stats in regime_analysis['regime_statistics'].items():
            print(f"{regime:12} | {stats['avg_duration_months']:6.2f} months | {stats['total_periods']:2d} periods | {stats['frequency_percentage']:5.1f}% frequency")
        
        print(f"\n🔍 DETAILED DURATION STATISTICS:")
        print("-" * 50)
        for regime, stats in regime_analysis['regime_statistics'].items():
            print(f"\n{regime}:")
            print(f"  • Average Duration: {stats['avg_duration_months']:.2f} months")
            print(f"  • Median Duration:  {stats['median_duration_months']:.2f} months")
            print(f"  • Min Duration:     {stats['min_duration_months']:.2f} months")
            print(f"  • Max Duration:     {stats['max_duration_months']:.2f} months")
            print(f"  • Standard Dev:     {stats['std_duration_months']:.2f} months")
    
    def demo_step_2_1a_requirement_2_decade_transitions(self):
        """
        DEMO: Step 2.1a Req 2 - Analyze regime transitions per decade
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 2 - Regime Transitions by Decade")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📈 REGIME TRANSITIONS BY DECADE:")
        print("-" * 40)
        
        for decade, count in regime_analysis['decade_transitions'].items():
            print(f"{decade:8} | {count:3d} transitions")
        
        total_transitions = sum(regime_analysis['decade_transitions'].values())
        print(f"\nTotal Transitions (1998-2025): {total_transitions}")
        
        # Calculate transition intensity
        print(f"\n🔍 TRANSITION INTENSITY ANALYSIS:")
        print("-" * 40)
        for decade, count in regime_analysis['decade_transitions'].items():
            intensity = count / 10  # transitions per year
            print(f"{decade}: {intensity:.1f} transitions/year")
    
    def demo_step_2_1a_requirement_3_stability_metrics(self):
        """
        DEMO: Step 2.1a Req 3 - Compute regime stability metrics
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 3 - Regime Stability Metrics")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📊 REGIME STABILITY ANALYSIS:")
        print("-" * 60)
        print("Regime       | Avg Duration | Std Dev | Stability Score")
        print("-" * 60)
        
        for regime, stats in regime_analysis['regime_statistics'].items():
            # Calculate stability score (higher = more stable)
            stability_score = stats['avg_duration_months'] / stats['std_duration_months'] if stats['std_duration_months'] > 0 else 0
            print(f"{regime:12} | {stats['avg_duration_months']:9.2f} | {stats['std_duration_months']:7.2f} | {stability_score:13.2f}")
        
        print(f"\n🔍 STABILITY INSIGHTS:")
        print("-" * 40)
        print("• Higher stability score = more predictable regime duration")
        print("• Lower standard deviation = more consistent duration")
    
    def demo_step_2_1a_requirement_4_transition_matrices(self):
        """
        DEMO: Step 2.1a Req 4 - Calculate transition probability matrices
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 4 - Transition Probability Matrices")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📊 TRANSITION PROBABILITY MATRIX:")
        print("-" * 70)
        
        # Create formatted transition matrix
        transition_probs = regime_analysis['transition_probabilities']
        regimes = list(transition_probs.keys())
        
        # Print header
        print(f"{'From/To':<12}", end="")
        for regime in regimes:
            print(f"{regime:<15}", end="")
        print()
        print("-" * 70)
        
        # Print matrix
        for from_regime in regimes:
            print(f"{from_regime:<12}", end="")
            for to_regime in regimes:
                prob = transition_probs[from_regime].get(to_regime, 0)
                print(f"{prob:<15.1%}", end="")
            print()
        
        print(f"\n🔍 KEY TRANSITION PATTERNS:")
        print("-" * 40)
        
        # Find highest probability transitions
        high_prob_transitions = []
        for from_regime, transitions in transition_probs.items():
            for to_regime, prob in transitions.items():
                if from_regime != to_regime and prob > 0.3:  # 30%+ probability
                    high_prob_transitions.append((from_regime, to_regime, prob))
        
        high_prob_transitions.sort(key=lambda x: x[2], reverse=True)
        
        for from_regime, to_regime, prob in high_prob_transitions[:5]:
            print(f"• {from_regime} → {to_regime}: {prob:.1%} probability")
    
    def demo_step_2_1a_requirement_5_seasonal_patterns(self):
        """
        DEMO: Step 2.1a Req 5 - Identify seasonal patterns in regime changes
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 5 - Seasonal Regime Change Patterns")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📅 REGIME TRANSITIONS BY MONTH:")
        print("-" * 50)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        seasonal_patterns = regime_analysis['seasonal_transition_patterns']
        
        for month_num in range(1, 13):
            month_name = month_names[month_num - 1]
            transitions = seasonal_patterns.get(month_num, 0)
            print(f"{month_name}: {transitions:2d} transitions")
        
        # Find peak and low months
        max_month = max(seasonal_patterns, key=seasonal_patterns.get)
        min_month = min(seasonal_patterns, key=seasonal_patterns.get)
        
        print(f"\n🔍 SEASONAL ANALYSIS:")
        print("-" * 30)
        print(f"• Peak Month: {month_names[max_month-1]} ({seasonal_patterns[max_month]} transitions)")
        print(f"• Low Month:  {month_names[min_month-1]} ({seasonal_patterns[min_month]} transitions)")
        print(f"• Range: {seasonal_patterns[max_month] - seasonal_patterns[min_month]} transition difference")
    
    def demo_step_2_1a_requirement_6_summary_statistics(self):
        """
        DEMO: Step 2.1a Req 6 - Create regime summary statistics table
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1a Requirement 6 - Regime Summary Statistics Table")
        print("="*80)
        
        regime_analysis = self.analyzer._analyze_regime_durations_and_transitions()
        
        print("\n📊 COMPREHENSIVE REGIME SUMMARY TABLE:")
        print("-" * 100)
        print(f"{'Regime':<12} | {'Periods':<7} | {'Avg Dur':<8} | {'Total Mon':<9} | {'Frequency':<9} | {'Min Dur':<7} | {'Max Dur':<7}")
        print("-" * 100)
        
        for regime, stats in regime_analysis['regime_statistics'].items():
            print(f"{regime:<12} | {stats['total_periods']:7d} | {stats['avg_duration_months']:8.2f} | {stats['total_months']:9.1f} | {stats['frequency_percentage']:8.1f}% | {stats['min_duration_months']:7.2f} | {stats['max_duration_months']:7.2f}")
        
        # Summary totals
        total_periods = sum(stats['total_periods'] for stats in regime_analysis['regime_statistics'].values())
        total_months = sum(stats['total_months'] for stats in regime_analysis['regime_statistics'].values())
        
        print("-" * 100)
        print(f"{'TOTAL':<12} | {total_periods:7d} | {'N/A':<8} | {total_months:9.1f} | {'100.0%':<9} | {'N/A':<7} | {'N/A':<7}")
    
    def demo_step_2_1b_economic_validation(self):
        """
        DEMO: Step 2.1b - Economic signal validation per regime (all requirements)
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.1b - Economic Signal Validation per Regime")
        print("="*80)
        
        economic_validation = self.analyzer._validate_economic_signals_by_regime()
        
        print("\n🏛️ ECONOMIC INDICATORS BY REGIME:")
        print("-" * 60)
        
        for regime, validation in economic_validation['regime_validations'].items():
            print(f"\n{regime.upper()} REGIME ({validation['observations']} observations):")
            print("-" * 40)
            
            # GDP/Growth indicators
            if 'GROWTH_COMPOSITE' in validation:
                print(f"Growth Composite: {validation['GROWTH_COMPOSITE']['mean']:6.3f} (avg)")
            
            # Inflation indicators  
            if 'INFLATION_COMPOSITE' in validation:
                print(f"Inflation Composite: {validation['INFLATION_COMPOSITE']['mean']:6.3f} (avg)")
            
            # Yield curve
            if 'DGS10' in validation:
                print(f"10Y Treasury: {validation['DGS10']['mean']:6.2f}% (avg)")
            if 'T10Y2Y' in validation:
                print(f"Yield Curve Spread: {validation['T10Y2Y']['mean']:6.2f}% (avg)")
            
            # Employment
            if 'UNRATE' in validation:
                print(f"Unemployment Rate: {validation['UNRATE']['mean']:6.2f}% (avg)")
            
            # Market stress
            if 'VIX' in validation:
                print(f"VIX: {validation['VIX']['mean']:6.2f} ({validation['VIX']['volatility_profile']} volatility)")
        
        print(f"\n🔍 CROSS-REGIME ECONOMIC COMPARISONS:")
        print("-" * 50)
        
        for indicator, regime_values in economic_validation['cross_regime_comparisons'].items():
            print(f"\n{indicator}:")
            sorted_regimes = sorted(regime_values.items(), key=lambda x: x[1], reverse=True)
            for regime, value in sorted_regimes:
                print(f"  {regime}: {value:8.3f}")
    
    def demo_step_2_2a_performance_metrics(self):
        """
        DEMO: Step 2.2a - Comprehensive performance metrics (all requirements)
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.2a - Comprehensive Performance Metrics by Regime")
        print("="*80)
        
        performance_metrics = self.analyzer._calculate_comprehensive_performance_metrics()
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        for regime in performance_metrics:
            print(f"\n🚀 {regime.upper()} REGIME PERFORMANCE:")
            print("-" * 70)
            print(f"{'Factor':<10} | {'Ann Ret':<8} | {'Sharpe':<7} | {'Sortino':<8} | {'Max DD':<8} | {'Win Rate':<8}")
            print("-" * 70)
            
            for factor in factors:
                if factor in performance_metrics[regime]:
                    metrics = performance_metrics[regime][factor]
                    print(f"{factor:<10} | {metrics['annualized_return']:7.1%} | {metrics['sharpe_ratio']:7.2f} | {metrics['sortino_ratio']:8.2f} | {metrics['max_drawdown']:7.1%} | {metrics['win_rate']:7.1%}")
            
            # Show S&P 500 if available
            if 'SP500_Monthly_Return' in performance_metrics[regime]:
                sp500_metrics = performance_metrics[regime]['SP500_Monthly_Return']
                print(f"{'S&P 500':<10} | {sp500_metrics['annualized_return']:7.1%} | {sp500_metrics['sharpe_ratio']:7.2f} | {sp500_metrics['sortino_ratio']:8.2f} | {sp500_metrics['max_drawdown']:7.1%} | {sp500_metrics['win_rate']:7.1%}")
        
        print(f"\n🔍 RISK METRICS BREAKDOWN:")
        print("-" * 50)
        
        for regime in performance_metrics:
            print(f"\n{regime} - Tail Risk Analysis:")
            for factor in factors:
                if factor in performance_metrics[regime]:
                    metrics = performance_metrics[regime][factor]
                    print(f"  {factor}: VaR(5%)={metrics['var_5_percent']:6.2%}, ES(5%)={metrics['expected_shortfall_5_percent']:6.2%}")
    
    def demo_step_2_2b_statistical_tests(self):
        """
        DEMO: Step 2.2b - Statistical significance testing (all requirements)
        """
        print("\n" + "="*80)
        print("DEMO: Step 2.2b - Statistical Significance Testing")
        print("="*80)
        
        statistical_tests = self.analyzer._run_statistical_significance_tests()
        
        print("\n📊 ANOVA TESTS FOR REGIME DIFFERENCES:")
        print("-" * 60)
        print(f"{'Factor':<15} | {'F-Statistic':<12} | {'P-Value':<10} | {'Significant':<12}")
        print("-" * 60)
        
        for factor, test in statistical_tests['anova_tests'].items():
            if 'f_statistic' in test:
                significant = "Yes" if test['significant'] else "No"
                print(f"{factor:<15} | {test['f_statistic']:12.2f} | {test['p_value']:10.4f} | {significant:<12}")
        
        print(f"\n🔍 BOOTSTRAP CONFIDENCE INTERVALS (Sample - Goldilocks):")
        print("-" * 70)
        
        if 'Goldilocks' in statistical_tests['bootstrap_confidence_intervals']:
            goldilocks_ci = statistical_tests['bootstrap_confidence_intervals']['Goldilocks']
            print(f"{'Factor':<15} | {'Mean':<8} | {'95% CI Lower':<12} | {'95% CI Upper':<12}")
            print("-" * 70)
            
            for factor, ci_data in goldilocks_ci.items():
                if 'mean' in ci_data:
                    print(f"{factor:<15} | {ci_data['mean']:8.3f} | {ci_data['ci_lower']:12.3f} | {ci_data['ci_upper']:12.3f}")
        
        print(f"\n🔄 REGIME TRANSITION IMPACT ANALYSIS:")
        print("-" * 60)
        
        if 'regime_transition_impact' in statistical_tests:
            transition_impact = statistical_tests['regime_transition_impact']
            print(f"{'Factor':<15} | {'Pre-Trans':<10} | {'Post-Trans':<11} | {'Change':<10}")
            print("-" * 60)
            
            for factor, impact in transition_impact.items():
                if 'pre_transition_mean' in impact:
                    change = impact['performance_change']
                    change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
                    print(f"{factor:<15} | {impact['pre_transition_mean']:10.3f} | {impact['post_transition_mean']:11.3f} | {change_str:<10}")
            
            # Show total transitions analyzed
            if transition_impact and 'total_transitions_analyzed' in list(transition_impact.values())[0]:
                total_transitions = list(transition_impact.values())[0]['total_transitions_analyzed']
                print(f"\nTotal transitions analyzed: {total_transitions}")
    
    def run_all_demos(self):
        """
        Run all individual substep demos
        """
        print("🎬 COMPREHENSIVE PHASE 2 INDIVIDUAL SUBSTEP DEMONSTRATIONS")
        print("=" * 80)
        print("Showcasing every requirement from BUSINESS_CYCLE_FACTOR_ANALYSIS_ROADMAP.md")
        print("=" * 80)
        
        # Step 2.1a demos
        self.demo_step_2_1a_requirement_1_average_regime_length()
        self.demo_step_2_1a_requirement_2_decade_transitions()
        self.demo_step_2_1a_requirement_3_stability_metrics()
        self.demo_step_2_1a_requirement_4_transition_matrices()
        self.demo_step_2_1a_requirement_5_seasonal_patterns()
        self.demo_step_2_1a_requirement_6_summary_statistics()
        
        # Step 2.1b demo
        self.demo_step_2_1b_economic_validation()
        
        # Step 2.2a demo
        self.demo_step_2_2a_performance_metrics()
        
        # Step 2.2b demo
        self.demo_step_2_2b_statistical_tests()
        
        print("\n" + "="*80)
        print("🎉 ALL INDIVIDUAL SUBSTEP DEMOS COMPLETED!")
        print("📋 Every roadmap requirement demonstrated with detailed examples")
        print("✅ Phase 2 implementation is comprehensive and ready for Phase 3")
        print("="*80)

def main():
    """
    Main function to run all demos
    """
    demos = Phase2SubstepDemos()
    demos.run_all_demos()

if __name__ == "__main__":
    main() 