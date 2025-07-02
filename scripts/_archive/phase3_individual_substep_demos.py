"""
Phase 3 Individual Substep Demos
Individual demo and test scripts for each Phase 3 visualization component
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3SubstepDemos:
    """
    Individual demos for each Phase 3 substep
    """
    
    def __init__(self, results_dir="results/business_cycle_analysis"):
        self.results_dir = Path(results_dir)
        
        # Load data
        self.aligned_data = pd.read_csv(
            self.results_dir / 'aligned_master_dataset_FIXED.csv',
            index_col=0,
            parse_dates=True
        )
        
        # Load performance metrics
        with open(self.results_dir / 'phase2_performance_analysis.json', 'r') as f:
            self.performance_metrics = json.load(f)
        
        logger.info("Phase 3 Substep Demos initialized")
    
    def demo_3_1a_interactive_timeline(self):
        """
        Demo 3.1a: Interactive timeline with regime overlay
        """
        logger.info("=== DEMO 3.1a: Interactive Timeline with Regime Overlay ===")
        
        try:
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
            
            # Add regime background rectangles with transition indicators
            for i, period in enumerate(regime_periods):
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
                
                # Add regime transition markers (vertical lines)
                if i < len(regime_periods) - 1:  # Don't add line after last period
                    fig.add_vline(
                        x=period['end'],
                        line_dash="dash",
                        line_color="black",
                        line_width=1,
                        opacity=0.7,
                        annotation_text=f"→ {regime_periods[i+1]['regime']}",
                        annotation_position="top",
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
            
            # Save corrected timeline
            fig.write_html(self.results_dir / 'interactive_timeline_regime_overlay_FIXED.html')
            
            logger.info("✅ Demo 3.1a: Interactive timeline completed successfully")
            logger.info(f"   • Regime transition indicators: {len(regime_periods)} periods with transition markers")
            logger.info(f"   • VIX subplot: Row 2 with thresholds at 25, 35, 50")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo 3.1a failed: {e}")
            return False
    
    def demo_3_2b_risk_adjusted_heatmap(self):
        """
        Demo 3.2b: Risk-adjusted performance heatmap with proper color scale
        """
        logger.info("=== DEMO 3.2b: Risk-Adjusted Performance Heatmap ===")
        
        try:
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
            
            # Create heatmap with proper color scale centered at 0
            fig = go.Figure(data=go.Heatmap(
                z=sharpe_matrix,
                x=regimes,
                y=factor_labels,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text,
                colorscale='RdYlGn',
                colorbar=dict(title="Sharpe Ratio"),
                zmid=0,  # Center colorscale at 0
                zmin=-1,  # Set appropriate range for Sharpe ratios
                zmax=2
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
            
            # Save corrected heatmap
            fig.write_html(self.results_dir / 'risk_adjusted_heatmap_FIXED.html')
            
            logger.info("✅ Demo 3.2b: Risk-adjusted heatmap completed successfully")
            logger.info("   • Color scale: RdYlGn centered at 0 with zmid=0")
            logger.info("   • Range: -1 to 2 for appropriate Sharpe ratio visualization")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo 3.2b failed: {e}")
            return False
    
    def demo_3_3a_factor_rotation_wheel(self):
        """
        Demo 3.3a: Factor rotation wheel with proper subplot structure
        """
        logger.info("=== DEMO 3.3a: Factor Rotation Wheel ===")
        
        try:
            # Extract performance data for each regime
            regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
            factors = ['Value', 'Quality', 'MinVol', 'Momentum']
            
            # Create polar/radar chart for each regime with explicit subplot specifications
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=regimes,
                specs=[[{"type": "polar"}, {"type": "polar"}],
                       [{"type": "polar"}, {"type": "polar"}]],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
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
                            fillcolor='rgba(106, 81, 163, 0.3)',
                            hovertemplate=f'<b>{regime}</b><br>' +
                                        'Factor: %{theta}<br>' +
                                        'Sharpe Ratio: %{r:.2f}<br>' +
                                        '<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # Update polar layout for this subplot
                    fig.update_polars(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(2, max(sharpe_values) * 1.1)]
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
            
            # Save corrected rotation wheel
            fig.write_html(self.results_dir / 'factor_rotation_wheel_FIXED.html')
            
            logger.info("✅ Demo 3.3a: Factor rotation wheel completed successfully")
            logger.info("   • Subplot structure: 2x2 grid with explicit polar specifications")
            logger.info("   • Regime subplots: All 4 regimes with proper polar charts")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo 3.3a failed: {e}")
            return False
    
    def demo_3_4b_momentum_persistence(self):
        """
        Demo 3.4b: Factor momentum persistence with significance bounds
        """
        logger.info("=== DEMO 3.4b: Factor Momentum Persistence ===")
        
        try:
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
                    
                    # Add significance bands using 1.96/sqrt(n) approximation
                    n_obs = len(factor_returns)
                    significance_bound = 1.96 / np.sqrt(n_obs)
                    
                    # Positive significance bound
                    fig.add_hline(
                        y=significance_bound,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"95% significance (+{significance_bound:.3f})",
                        annotation_position="top right",
                        row=row, col=col
                    )
                    
                    # Negative significance bound
                    fig.add_hline(
                        y=-significance_bound,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"95% significance (-{significance_bound:.3f})",
                        annotation_position="bottom right",
                        row=row, col=col
                    )
                    
                    # Add shaded significance region
                    fig.add_hrect(
                        y0=-significance_bound,
                        y1=significance_bound,
                        fillcolor="red",
                        opacity=0.1,
                        layer="below",
                        row=row, col=col
                    )
            
            fig.update_layout(
                title={
                    'text': 'Factor Momentum Persistence Analysis<br><sub>Autocorrelation by Lag Period with 95% Significance Bounds</sub>',
                    'x': 0.5,
                    'font': {'size': 16}
                },
                height=600,
                width=1000
            )
            
            # Update axes
            fig.update_xaxes(title_text="Lag (Months)")
            fig.update_yaxes(title_text="Autocorrelation")
            
            # Save corrected momentum persistence
            fig.write_html(self.results_dir / 'momentum_persistence_analysis_FIXED.html')
            
            logger.info("✅ Demo 3.4b: Momentum persistence completed successfully")
            logger.info(f"   • Statistical significance bounds: ±{significance_bound:.3f} (95% confidence)")
            logger.info("   • Significance testing: 1.96/sqrt(n) with n = sample size")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo 3.4b failed: {e}")
            return False
    
    def run_all_demos(self):
        """
        Run all individual demos to test and fix failed components
        """
        logger.info("=" * 80)
        logger.info("RUNNING PHASE 3 INDIVIDUAL SUBSTEP DEMOS")
        logger.info("=" * 80)
        
        results = {}
        
        # Demo 3.1a: Interactive timeline
        results['3.1a'] = self.demo_3_1a_interactive_timeline()
        
        # Demo 3.2b: Risk-adjusted heatmap
        results['3.2b'] = self.demo_3_2b_risk_adjusted_heatmap()
        
        # Demo 3.3a: Factor rotation wheel
        results['3.3a'] = self.demo_3_3a_factor_rotation_wheel()
        
        # Demo 3.4b: Momentum persistence
        results['3.4b'] = self.demo_3_4b_momentum_persistence()
        
        # Summary
        successful_demos = sum(results.values())
        total_demos = len(results)
        
        logger.info("=" * 80)
        logger.info("DEMO RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Successful Demos: {successful_demos}/{total_demos}")
        
        for demo, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"  Demo {demo}: {status}")
        
        if successful_demos == total_demos:
            logger.info("\n🎉 ALL DEMOS PASSED! Fixed components ready for integration.")
        else:
            logger.warning(f"\n⚠️  {total_demos - successful_demos} demos failed. Review logs for details.")
        
        return results

def main():
    """
    Run Phase 3 individual substep demos
    """
    demos = Phase3SubstepDemos()
    results = demos.run_all_demos()
    
    if all(results.values()):
        logger.info("✅ All demos completed successfully - Ready to integrate fixes")
        return True
    else:
        logger.error("❌ Some demos failed - Review and fix before proceeding")
        return False

if __name__ == "__main__":
    main() 