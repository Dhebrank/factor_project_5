"""
Phase 3 Timeline Fix
Fixes the datetime arithmetic issue in the interactive timeline component
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimelineFixer:
    """
    Fixes the interactive timeline with proper datetime handling
    """
    
    def __init__(self, results_dir="results/business_cycle_analysis"):
        self.results_dir = Path(results_dir)
        
        # Load data
        self.aligned_data = pd.read_csv(
            self.results_dir / 'aligned_master_dataset_FIXED.csv',
            index_col=0,
            parse_dates=True
        )
        
        logger.info("Timeline Fixer initialized")
    
    def create_fixed_interactive_timeline(self):
        """
        Create interactive timeline with proper datetime handling
        """
        logger.info("=== CREATING FIXED INTERACTIVE TIMELINE ===")
        
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
            
            # Get regime data
            regime_col = self.aligned_data['ECONOMIC_REGIME']
            dates = self.aligned_data.index
            
            # FIXED: Create regime periods with proper datetime handling
            regime_periods = []
            current_regime = None
            start_date = None
            
            # Convert to list for proper iteration
            regime_items = list(regime_col.items())
            
            for i, (date, regime) in enumerate(regime_items):
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
                # Add background rectangle
                fig.add_shape(
                    type="rect",
                    x0=period['start'],
                    x1=period['end'],
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
            
            # Add S&P 500 performance line if available
            if 'SP500_Monthly_Return' in self.aligned_data.columns:
                sp500_returns = self.aligned_data['SP500_Monthly_Return'].fillna(0)
                sp500_cumulative = (1 + sp500_returns).cumprod()
                
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
                    factor_returns = self.aligned_data[factor].fillna(0)
                    factor_cumulative = (1 + factor_returns).cumprod()
                    
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
                vix_data = self.aligned_data['VIX'].fillna(0)  # Use 0 instead of ffill
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=vix_data,
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
                    'text': 'Business Cycle Factor Performance Analysis (1998-2025)<br><sub>FIXED: Regime Transitions & VIX Subplot</sub>',
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
            
            # Save fixed timeline
            fig.write_html(self.results_dir / 'interactive_timeline_regime_overlay_FIXED.html')
            
            logger.info("✅ Fixed interactive timeline created successfully")
            logger.info(f"   • Regime periods: {len(regime_periods)} periods identified")
            logger.info(f"   • Transition markers: {len(regime_periods)-1} transitions marked")
            logger.info("   • VIX subplot: Row 2 with threshold lines at 25, 35, 50")
            logger.info("   • Fixed: Proper datetime handling without arithmetic operations")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create fixed timeline: {e}")
            return False
    
    def integrate_all_fixes(self):
        """
        Integrate all fixed components by replacing original files
        """
        logger.info("=== INTEGRATING ALL FIXES ===")
        
        try:
            # Files to replace with fixed versions
            fixes = {
                'interactive_timeline_regime_overlay_FIXED.html': 'interactive_timeline_regime_overlay.html',
                'risk_adjusted_heatmap_FIXED.html': 'risk_adjusted_heatmap.html',
                'factor_rotation_wheel_FIXED.html': 'factor_rotation_wheel.html',
                'momentum_persistence_analysis_FIXED.html': 'momentum_persistence_analysis.html'
            }
            
            integration_results = {}
            
            for fixed_file, original_file in fixes.items():
                fixed_path = self.results_dir / fixed_file
                original_path = self.results_dir / original_file
                
                if fixed_path.exists():
                    # Replace original with fixed version
                    import shutil
                    shutil.copy2(fixed_path, original_path)
                    integration_results[original_file] = True
                    logger.info(f"✅ Integrated fix: {original_file}")
                else:
                    integration_results[original_file] = False
                    logger.warning(f"⚠️  Fixed file not found: {fixed_file}")
            
            # Summary
            successful_integrations = sum(integration_results.values())
            total_integrations = len(integration_results)
            
            logger.info(f"Integration Results: {successful_integrations}/{total_integrations} fixes applied")
            
            if successful_integrations == total_integrations:
                logger.info("🎉 ALL FIXES SUCCESSFULLY INTEGRATED!")
                return True
            else:
                logger.warning("⚠️  Some fixes could not be integrated")
                return False
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return False

def main():
    """
    Run timeline fix and integration
    """
    fixer = TimelineFixer()
    
    # Create fixed timeline
    timeline_success = fixer.create_fixed_interactive_timeline()
    
    # Integrate all fixes
    if timeline_success:
        integration_success = fixer.integrate_all_fixes()
        
        if integration_success:
            logger.info("✅ All Phase 3 fixes completed and integrated successfully")
            return True
        else:
            logger.error("❌ Integration failed")
            return False
    else:
        logger.error("❌ Timeline fix failed")
        return False

if __name__ == "__main__":
    main() 