#!/usr/bin/env python3
"""
Generate Risk-Return Scatterplot Analysis
Creates comprehensive risk-return visualizations comparing factor performance
under different regime classification approaches.

Author: Claude Code
Date: July 3, 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskReturnAnalyzer:
    """Generate risk-return analysis for regime classification approaches"""
    
    def __init__(self, results_dir="results/enhanced_regime_analysis_fixed"):
        self.results_dir = Path(results_dir)
        self.data_dir = Path("results/business_cycle_analysis/_archive")
        
        # Load necessary data
        self.aligned_data = None
        self.factor_performance = {}
        
        logger.info("RiskReturnAnalyzer initialized")
    
    def load_data(self):
        """Load aligned data and calculate risk-return metrics"""
        logger.info("Loading data for risk-return analysis...")
        
        try:
            # Load aligned dataset
            aligned_file = self.data_dir / "aligned_master_dataset_FIXED.csv"
            self.aligned_data = pd.read_csv(aligned_file, index_col=0, parse_dates=True)
            logger.info(f"✓ Loaded aligned data: {len(self.aligned_data)} observations")
            
            # Extract regime classifications
            self.original_monthly = self.aligned_data['ECONOMIC_REGIME'].copy()
            
            # Implement persistence-required classification
            self.persistence_required = self._implement_persistence_required()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _implement_persistence_required(self):
        """Recreate persistence-required classification"""
        monthly_regimes = self.original_monthly.copy()
        persistence_required = pd.Series(index=self.aligned_data.index, dtype='object')
        
        current_confirmed_regime = monthly_regimes.iloc[0]
        consecutive_count = 1
        provisional_regime = monthly_regimes.iloc[0]
        persistence_required.iloc[0] = current_confirmed_regime
        
        for i in range(1, len(monthly_regimes)):
            current_signal = monthly_regimes.iloc[i]
            
            if current_signal == provisional_regime:
                consecutive_count += 1
                if consecutive_count >= 3 and current_signal != current_confirmed_regime:
                    current_confirmed_regime = current_signal
            else:
                provisional_regime = current_signal
                consecutive_count = 1
            
            persistence_required.iloc[i] = current_confirmed_regime
        
        return persistence_required
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive risk-return metrics for all factors and approaches"""
        logger.info("Calculating comprehensive risk-return metrics...")
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Also add S&P 500 if available
        if 'SP500' in self.aligned_data.columns:
            factors.append('SP500')
        
        # Calculate metrics for different approaches
        approaches = {
            'Overall Period': None,  # No regime filter
            'Original Monthly': self.original_monthly,
            'Persistence-Required': self.persistence_required
        }
        
        self.risk_return_data = {}
        
        for approach_name, regime_series in approaches.items():
            approach_metrics = {}
            
            if approach_name == 'Overall Period':
                # Calculate for entire period
                for factor in factors:
                    if factor in self.aligned_data.columns:
                        returns = self.aligned_data[factor]
                        approach_metrics[factor] = {
                            'annual_return': returns.mean() * 12 * 100,
                            'annual_volatility': returns.std() * np.sqrt(12) * 100,
                            'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0,
                            'max_drawdown': self._calculate_max_drawdown(returns),
                            'observations': len(returns)
                        }
            else:
                # Calculate for each regime within approach
                regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
                
                for regime in regimes:
                    regime_mask = regime_series == regime
                    regime_metrics = {}
                    
                    for factor in factors:
                        if factor in self.aligned_data.columns:
                            returns = self.aligned_data[factor][regime_mask]
                            
                            if len(returns) > 0:
                                regime_metrics[factor] = {
                                    'annual_return': returns.mean() * 12 * 100,
                                    'annual_volatility': returns.std() * np.sqrt(12) * 100,
                                    'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0,
                                    'max_drawdown': self._calculate_max_drawdown(returns),
                                    'observations': len(returns)
                                }
                    
                    if regime_metrics:
                        approach_metrics[regime] = regime_metrics
            
            self.risk_return_data[approach_name] = approach_metrics
        
        logger.info(f"✓ Calculated metrics for {len(self.risk_return_data)} approaches")
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def create_risk_return_visualizations(self):
        """Create comprehensive risk-return scatterplots"""
        logger.info("Creating risk-return visualizations...")
        
        # 1. Overall comparison across approaches
        self._create_overall_risk_return_plot()
        
        # 2. Regime-specific risk-return plots
        self._create_regime_specific_plots()
        
        # 3. Factor trajectory plots showing regime transitions
        self._create_factor_trajectory_plots()
        
        # 4. Efficient frontier comparison
        self._create_efficient_frontier_comparison()
    
    def _create_overall_risk_return_plot(self):
        """Create overall risk-return comparison"""
        fig = go.Figure()
        
        # Colors for factors
        factor_colors = {
            'Value': '#1f77b4',
            'Quality': '#ff7f0e', 
            'MinVol': '#2ca02c',
            'Momentum': '#d62728',
            'SP500': '#000000'
        }
        
        # Markers for approaches
        marker_shapes = {
            'Overall Period': 'circle',
            'Original Monthly': 'square',
            'Persistence-Required': 'diamond'
        }
        
        # Add data points
        for approach in ['Overall Period', 'Original Monthly', 'Persistence-Required']:
            if approach == 'Overall Period':
                # Single set of metrics
                metrics = self.risk_return_data[approach]
                for factor, data in metrics.items():
                    fig.add_trace(go.Scatter(
                        x=[data['annual_volatility']],
                        y=[data['annual_return']],
                        mode='markers+text',
                        name=f"{factor} ({approach})",
                        marker=dict(
                            size=15,
                            color=factor_colors.get(factor, 'gray'),
                            symbol=marker_shapes[approach],
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        text=[factor],
                        textposition='top center',
                        hovertemplate=f"{factor} - {approach}<br>" +
                                     f"Return: {data['annual_return']:.1f}%<br>" +
                                     f"Volatility: {data['annual_volatility']:.1f}%<br>" +
                                     f"Sharpe: {data['sharpe_ratio']:.2f}<br>" +
                                     f"Max DD: {data['max_drawdown']:.1f}%<extra></extra>"
                    ))
            else:
                # Average across regimes for comparison
                all_returns = []
                all_vols = []
                
                for regime, regime_metrics in self.risk_return_data[approach].items():
                    for factor, data in regime_metrics.items():
                        all_returns.append(data['annual_return'])
                        all_vols.append(data['annual_volatility'])
                
                if all_returns:
                    for factor in ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500']:
                        # Calculate weighted average based on observations
                        factor_returns = []
                        factor_vols = []
                        total_obs = 0
                        
                        for regime, regime_metrics in self.risk_return_data[approach].items():
                            if factor in regime_metrics:
                                data = regime_metrics[factor]
                                obs = data['observations']
                                factor_returns.append(data['annual_return'] * obs)
                                factor_vols.append(data['annual_volatility'] * obs)
                                total_obs += obs
                        
                        if factor_returns and total_obs > 0:
                            avg_return = sum(factor_returns) / total_obs
                            avg_vol = sum(factor_vols) / total_obs
                            
                            fig.add_trace(go.Scatter(
                                x=[avg_vol],
                                y=[avg_return],
                                mode='markers',
                                name=f"{factor} ({approach})",
                                marker=dict(
                                    size=12,
                                    color=factor_colors.get(factor, 'gray'),
                                    symbol=marker_shapes[approach],
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                showlegend=False,
                                hovertemplate=f"{factor} - {approach}<br>" +
                                             f"Avg Return: {avg_return:.1f}%<br>" +
                                             f"Avg Volatility: {avg_vol:.1f}%<extra></extra>"
                            ))
        
        # Add efficient frontier line (simplified)
        vols = np.linspace(10, 20, 50)
        returns = 0.5 * vols  # Simplified assumption
        fig.add_trace(go.Scatter(
            x=vols,
            y=returns,
            mode='lines',
            name='Theoretical Frontier',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ))
        
        # Add annotations
        fig.add_annotation(
            x=12, y=14,
            text="Higher Sharpe<br>Ratio",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-40
        )
        
        fig.update_layout(
            title='Risk-Return Analysis: Factor Performance Across Regime Classifications<br>' +
                  '<sub>Comparing Overall Period, Original Monthly, and Persistence-Required Approaches</sub>',
            xaxis_title='Annual Volatility (%)',
            yaxis_title='Annual Return (%)',
            height=700,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save plot
        output_file = self.results_dir / 'risk_return_overall_comparison.html'
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved overall risk-return plot: {output_file}")
    
    def _create_regime_specific_plots(self):
        """Create regime-specific risk-return plots"""
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=regimes,
            horizontal_spacing=0.15,
            vertical_spacing=0.15
        )
        
        factor_colors = {
            'Value': '#1f77b4',
            'Quality': '#ff7f0e', 
            'MinVol': '#2ca02c',
            'Momentum': '#d62728',
            'SP500': '#000000'
        }
        
        approaches = ['Original Monthly', 'Persistence-Required']
        symbols = ['circle', 'diamond']
        
        for regime_idx, regime in enumerate(regimes):
            row = regime_idx // 2 + 1
            col = regime_idx % 2 + 1
            
            for approach_idx, approach in enumerate(approaches):
                if regime in self.risk_return_data[approach]:
                    regime_data = self.risk_return_data[approach][regime]
                    
                    for factor, metrics in regime_data.items():
                        fig.add_trace(
                            go.Scatter(
                                x=[metrics['annual_volatility']],
                                y=[metrics['annual_return']],
                                mode='markers',
                                name=f"{factor} - {approach.split()[0]}",
                                marker=dict(
                                    size=12,
                                    color=factor_colors.get(factor, 'gray'),
                                    symbol=symbols[approach_idx],
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                legendgroup=f"{factor}_{approach}",
                                showlegend=(regime_idx == 0),
                                hovertemplate=f"{factor} - {approach}<br>" +
                                             f"Return: {metrics['annual_return']:.1f}%<br>" +
                                             f"Volatility: {metrics['annual_volatility']:.1f}%<br>" +
                                             f"Sharpe: {metrics['sharpe_ratio']:.2f}<extra></extra>"
                            ),
                            row=row, col=col
                        )
            
            # Add zero lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=row, col=col)
        
        fig.update_layout(
            title='Risk-Return by Economic Regime: Original Monthly vs Persistence-Required<br>' +
                  '<sub>How factor risk-return profiles change across different economic environments</sub>',
            height=800,
            showlegend=True
        )
        
        # Update axes
        for i in range(1, 5):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            fig.update_xaxes(title_text="Annual Volatility (%)" if row == 2 else "", row=row, col=col)
            fig.update_yaxes(title_text="Annual Return (%)" if col == 1 else "", row=row, col=col)
        
        # Save plot
        output_file = self.results_dir / 'risk_return_by_regime.html'
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved regime-specific risk-return plot: {output_file}")
    
    def _create_factor_trajectory_plots(self):
        """Create factor trajectory plots showing how risk-return changes"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Value', 'Quality', 'MinVol', 'Momentum'],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        regime_colors = {
            'Goldilocks': '#2E8B57',
            'Overheating': '#FF6347',
            'Stagflation': '#FFD700', 
            'Recession': '#8B0000'
        }
        
        for factor_idx, factor in enumerate(factors):
            row = factor_idx // 2 + 1
            col = factor_idx % 2 + 1
            
            # Plot trajectory for each approach
            for approach in ['Original Monthly', 'Persistence-Required']:
                x_vals = []
                y_vals = []
                colors = []
                texts = []
                
                for regime in regimes:
                    if regime in self.risk_return_data[approach] and \
                       factor in self.risk_return_data[approach][regime]:
                        metrics = self.risk_return_data[approach][regime][factor]
                        x_vals.append(metrics['annual_volatility'])
                        y_vals.append(metrics['annual_return'])
                        colors.append(regime_colors[regime])
                        texts.append(regime)
                
                if x_vals:
                    # Add connecting lines
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines',
                            line=dict(
                                color='gray',
                                width=1,
                                dash='dash' if approach == 'Original Monthly' else 'solid'
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=row, col=col
                    )
                    
                    # Add regime points
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='markers+text',
                            marker=dict(
                                size=12,
                                color=colors,
                                line=dict(width=2, color='DarkSlateGrey')
                            ),
                            text=[t[0] for t in texts],  # First letter only
                            textposition='middle center',
                            textfont=dict(color='white', size=8),
                            name=approach,
                            legendgroup=approach,
                            showlegend=(factor_idx == 0),
                            hovertemplate='%{text}<br>Return: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>'
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title='Factor Risk-Return Trajectories Across Regimes<br>' +
                  '<sub>G=Goldilocks, O=Overheating, S=Stagflation, R=Recession</sub>',
            height=800,
            showlegend=True
        )
        
        # Update axes
        for i in range(1, 5):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            fig.update_xaxes(title_text="Annual Volatility (%)" if row == 2 else "", row=row, col=col)
            fig.update_yaxes(title_text="Annual Return (%)" if col == 1 else "", row=row, col=col)
        
        # Save plot
        output_file = self.results_dir / 'risk_return_factor_trajectories.html'
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved factor trajectory plot: {output_file}")
    
    def _create_efficient_frontier_comparison(self):
        """Create efficient frontier comparison between approaches"""
        fig = go.Figure()
        
        # Calculate portfolio combinations for each approach
        approaches = ['Original Monthly', 'Persistence-Required']
        colors = ['blue', 'red']
        
        for approach_idx, approach in enumerate(approaches):
            # Get average metrics across regimes
            factor_metrics = {}
            
            for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                total_return = 0
                total_vol = 0
                total_obs = 0
                
                for regime, regime_data in self.risk_return_data[approach].items():
                    if factor in regime_data:
                        metrics = regime_data[factor]
                        obs = metrics['observations']
                        total_return += metrics['annual_return'] * obs
                        total_vol += metrics['annual_volatility'] * obs
                        total_obs += obs
                
                if total_obs > 0:
                    factor_metrics[factor] = {
                        'return': total_return / total_obs,
                        'volatility': total_vol / total_obs
                    }
            
            # Generate efficient frontier (simplified)
            frontier_points = []
            
            # Create portfolio combinations
            for i in range(100):
                weights = np.random.dirichlet(np.ones(4))  # Random weights summing to 1
                
                portfolio_return = sum(weights[j] * factor_metrics[f]['return'] 
                                     for j, f in enumerate(factor_metrics.keys()))
                portfolio_vol = np.sqrt(sum(weights[j]**2 * factor_metrics[f]['volatility']**2 
                                          for j, f in enumerate(factor_metrics.keys())))
                
                frontier_points.append((portfolio_vol, portfolio_return))
            
            # Sort by volatility and keep efficient points
            frontier_points.sort(key=lambda x: x[0])
            efficient_frontier = []
            max_return = -float('inf')
            
            for vol, ret in frontier_points:
                if ret > max_return:
                    efficient_frontier.append((vol, ret))
                    max_return = ret
            
            # Plot efficient frontier
            if efficient_frontier:
                x_vals, y_vals = zip(*efficient_frontier)
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        name=f'{approach} Frontier',
                        line=dict(color=colors[approach_idx], width=3),
                        hovertemplate='Portfolio<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                    )
                )
            
            # Add individual factors
            for factor, metrics in factor_metrics.items():
                fig.add_trace(
                    go.Scatter(
                        x=[metrics['volatility']],
                        y=[metrics['return']],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=colors[approach_idx],
                            symbol='diamond',
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        text=[factor[0]],  # First letter
                        textposition='middle center',
                        textfont=dict(color='white'),
                        name=f'{factor} ({approach})',
                        showlegend=False,
                        hovertemplate=f'{factor} - {approach}<br>Return: {metrics["return"]:.1f}%<br>Vol: {metrics["volatility"]:.1f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title='Efficient Frontier Comparison: Original Monthly vs Persistence-Required<br>' +
                  '<sub>Potential portfolio combinations under each regime classification approach</sub>',
            xaxis_title='Annual Volatility (%)',
            yaxis_title='Annual Return (%)',
            height=600,
            hovermode='closest',
            showlegend=True
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Save plot
        output_file = self.results_dir / 'efficient_frontier_comparison.html'
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved efficient frontier comparison: {output_file}")
    
    def generate_risk_return_report(self):
        """Generate comprehensive risk-return analysis report"""
        logger.info("Generating risk-return analysis report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'key_findings': {},
            'factor_performance': self.risk_return_data,
            'visualization_files': [
                'risk_return_overall_comparison.html',
                'risk_return_by_regime.html',
                'risk_return_factor_trajectories.html',
                'efficient_frontier_comparison.html'
            ]
        }
        
        # Analyze key findings
        # Compare Sharpe ratios
        overall_sharpes = {}
        for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
            if factor in self.risk_return_data['Overall Period']:
                overall_sharpes[factor] = self.risk_return_data['Overall Period'][factor]['sharpe_ratio']
        
        report['key_findings']['best_overall_sharpe'] = max(overall_sharpes, key=overall_sharpes.get)
        report['key_findings']['sharpe_rankings'] = dict(sorted(overall_sharpes.items(), 
                                                               key=lambda x: x[1], reverse=True))
        
        # Save report
        report_file = self.results_dir / 'risk_return_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ Saved risk-return analysis report: {report_file}")
        
        return report
    
    def run_complete_analysis(self):
        """Run complete risk-return analysis"""
        logger.info("🚀 Starting Risk-Return Analysis")
        logger.info("="*70)
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Exiting.")
            return None
        
        # Calculate metrics
        self.calculate_comprehensive_metrics()
        
        # Create visualizations
        self.create_risk_return_visualizations()
        
        # Generate report
        report = self.generate_risk_return_report()
        
        logger.info("\n" + "="*70)
        logger.info("✅ Risk-Return Analysis Complete!")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        logger.info("="*70)
        
        return report


if __name__ == "__main__":
    # Run risk-return analysis
    analyzer = RiskReturnAnalyzer()
    report = analyzer.run_complete_analysis()
    
    if report:
        print("\n🏆 RISK-RETURN ANALYSIS COMPLETE! 🏆")
        print(f"Best overall Sharpe ratio: {report['key_findings']['best_overall_sharpe']}")
        print(f"Visualizations created: {len(report['visualization_files'])}")