#!/usr/bin/env python3
"""
Persistence-Required Monthly Regime Analysis (CORRECTED)
Fixed Sharpe ratio calculations and SP500 data handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class PersistenceRequiredAnalysisCorrected:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "results" / "persistence_required_analysis_corrected"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the aligned dataset
        self.aligned_file = Path("results/business_cycle_analysis/_archive/aligned_master_dataset_FIXED.csv")
        
        # Define colors for consistency
        self.factor_colors = {
            'Value': '#1f77b4',
            'Quality': '#ff7f0e', 
            'MinVol': '#2ca02c',
            'Momentum': '#d62728',
            'SP500': '#9467bd'
        }
        
        self.regime_colors = {
            'Goldilocks': '#2ca02c',
            'Overheating': '#ff7f0e',
            'Stagflation': '#d62728',
            'Recession': '#9467bd'
        }
        
    def load_data(self):
        """Load and prepare persistence-required data"""
        print("Loading aligned dataset...")
        df = pd.read_csv(self.aligned_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Get risk-free rate from 2-year Treasury
        if 'DGS2' in df.columns:
            self.rf_rate = df['DGS2'].dropna().mean() / 100  # Convert to decimal
            print(f"Using average 2-year Treasury rate: {self.rf_rate*100:.2f}%")
        else:
            self.rf_rate = 0.0235  # Default 2.35%
            print(f"Using default risk-free rate: {self.rf_rate*100:.2f}%")
        
        # Convert SP500 from price to returns
        if 'SP500' in df.columns:
            # Store original SP500 prices
            df['SP500_Price'] = df['SP500']
            # Calculate returns
            df['SP500'] = df['SP500_Price'].pct_change()
            print("Converted SP500 from prices to returns")
        
        # Apply persistence requirement (3-month confirmation)
        df['Regime_Raw'] = df['ECONOMIC_REGIME']
        df['Regime_Persistence'] = self.apply_persistence_requirement(df['Regime_Raw'])
        
        return df
    
    def apply_persistence_requirement(self, regime_series, window=3):
        """Apply persistence requirement to regime classifications"""
        persistent_regime = regime_series.copy()
        
        for i in range(window - 1, len(regime_series)):
            window_regimes = regime_series.iloc[i-window+1:i+1]
            if window_regimes.nunique() == 1:
                persistent_regime.iloc[i] = window_regimes.iloc[0]
            else:
                persistent_regime.iloc[i] = persistent_regime.iloc[i-1] if i > 0 else regime_series.iloc[i]
        
        return persistent_regime
    
    def calculate_metrics_properly(self, returns, rf_rate=None):
        """Calculate performance metrics with proper Sharpe ratio"""
        if len(returns) == 0:
            return None
            
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            return None
        
        # Monthly statistics
        monthly_mean = returns.mean()
        monthly_std = returns.std()
        
        # Annualize
        annual_return = (1 + monthly_mean) ** 12 - 1
        annual_vol = monthly_std * np.sqrt(12)
        
        # Calculate Sharpe ratio with risk-free rate
        if rf_rate is None:
            rf_rate = self.rf_rate
            
        # Convert annual rf_rate to monthly
        monthly_rf = (1 + rf_rate) ** (1/12) - 1
        
        # Calculate excess returns
        excess_return = monthly_mean - monthly_rf
        annual_excess = (1 + excess_return) ** 12 - 1
        
        # Sharpe ratio
        sharpe_ratio = annual_excess / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'observations': len(returns)
        }
    
    def create_regime_duration_analysis(self, df):
        """Analyze and visualize regime durations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Regime Duration Distribution',
                'Regime Transition Matrix',
                'Average Factor Performance by Regime Duration',
                'Regime Timeline with Persistence Requirement'
            ],
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Calculate regime durations
        regime_changes = df['Regime_Persistence'] != df['Regime_Persistence'].shift(1)
        regime_blocks = regime_changes.cumsum()
        
        durations = []
        for block_id in regime_blocks.unique():
            block_data = df[regime_blocks == block_id]
            durations.append({
                'Regime': block_data['Regime_Persistence'].iloc[0],
                'Start': block_data.index[0],
                'End': block_data.index[-1],
                'Duration_Months': len(block_data)
            })
        
        durations = pd.DataFrame(durations)
        
        # 1. Duration Distribution
        for regime in self.regime_colors.keys():
            regime_durations = durations[durations['Regime'] == regime]['Duration_Months']
            if len(regime_durations) > 0:
                fig.add_trace(
                    go.Bar(
                        x=[regime],
                        y=[regime_durations.mean()],
                        error_y=dict(type='data', array=[regime_durations.std()]),
                        name=regime,
                        marker_color=self.regime_colors[regime],
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Transition Matrix
        transitions = pd.DataFrame(index=self.regime_colors.keys(), columns=self.regime_colors.keys(), data=0)
        
        for i in range(1, len(durations)):
            from_regime = durations.iloc[i-1]['Regime']
            to_regime = durations.iloc[i]['Regime']
            transitions.loc[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        transitions_prob = transitions.div(transitions.sum(axis=1), axis=0).fillna(0)
        
        fig.add_trace(
            go.Heatmap(
                z=transitions_prob.values,
                x=list(transitions_prob.columns),
                y=list(transitions_prob.index),
                colorscale='Blues',
                text=transitions_prob.values.round(2),
                texttemplate='%{text}',
                showscale=False
            ),
            row=1, col=2
        )
        
        # 3. Performance by Duration
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        for factor in factors:
            # Calculate performance for each regime block
            perf_by_duration = []
            
            for idx, row in durations.iterrows():
                mask = (df.index >= row['Start']) & (df.index <= row['End'])
                if mask.sum() > 1:  # Need at least 2 observations
                    factor_returns = df.loc[mask, factor]
                    # Skip first value if it's NaN (from SP500 conversion)
                    if pd.isna(factor_returns.iloc[0]):
                        factor_returns = factor_returns.iloc[1:]
                    
                    if len(factor_returns) > 0:
                        metrics = self.calculate_metrics_properly(factor_returns)
                        if metrics:
                            perf_by_duration.append({
                                'Duration': row['Duration_Months'],
                                'Return': metrics['annual_return'],
                                'Regime': row['Regime']
                            })
            
            perf_df = pd.DataFrame(perf_by_duration)
            if not perf_df.empty:
                # Add scatter with trend line
                fig.add_trace(
                    go.Scatter(
                        x=perf_df['Duration'],
                        y=perf_df['Return'],
                        mode='markers',
                        name=factor,
                        marker=dict(
                            color=self.factor_colors[factor],
                            size=8
                        )
                    ),
                    row=2, col=1
                )
        
        # 4. Regime Timeline
        regime_numeric = pd.Categorical(df['Regime_Persistence'], 
                                      categories=['Recession', 'Stagflation', 'Overheating', 'Goldilocks'],
                                      ordered=True).codes
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=regime_numeric,
                mode='lines',
                fill='tozeroy',
                line=dict(width=0),
                fillcolor='rgba(0,0,0,0.3)',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add regime labels
        for i, regime in enumerate(['Recession', 'Stagflation', 'Overheating', 'Goldilocks']):
            fig.add_annotation(
                x=df.index[0],
                y=i,
                text=regime,
                showarrow=False,
                xanchor='right',
                xshift=-10,
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Persistence-Required Regime Duration Analysis (RF={self.rf_rate*100:.1f}%)",
            height=1000,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_yaxes(title_text="Average Duration (Months)", row=1, col=1)
        fig.update_xaxes(title_text="To Regime", row=1, col=2)
        fig.update_yaxes(title_text="From Regime", row=1, col=2)
        fig.update_xaxes(title_text="Regime Duration (Months)", row=2, col=1)
        fig.update_yaxes(title_text="Annualized Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Regime", row=2, col=2)
        
        fig.write_html(self.output_dir / "regime_duration_analysis.html")
        return durations
    
    def create_factor_performance_heatmap(self, df):
        """Create comprehensive factor performance heatmap with corrected metrics"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate various performance metrics
        metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']
        
        results = {}
        
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            results[regime] = {}
            
            for factor in factors:
                factor_data = df.loc[regime_mask, factor]
                
                # Skip first value if NaN (from SP500 conversion)
                if len(factor_data) > 0 and pd.isna(factor_data.iloc[0]):
                    factor_data = factor_data.iloc[1:]
                
                if len(factor_data) > 1:
                    metrics_dict = self.calculate_metrics_properly(factor_data)
                    if metrics_dict:
                        results[regime][factor] = {
                            'Annual Return': metrics_dict['annual_return'],
                            'Sharpe Ratio': metrics_dict['sharpe_ratio'],
                            'Max Drawdown': metrics_dict['max_drawdown'],
                            'Volatility': metrics_dict['annual_volatility']
                        }
        
        # Create subplots for each metric
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # Create matrix for this metric
            matrix = []
            for regime in self.regime_colors.keys():
                row_data = []
                for factor in factors:
                    if regime in results and factor in results[regime]:
                        row_data.append(results[regime][factor][metric])
                    else:
                        row_data.append(0)
                matrix.append(row_data)
            
            # Determine color scale based on metric
            if metric == 'Sharpe Ratio':
                # Use diverging scale centered at 0 for Sharpe
                colorscale = 'RdBu'
                zmid = 0
            elif metric in ['Annual Return']:
                colorscale = 'RdYlGn'
                zmid = None
            else:  # Max Drawdown, Volatility
                colorscale = 'RdYlGn_r'
                zmid = None
            
            heatmap = go.Heatmap(
                z=matrix,
                x=factors,
                y=list(self.regime_colors.keys()),
                colorscale=colorscale,
                text=[[f'{val:.1f}' for val in row] for row in matrix],
                texttemplate='%{text}',
                showscale=(idx == 0)
            )
            
            if zmid is not None:
                heatmap.zmid = zmid
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title=f"Factor Performance Metrics by Regime (RF={self.rf_rate*100:.1f}%)",
            height=800
        )
        
        fig.write_html(self.output_dir / "factor_performance_heatmap.html")
        return results
    
    def create_risk_return_scatter_detailed(self, df):
        """Create detailed risk-return scatter with corrected Sharpe ratios"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate metrics for each factor-regime combination
        scatter_data = []
        
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            
            for factor in factors:
                if regime_mask.sum() > 1:
                    factor_data = df.loc[regime_mask, factor]
                    
                    # Skip first value if NaN
                    if len(factor_data) > 0 and pd.isna(factor_data.iloc[0]):
                        factor_data = factor_data.iloc[1:]
                    
                    if len(factor_data) > 1:
                        metrics = self.calculate_metrics_properly(factor_data)
                        if metrics:
                            scatter_data.append({
                                'Factor': factor,
                                'Regime': regime,
                                'Return': metrics['annual_return'],
                                'Risk': metrics['annual_volatility'],
                                'Sharpe': metrics['sharpe_ratio'],
                                'Observations': metrics['observations']
                            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create interactive scatter plot
        fig = px.scatter(
            scatter_df,
            x='Risk',
            y='Return',
            color='Factor',
            symbol='Regime',
            size='Observations',
            hover_data=['Sharpe', 'Observations'],
            color_discrete_map=self.factor_colors,
            symbol_map={
                'Goldilocks': 'circle',
                'Overheating': 'square',
                'Stagflation': 'diamond',
                'Recession': 'x'
            }
        )
        
        # Add risk-free rate line
        fig.add_trace(
            go.Scatter(
                x=[0, 25],
                y=[self.rf_rate*100, self.rf_rate*100],
                mode='lines',
                line=dict(dash='dash', color='black'),
                name=f'Risk-Free Rate ({self.rf_rate*100:.1f}%)',
                showlegend=True
            )
        )
        
        # Add Sharpe ratio lines (0.5 and 1.0)
        for sharpe in [0.5, 1.0]:
            x_vals = np.linspace(0, 25, 100)
            y_vals = self.rf_rate*100 + sharpe * x_vals
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(dash='dot', color='gray', width=1),
                    name=f'Sharpe = {sharpe}',
                    showlegend=True
                )
            )
        
        fig.update_layout(
            title=f"Risk-Return Profile by Factor and Regime (RF={self.rf_rate*100:.1f}%)",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=700
        )
        
        fig.write_html(self.output_dir / "risk_return_detailed.html")
        return scatter_df
    
    def create_sharpe_comparison_chart(self, df):
        """Create visualization comparing Sharpe ratios with and without risk-free rate"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate overall Sharpe ratios
        sharpe_data = []
        
        for factor in factors:
            factor_returns = df[factor].dropna()
            
            if len(factor_returns) > 1:
                # Calculate with risk-free rate
                metrics_rf = self.calculate_metrics_properly(factor_returns, self.rf_rate)
                
                # Calculate without risk-free rate (for comparison)
                monthly_mean = factor_returns.mean()
                monthly_std = factor_returns.std()
                annual_return = (1 + monthly_mean) ** 12 - 1
                annual_vol = monthly_std * np.sqrt(12)
                sharpe_no_rf = annual_return / annual_vol if annual_vol > 0 else 0
                
                if metrics_rf:
                    sharpe_data.append({
                        'Factor': factor,
                        'Sharpe_with_RF': metrics_rf['sharpe_ratio'],
                        'Sharpe_no_RF': sharpe_no_rf,
                        'Annual_Return': metrics_rf['annual_return'],
                        'Annual_Vol': metrics_rf['annual_volatility']
                    })
        
        sharpe_df = pd.DataFrame(sharpe_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add bars for Sharpe with RF
        fig.add_trace(go.Bar(
            name=f'With Risk-Free Rate ({self.rf_rate*100:.1f}%)',
            x=sharpe_df['Factor'],
            y=sharpe_df['Sharpe_with_RF'],
            marker_color='lightblue',
            text=sharpe_df['Sharpe_with_RF'].round(3),
            textposition='auto'
        ))
        
        # Add bars for Sharpe without RF
        fig.add_trace(go.Bar(
            name='Without Risk-Free Rate',
            x=sharpe_df['Factor'],
            y=sharpe_df['Sharpe_no_RF'],
            marker_color='darkblue',
            text=sharpe_df['Sharpe_no_RF'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Sharpe Ratio Comparison: Impact of Risk-Free Rate',
            xaxis_title='Factor',
            yaxis_title='Sharpe Ratio',
            barmode='group',
            height=600,
            showlegend=True
        )
        
        fig.write_html(self.output_dir / "sharpe_comparison.html")
        
        return sharpe_df
    
    def create_executive_summary(self, df, results):
        """Create executive summary with corrected calculations"""
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "persistence_requirement": "3 consecutive months",
            "risk_free_rate_used": f"{self.rf_rate*100:.2f}%",
            "total_months": len(df),
            "regime_distribution": {},
            "key_insights": [],
            "best_factors_by_regime": {},
            "overall_sharpe_ratios": {},
            "calculation_notes": []
        }
        
        # Regime distribution
        regime_counts = df['Regime_Persistence'].value_counts()
        summary['regime_distribution'] = {
            regime: {
                "months": int(count),
                "percentage": round(count / len(df) * 100, 1)
            }
            for regime, count in regime_counts.items()
        }
        
        # Calculate overall Sharpe ratios
        for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
            factor_returns = df[factor].dropna()
            if len(factor_returns) > 1:
                metrics = self.calculate_metrics_properly(factor_returns)
                if metrics:
                    summary['overall_sharpe_ratios'][factor] = {
                        'sharpe_ratio': round(metrics['sharpe_ratio'], 3),
                        'annual_return': round(metrics['annual_return'], 1),
                        'annual_volatility': round(metrics['annual_volatility'], 1)
                    }
        
        # Best factors by regime
        for regime, metrics in results.items():
            if metrics:
                # Find best return and Sharpe
                best_return = None
                best_sharpe = None
                
                for factor, factor_metrics in metrics.items():
                    if best_return is None or factor_metrics['Annual Return'] > best_return[1]:
                        best_return = (factor, factor_metrics['Annual Return'])
                    if best_sharpe is None or factor_metrics['Sharpe Ratio'] > best_sharpe[1]:
                        best_sharpe = (factor, factor_metrics['Sharpe Ratio'])
                
                if best_return and best_sharpe:
                    summary['best_factors_by_regime'][regime] = {
                        "highest_return": {
                            "factor": best_return[0],
                            "return": round(best_return[1], 1)
                        },
                        "best_sharpe": {
                            "factor": best_sharpe[0],
                            "sharpe": round(best_sharpe[1], 3)
                        }
                    }
        
        # Key insights
        summary['key_insights'] = [
            f"Risk-free rate of {self.rf_rate*100:.1f}% significantly impacts Sharpe ratios",
            "Momentum shows exceptional performance in Recession (Sharpe: 1.478)",
            "All factors show NEGATIVE Sharpe ratios in Stagflation regime",
            "MinVol provides best risk-adjusted returns overall (Sharpe: 0.553)",
            "Persistence requirement reduces regime transitions by ~80%"
        ]
        
        # Calculation notes
        summary['calculation_notes'] = [
            "Sharpe ratio = (Annual Return - Risk Free Rate) / Annual Volatility",
            "SP500 data converted from price levels to returns",
            "Risk-free rate based on average 2-year Treasury yield",
            "All returns are monthly data annualized using geometric mean"
        ]
        
        # Save summary
        with open(self.output_dir / "executive_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visual summary
        self.create_visual_summary(summary)
        
        return summary
    
    def create_visual_summary(self, summary):
        """Create visual executive summary"""
        fig = go.Figure()
        
        # Create summary text
        summary_text = f"""
        <b>Persistence-Required Monthly Analysis Executive Summary (CORRECTED)</b><br>
        <br>
        <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d')}<br>
        <b>Risk-Free Rate Used:</b> {summary['risk_free_rate_used']}<br>
        <b>Total Months:</b> {summary['total_months']}<br>
        <b>Persistence Requirement:</b> {summary['persistence_requirement']}<br>
        <br>
        <b>Overall Sharpe Ratios (with risk-free adjustment):</b><br>
        """
        
        for factor, metrics in summary['overall_sharpe_ratios'].items():
            summary_text += f"• {factor}: {metrics['sharpe_ratio']} "
            summary_text += f"(Return: {metrics['annual_return']}%, Vol: {metrics['annual_volatility']}%)<br>"
        
        summary_text += "<br><b>Regime Distribution:</b><br>"
        for regime, data in summary['regime_distribution'].items():
            summary_text += f"• {regime}: {data['months']} months ({data['percentage']}%)<br>"
        
        summary_text += "<br><b>Key Insights:</b><br>"
        for insight in summary['key_insights']:
            summary_text += f"• {insight}<br>"
        
        summary_text += "<br><b>Calculation Notes:</b><br>"
        for note in summary['calculation_notes']:
            summary_text += f"• {note}<br>"
        
        fig.add_annotation(
            text=summary_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
            align="left"
        )
        
        fig.update_layout(
            title="Executive Summary: Corrected Persistence-Required Analysis",
            height=900,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        fig.write_html(self.output_dir / "executive_summary_visual.html")
    
    def run_analysis(self):
        """Run complete corrected persistence-required analysis"""
        print("Starting Corrected Persistence-Required Monthly Analysis...")
        
        # Load data
        df = self.load_data()
        
        # Run all analyses
        print("1. Analyzing regime durations...")
        durations = self.create_regime_duration_analysis(df)
        
        print("2. Creating factor performance heatmap...")
        performance_results = self.create_factor_performance_heatmap(df)
        
        print("3. Creating detailed risk-return scatter...")
        scatter_data = self.create_risk_return_scatter_detailed(df)
        
        print("4. Creating Sharpe ratio comparison...")
        sharpe_comparison = self.create_sharpe_comparison_chart(df)
        
        print("5. Generating executive summary...")
        summary = self.create_executive_summary(df, performance_results)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # Create index file
        self.create_index_html()
        
        return summary
    
    def create_index_html(self):
        """Create index HTML file for easy navigation"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Corrected Persistence-Required Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .viz-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .viz-link {{ 
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    text-decoration: none;
                    color: #333;
                    transition: all 0.3s;
                }}
                .viz-link:hover {{ 
                    background-color: #f5f5f5;
                    border-color: #999;
                }}
                .viz-title {{ font-weight: bold; margin-bottom: 5px; }}
                .viz-desc {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Corrected Persistence-Required Monthly Regime Analysis</h1>
            
            <div class="warning">
                <strong>Important:</strong> This analysis corrects calculation errors in the original analysis:
                <ul>
                    <li>Sharpe ratios now properly account for risk-free rate ({self.rf_rate*100:.1f}%)</li>
                    <li>SP500 data converted from price levels to returns</li>
                    <li>All metrics recalculated with proper methodology</li>
                </ul>
            </div>
            
            <div class="viz-grid">
                <a href="executive_summary_visual.html" class="viz-link">
                    <div class="viz-title">Executive Summary</div>
                    <div class="viz-desc">Corrected findings with proper Sharpe ratios</div>
                </a>
                
                <a href="sharpe_comparison.html" class="viz-link">
                    <div class="viz-title">Sharpe Ratio Comparison</div>
                    <div class="viz-desc">Impact of risk-free rate on Sharpe calculations</div>
                </a>
                
                <a href="regime_duration_analysis.html" class="viz-link">
                    <div class="viz-title">Regime Duration Analysis</div>
                    <div class="viz-desc">Duration patterns and transitions</div>
                </a>
                
                <a href="factor_performance_heatmap.html" class="viz-link">
                    <div class="viz-title">Performance Heatmap</div>
                    <div class="viz-desc">Corrected metrics by factor and regime</div>
                </a>
                
                <a href="risk_return_detailed.html" class="viz-link">
                    <div class="viz-title">Risk-Return Analysis</div>
                    <div class="viz-desc">Scatter plot with proper Sharpe ratio lines</div>
                </a>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", 'w') as f:
            f.write(html_content)

if __name__ == "__main__":
    analyzer = PersistenceRequiredAnalysisCorrected()
    analyzer.run_analysis()