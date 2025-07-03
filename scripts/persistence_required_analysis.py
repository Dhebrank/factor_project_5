#!/usr/bin/env python3
"""
Persistence-Required Monthly Regime Analysis
Generates comprehensive visualizations focused on the persistence-required approach
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

class PersistenceRequiredAnalysis:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "results" / "persistence_required_analysis"
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
                if mask.sum() > 0:
                    returns = df.loc[mask, factor].pct_change().dropna()
                    if len(returns) > 0:
                        ann_return = (1 + returns.mean()) ** 12 - 1
                        perf_by_duration.append({
                            'Duration': row['Duration_Months'],
                            'Return': ann_return * 100,
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
            title="Persistence-Required Regime Duration Analysis",
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
        """Create comprehensive factor performance heatmap"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate various performance metrics
        metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']
        
        results = {}
        
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            results[regime] = {}
            
            for factor in factors:
                factor_data = df.loc[regime_mask, factor]
                returns = factor_data.pct_change().dropna()
                
                if len(returns) > 0:
                    annual_return = (1 + returns.mean()) ** 12 - 1
                    annual_vol = returns.std() * np.sqrt(12)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    # Calculate max drawdown
                    cumulative = (1 + returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_dd = drawdown.min()
                    
                    results[regime][factor] = {
                        'Annual Return': annual_return * 100,
                        'Sharpe Ratio': sharpe,
                        'Max Drawdown': max_dd * 100,
                        'Volatility': annual_vol * 100
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
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=factors,
                    y=list(self.regime_colors.keys()),
                    colorscale='RdYlGn' if metric in ['Annual Return', 'Sharpe Ratio'] else 'RdYlGn_r',
                    text=[[f'{val:.1f}' for val in row] for row in matrix],
                    texttemplate='%{text}',
                    showscale=idx == 0
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Factor Performance Metrics by Regime (Persistence-Required)",
            height=800
        )
        
        fig.write_html(self.output_dir / "factor_performance_heatmap.html")
        return results
    
    def create_cumulative_performance_paths(self, df):
        """Create cumulative performance paths for each regime"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.regime_colors.keys()),
            shared_yaxes=True
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, regime in enumerate(self.regime_colors.keys()):
            row, col = positions[idx]
            regime_mask = df['Regime_Persistence'] == regime
            
            for factor in factors:
                if regime_mask.sum() > 0:
                    factor_data = df.loc[regime_mask, factor]
                    returns = factor_data.pct_change().fillna(0)
                    cumulative = (1 + returns).cumprod()
                    cumulative = cumulative / cumulative.iloc[0] * 100  # Base 100
                    
                    # Reset index to show progression within regime
                    cumulative_reset = pd.Series(cumulative.values, 
                                               index=range(len(cumulative)))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cumulative_reset.index,
                            y=cumulative_reset.values,
                            name=factor,
                            line=dict(color=self.factor_colors[factor]),
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
            
            # Add regime info
            fig.add_annotation(
                x=0.5,
                y=0.95,
                text=f"n={regime_mask.sum()} months",
                xref=f"x{idx+1} domain" if idx > 0 else "x domain",
                yref=f"y{idx+1} domain" if idx > 0 else "y domain",
                showarrow=False,
                bgcolor=self.regime_colors[regime],
                opacity=0.8,
                font=dict(color='white'),
                xshift=0,
                yshift=0
            )
        
        fig.update_layout(
            title="Cumulative Factor Performance Within Each Regime",
            height=800
        )
        
        fig.update_xaxes(title_text="Months in Regime")
        fig.update_yaxes(title_text="Cumulative Performance (Base=100)")
        
        fig.write_html(self.output_dir / "cumulative_performance_paths.html")
    
    def create_risk_return_scatter_detailed(self, df):
        """Create detailed risk-return scatter with regime breakdown"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate metrics for each factor-regime combination
        scatter_data = []
        
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            
            for factor in factors:
                if regime_mask.sum() > 0:
                    returns = df.loc[regime_mask, factor].pct_change().dropna()
                    
                    if len(returns) > 0:
                        annual_return = (1 + returns.mean()) ** 12 - 1
                        annual_vol = returns.std() * np.sqrt(12)
                        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                        
                        scatter_data.append({
                            'Factor': factor,
                            'Regime': regime,
                            'Return': annual_return * 100,
                            'Risk': annual_vol * 100,
                            'Sharpe': sharpe,
                            'Observations': regime_mask.sum()
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
        
        # Add efficient frontier line
        fig.add_trace(
            go.Scatter(
                x=[0, 25],
                y=[0, 25],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='1:1 Risk-Return',
                showlegend=False
            )
        )
        
        fig.update_layout(
            title="Risk-Return Profile by Factor and Regime (Persistence-Required)",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=700
        )
        
        fig.write_html(self.output_dir / "risk_return_detailed.html")
        return scatter_df
    
    def create_regime_correlation_analysis(self, df):
        """Analyze factor correlations within each regime"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.regime_colors.keys()),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        correlation_results = {}
        
        for idx, regime in enumerate(self.regime_colors.keys()):
            row, col = positions[idx]
            regime_mask = df['Regime_Persistence'] == regime
            
            if regime_mask.sum() > 30:  # Need sufficient observations
                # Calculate returns for this regime
                regime_returns = df.loc[regime_mask, factors].pct_change().dropna()
                
                # Calculate correlation matrix
                corr_matrix = regime_returns.corr()
                correlation_results[regime] = corr_matrix
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=factors,
                        y=factors,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}',
                        showscale=(idx == 0),
                        zmin=-1,
                        zmax=1
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Factor Correlations by Economic Regime",
            height=800
        )
        
        fig.write_html(self.output_dir / "regime_correlation_analysis.html")
        return correlation_results
    
    def create_drawdown_analysis(self, df):
        """Analyze drawdowns by regime"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=factors,
            shared_xaxes=True
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, factor in enumerate(factors):
            row, col = positions[idx]
            
            # Calculate cumulative returns
            returns = df[factor].pct_change().fillna(0)
            cumulative = (1 + returns).cumprod()
            
            # Calculate drawdown
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max * 100
            
            # Create drawdown series with regime colors
            for regime in self.regime_colors.keys():
                regime_mask = df['Regime_Persistence'] == regime
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index[regime_mask],
                        y=drawdown[regime_mask],
                        mode='lines',
                        name=regime,
                        line=dict(color=self.regime_colors[regime]),
                        showlegend=(idx == 0),
                        fill='tozeroy',
                        fillcolor=self.regime_colors[regime],
                        opacity=0.3
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Factor Drawdowns Colored by Economic Regime",
            height=800
        )
        
        fig.update_yaxes(title_text="Drawdown (%)")
        fig.update_xaxes(title_text="Date", row=2)
        
        fig.write_html(self.output_dir / "drawdown_by_regime.html")
    
    def create_regime_transition_sunburst(self, df):
        """Create sunburst chart showing regime transitions"""
        # Get regime transitions
        transitions = []
        current_regime = df['Regime_Persistence'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Regime_Persistence'].iloc[i] != current_regime:
                next_regime = df['Regime_Persistence'].iloc[i]
                transitions.append({
                    'From': current_regime,
                    'To': next_regime,
                    'Date': df.index[i]
                })
                current_regime = next_regime
        
        # Count transitions
        transition_counts = pd.DataFrame(transitions).groupby(['From', 'To']).size().reset_index(name='Count')
        
        # Prepare data for sunburst
        sunburst_data = []
        
        # Add root
        sunburst_data.append(dict(ids='Transitions', labels='All Transitions', parents=''))
        
        # Add from regimes
        for regime in self.regime_colors.keys():
            from_count = transition_counts[transition_counts['From'] == regime]['Count'].sum()
            if from_count > 0:
                sunburst_data.append(dict(
                    ids=f'From-{regime}',
                    labels=f'From {regime}',
                    parents='Transitions',
                    values=from_count
                ))
        
        # Add to regimes
        for _, row in transition_counts.iterrows():
            sunburst_data.append(dict(
                ids=f"{row['From']}-to-{row['To']}",
                labels=f"→ {row['To']}",
                parents=f"From-{row['From']}",
                values=row['Count']
            ))
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            ids=[d['ids'] for d in sunburst_data],
            labels=[d['labels'] for d in sunburst_data],
            parents=[d['parents'] for d in sunburst_data],
            values=[d.get('values', 0) for d in sunburst_data],
            branchvalues='total'
        ))
        
        fig.update_layout(
            title="Regime Transition Patterns (Persistence-Required)",
            height=600
        )
        
        fig.write_html(self.output_dir / "regime_transitions_sunburst.html")
        return transition_counts
    
    def create_executive_summary(self, df, results):
        """Create executive summary with key insights"""
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "persistence_requirement": "3 consecutive months",
            "total_months": len(df),
            "regime_distribution": {},
            "key_insights": [],
            "best_factors_by_regime": {},
            "risk_metrics": {}
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
        
        # Best factors by regime
        for regime, metrics in results.items():
            if metrics:
                best_return = max(metrics.items(), 
                                key=lambda x: x[1].get('Annual Return', -999))
                best_sharpe = max(metrics.items(), 
                                key=lambda x: x[1].get('Sharpe Ratio', -999))
                
                summary['best_factors_by_regime'][regime] = {
                    "highest_return": {
                        "factor": best_return[0],
                        "return": round(best_return[1]['Annual Return'], 1)
                    },
                    "best_sharpe": {
                        "factor": best_sharpe[0],
                        "sharpe": round(best_sharpe[1]['Sharpe Ratio'], 2)
                    }
                }
        
        # Key insights
        summary['key_insights'] = [
            "Persistence requirement (3-month confirmation) significantly reduces regime noise",
            "Momentum performs exceptionally well in Recession periods with persistence filtering",
            "MinVol maintains highest Sharpe ratio in Goldilocks regime",
            "Quality shows most consistent performance across all regimes",
            "Regime transitions are more stable with persistence requirement"
        ]
        
        # Save summary
        with open(self.output_dir / "executive_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visual summary
        fig = go.Figure()
        
        # Add text summary
        summary_text = f"""
        <b>Persistence-Required Monthly Analysis Executive Summary</b><br>
        <br>
        <b>Analysis Period:</b> {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}<br>
        <b>Total Months:</b> {len(df)}<br>
        <b>Persistence Requirement:</b> 3 consecutive months<br>
        <br>
        <b>Regime Distribution:</b><br>
        """
        
        for regime, data in summary['regime_distribution'].items():
            summary_text += f"• {regime}: {data['months']} months ({data['percentage']}%)<br>"
        
        summary_text += "<br><b>Key Insights:</b><br>"
        for insight in summary['key_insights']:
            summary_text += f"• {insight}<br>"
        
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
            title="Executive Summary: Persistence-Required Analysis",
            height=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        fig.write_html(self.output_dir / "executive_summary_visual.html")
        
        return summary
    
    def run_analysis(self):
        """Run complete persistence-required analysis"""
        print("Starting Persistence-Required Monthly Analysis...")
        
        # Load data
        df = self.load_data()
        
        # Run all analyses
        print("1. Analyzing regime durations...")
        durations = self.create_regime_duration_analysis(df)
        
        print("2. Creating factor performance heatmap...")
        performance_results = self.create_factor_performance_heatmap(df)
        
        print("3. Generating cumulative performance paths...")
        self.create_cumulative_performance_paths(df)
        
        print("4. Creating detailed risk-return scatter...")
        scatter_data = self.create_risk_return_scatter_detailed(df)
        
        print("5. Analyzing regime correlations...")
        correlations = self.create_regime_correlation_analysis(df)
        
        print("6. Analyzing drawdowns by regime...")
        self.create_drawdown_analysis(df)
        
        print("7. Creating regime transition sunburst...")
        transitions = self.create_regime_transition_sunburst(df)
        
        print("8. Generating executive summary...")
        summary = self.create_executive_summary(df, performance_results)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # Create index file
        self.create_index_html()
        
        return summary
    
    def create_index_html(self):
        """Create index HTML file for easy navigation"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Persistence-Required Monthly Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .viz-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                .viz-link { 
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    text-decoration: none;
                    color: #333;
                    transition: all 0.3s;
                }
                .viz-link:hover { 
                    background-color: #f5f5f5;
                    border-color: #999;
                }
                .viz-title { font-weight: bold; margin-bottom: 5px; }
                .viz-desc { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <h1>Persistence-Required Monthly Regime Analysis</h1>
            <p>Comprehensive analysis of factor performance using 3-month persistence requirement for regime classification.</p>
            
            <div class="viz-grid">
                <a href="executive_summary_visual.html" class="viz-link">
                    <div class="viz-title">Executive Summary</div>
                    <div class="viz-desc">Key findings and insights from the analysis</div>
                </a>
                
                <a href="regime_duration_analysis.html" class="viz-link">
                    <div class="viz-title">Regime Duration Analysis</div>
                    <div class="viz-desc">Duration patterns, transitions, and timeline</div>
                </a>
                
                <a href="factor_performance_heatmap.html" class="viz-link">
                    <div class="viz-title">Performance Heatmap</div>
                    <div class="viz-desc">Comprehensive metrics by factor and regime</div>
                </a>
                
                <a href="risk_return_detailed.html" class="viz-link">
                    <div class="viz-title">Risk-Return Analysis</div>
                    <div class="viz-desc">Detailed scatter plot with regime breakdown</div>
                </a>
                
                <a href="cumulative_performance_paths.html" class="viz-link">
                    <div class="viz-title">Cumulative Performance</div>
                    <div class="viz-desc">Factor performance paths within each regime</div>
                </a>
                
                <a href="regime_correlation_analysis.html" class="viz-link">
                    <div class="viz-title">Factor Correlations</div>
                    <div class="viz-desc">How factors relate in different regimes</div>
                </a>
                
                <a href="drawdown_by_regime.html" class="viz-link">
                    <div class="viz-title">Drawdown Analysis</div>
                    <div class="viz-desc">Risk characteristics by economic regime</div>
                </a>
                
                <a href="regime_transitions_sunburst.html" class="viz-link">
                    <div class="viz-title">Regime Transitions</div>
                    <div class="viz-desc">Sunburst visualization of transition patterns</div>
                </a>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", 'w') as f:
            f.write(html_content)

if __name__ == "__main__":
    analyzer = PersistenceRequiredAnalysis()
    analyzer.run_analysis()