#!/usr/bin/env python3
"""
Create missing visualizations for persistence-required analysis
Matches the visualizations from business_cycle_analysis folder
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

class MissingVisualizationsCreator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "results" / "persistence_required_analysis_corrected"
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
        
        # Get risk-free rate
        if 'DGS2' in df.columns:
            self.rf_rate = df['DGS2'].dropna().mean() / 100
        else:
            self.rf_rate = 0.0235
        
        # Convert SP500 from price to returns
        if 'SP500' in df.columns:
            df['SP500_Price'] = df['SP500']
            df['SP500'] = df['SP500_Price'].pct_change()
        
        # Apply persistence requirement
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
    
    def create_comprehensive_dashboard(self, df):
        """Create comprehensive business cycle dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Regime Timeline', 'Factor Performance by Regime', 'Current Regime Status',
                'Transition Probabilities', 'Risk-Return Scatter', 'Rolling Sharpe Ratios',
                'Factor Correlations', 'Regime Duration Stats', 'Cumulative Performance'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'indicator'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'scatter'}]
            ],
            row_heights=[0.3, 0.35, 0.35],
            column_widths=[0.35, 0.35, 0.3]
        )
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # 1. Regime Timeline
        regime_numeric = pd.Categorical(df['Regime_Persistence'], 
                                      categories=['Recession', 'Stagflation', 'Overheating', 'Goldilocks'],
                                      ordered=True).codes
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=regime_numeric,
                mode='lines',
                fill='tozeroy',
                name='Regime',
                line=dict(width=0),
                fillcolor='rgba(0,100,200,0.3)'
            ),
            row=1, col=1
        )
        
        # 2. Factor Performance by Regime
        perf_data = []
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            for factor in factors:
                returns = df.loc[regime_mask, factor].dropna()
                if len(returns) > 1:
                    annual_return = (1 + returns.mean()) ** 12 - 1
                    perf_data.append({
                        'Regime': regime,
                        'Factor': factor,
                        'Return': annual_return * 100
                    })
        
        perf_df = pd.DataFrame(perf_data)
        for factor in factors:
            factor_data = perf_df[perf_df['Factor'] == factor]
            fig.add_trace(
                go.Bar(
                    x=factor_data['Regime'],
                    y=factor_data['Return'],
                    name=factor,
                    marker_color=self.factor_colors[factor],
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Current Regime Status
        current_regime = df['Regime_Persistence'].iloc[-1]
        fig.add_trace(
            go.Indicator(
                mode="number+delta+gauge",
                value=regime_numeric[-1],
                title={'text': f"Current: {current_regime}"},
                delta={'reference': regime_numeric[-30]},
                gauge={
                    'axis': {'range': [0, 3]},
                    'bar': {'color': self.regime_colors[current_regime]},
                    'steps': [
                        {'range': [0, 1], 'color': self.regime_colors['Recession']},
                        {'range': [1, 2], 'color': self.regime_colors['Stagflation']},
                        {'range': [2, 3], 'color': self.regime_colors['Overheating']},
                        {'range': [3, 4], 'color': self.regime_colors['Goldilocks']}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # 4. Transition Probabilities
        transitions = pd.DataFrame(index=self.regime_colors.keys(), columns=self.regime_colors.keys(), data=0)
        regime_changes = df['Regime_Persistence'] != df['Regime_Persistence'].shift(1)
        regime_blocks = regime_changes.cumsum()
        
        for i in range(1, regime_blocks.max()):
            block_data = df[regime_blocks == i]
            from_regime = block_data['Regime_Persistence'].iloc[0]
            if i < regime_blocks.max():
                next_block = df[regime_blocks == i + 1]
                to_regime = next_block['Regime_Persistence'].iloc[0]
                transitions.loc[from_regime, to_regime] += 1
        
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
            row=2, col=1
        )
        
        # 5. Risk-Return Scatter
        scatter_data = []
        for factor in factors:
            returns = df[factor].dropna()
            if len(returns) > 1:
                annual_return = (1 + returns.mean()) ** 12 - 1
                annual_vol = returns.std() * np.sqrt(12)
                monthly_rf = (1 + self.rf_rate) ** (1/12) - 1
                excess_return = returns.mean() - monthly_rf
                annual_excess = (1 + excess_return) ** 12 - 1
                sharpe = annual_excess / annual_vol if annual_vol > 0 else 0
                
                scatter_data.append({
                    'Factor': factor,
                    'Return': annual_return * 100,
                    'Risk': annual_vol * 100,
                    'Sharpe': sharpe
                })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig.add_trace(
            go.Scatter(
                x=scatter_df['Risk'],
                y=scatter_df['Return'],
                mode='markers+text',
                text=scatter_df['Factor'],
                textposition='top center',
                marker=dict(
                    size=scatter_df['Sharpe'] * 20 + 10,
                    color=[self.factor_colors[f] for f in scatter_df['Factor']]
                ),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Rolling Sharpe Ratios
        window = 36  # 3-year rolling
        for factor in factors:
            returns = df[factor].dropna()
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            
            # Calculate rolling Sharpe with risk-free rate
            monthly_rf = (1 + self.rf_rate) ** (1/12) - 1
            rolling_excess = rolling_mean - monthly_rf
            rolling_sharpe = (rolling_excess * 12) / (rolling_std * np.sqrt(12))
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_sharpe,
                    name=factor,
                    line=dict(color=self.factor_colors[factor]),
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # 7. Factor Correlations (Overall)
        corr_matrix = df[factors].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=factors,
                y=factors,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                showscale=False
            ),
            row=3, col=1
        )
        
        # 8. Regime Duration Stats
        durations = []
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            if regime_mask.sum() > 0:
                # Count consecutive occurrences
                regime_changes = regime_mask != regime_mask.shift(1)
                regime_groups = regime_changes.cumsum()
                regime_durations = regime_mask.groupby(regime_groups).sum()
                regime_durations = regime_durations[regime_durations > 0]
                
                if len(regime_durations) > 0:
                    durations.append({
                        'Regime': regime,
                        'Avg_Duration': regime_durations.mean(),
                        'Count': len(regime_durations)
                    })
        
        duration_df = pd.DataFrame(durations)
        
        fig.add_trace(
            go.Bar(
                x=duration_df['Regime'],
                y=duration_df['Avg_Duration'],
                marker_color=[self.regime_colors[r] for r in duration_df['Regime']],
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 9. Cumulative Performance
        for factor in factors:
            returns = df[factor].fillna(0)
            cumulative = (1 + returns).cumprod()
            cumulative = cumulative / cumulative.iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=cumulative,
                    name=factor,
                    line=dict(color=self.factor_colors[factor]),
                    showlegend=False
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=f"Comprehensive Business Cycle Dashboard (Persistence-Required, RF={self.rf_rate*100:.1f}%)",
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=1, col=1)
        fig.update_xaxes(title_text="Regime", row=1, col=2)
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="To Regime", row=2, col=1)
        fig.update_yaxes(title_text="From Regime", row=2, col=1)
        fig.update_xaxes(title_text="Annual Volatility (%)", row=2, col=2)
        fig.update_yaxes(title_text="Annual Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=3)
        fig.update_yaxes(title_text="Rolling Sharpe Ratio", row=2, col=3)
        fig.update_xaxes(title_text="Factor", row=3, col=1)
        fig.update_yaxes(title_text="Factor", row=3, col=1)
        fig.update_xaxes(title_text="Regime", row=3, col=2)
        fig.update_yaxes(title_text="Avg Duration (Months)", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=3)
        fig.update_yaxes(title_text="Cumulative Return (Base=100)", row=3, col=3)
        
        fig.write_html(self.output_dir / "comprehensive_business_cycle_dashboard.html")
        
    def create_factor_rotation_wheel(self, df):
        """Create factor rotation wheel visualization"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate 12-month rolling performance for each factor
        window = 12
        rolling_returns = {}
        
        for factor in factors:
            returns = df[factor].dropna()
            rolling_mean = returns.rolling(window).mean()
            annual_rolling = (1 + rolling_mean) ** 12 - 1
            rolling_returns[factor] = annual_rolling * 100
        
        # Find which factor is leading at each point
        rolling_df = pd.DataFrame(rolling_returns).dropna()
        leading_factor = rolling_df.idxmax(axis=1)
        
        # Count transitions
        transitions = []
        for i in range(1, len(leading_factor)):
            if leading_factor.iloc[i] != leading_factor.iloc[i-1]:
                transitions.append({
                    'from': leading_factor.iloc[i-1],
                    'to': leading_factor.iloc[i],
                    'date': leading_factor.index[i],
                    'regime': df.loc[leading_factor.index[i], 'Regime_Persistence']
                })
        
        # Create circular layout
        angles = {
            'Value': 0,
            'Quality': 90,
            'MinVol': 180,
            'Momentum': 270
        }
        
        fig = go.Figure()
        
        # Add factor nodes
        for factor, angle in angles.items():
            rad = np.radians(angle)
            x = np.cos(rad)
            y = np.sin(rad)
            
            # Count how often this factor leads
            lead_count = (leading_factor == factor).sum()
            lead_pct = lead_count / len(leading_factor) * 100
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=50 + lead_pct,
                    color=self.factor_colors[factor]
                ),
                text=[f"{factor}<br>{lead_pct:.1f}%"],
                textposition='middle center',
                name=factor,
                showlegend=False
            ))
        
        # Add transitions as arrows
        transition_counts = {}
        for trans in transitions:
            key = f"{trans['from']}->{trans['to']}"
            if key not in transition_counts:
                transition_counts[key] = 0
            transition_counts[key] += 1
        
        # Draw arrows for transitions
        for trans_key, count in transition_counts.items():
            from_factor, to_factor = trans_key.split('->')
            
            # Get positions
            from_rad = np.radians(angles[from_factor])
            to_rad = np.radians(angles[to_factor])
            
            x0, y0 = np.cos(from_rad) * 0.8, np.sin(from_rad) * 0.8
            x1, y1 = np.cos(to_rad) * 0.8, np.sin(to_rad) * 0.8
            
            # Create curved path
            mid_angle = (angles[from_factor] + angles[to_factor]) / 2
            if abs(angles[from_factor] - angles[to_factor]) > 180:
                mid_angle += 180
            
            mid_rad = np.radians(mid_angle)
            mid_x = np.cos(mid_rad) * 0.5
            mid_y = np.sin(mid_rad) * 0.5
            
            # Generate curve points
            t = np.linspace(0, 1, 20)
            x_curve = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
            y_curve = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
            
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_curve,
                mode='lines',
                line=dict(
                    width=count/2,
                    color='rgba(128,128,128,0.3)'
                ),
                hovertext=f"{trans_key}: {count} transitions",
                hoverinfo='text',
                showlegend=False
            ))
        
        # Add regime breakdown
        regime_leadership = {}
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            regime_leader = leading_factor[regime_mask]
            if len(regime_leader) > 0:
                regime_leadership[regime] = regime_leader.value_counts(normalize=True).to_dict()
        
        # Add regime info as annotations
        annotation_text = "<b>Leadership by Regime:</b><br>"
        for regime, leaders in regime_leadership.items():
            annotation_text += f"<br><b>{regime}:</b><br>"
            for factor, pct in sorted(leaders.items(), key=lambda x: x[1], reverse=True):
                annotation_text += f"  {factor}: {pct*100:.1f}%<br>"
        
        fig.add_annotation(
            x=1.5,
            y=1,
            text=annotation_text,
            showarrow=False,
            xanchor='left',
            align='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_layout(
            title="Factor Rotation Wheel (Persistence-Required)",
            xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            plot_bgcolor='white'
        )
        
        fig.write_html(self.output_dir / "factor_rotation_wheel.html")
    
    def create_interactive_timeline(self, df):
        """Create interactive timeline with regime overlay"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=[
                'Factor Performance with Regime Overlay',
                'Economic Indicators',
                'Regime Probability',
                'Factor Leadership'
            ],
            vertical_spacing=0.05
        )
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # 1. Factor Performance with Regime Overlay
        for factor in factors:
            cumulative = (1 + df[factor].fillna(0)).cumprod()
            cumulative = cumulative / cumulative.iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=cumulative,
                    name=factor,
                    line=dict(color=self.factor_colors[factor]),
                    legendgroup='factors'
                ),
                row=1, col=1
            )
        
        # Add regime shading
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            regime_periods = []
            
            # Find continuous periods
            in_regime = False
            start = None
            
            for i, (date, is_regime) in enumerate(regime_mask.items()):
                if is_regime and not in_regime:
                    start = date
                    in_regime = True
                elif not is_regime and in_regime:
                    regime_periods.append((start, df.index[i-1]))
                    in_regime = False
            
            if in_regime:
                regime_periods.append((start, df.index[-1]))
            
            # Add shaded regions
            for start, end in regime_periods:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=self.regime_colors[regime],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
        
        # 2. Economic Indicators
        if 'GROWTH_COMPOSITE' in df.columns and 'INFLATION_COMPOSITE' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['GROWTH_COMPOSITE'],
                    name='Growth Composite',
                    line=dict(color='green'),
                    legendgroup='indicators'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['INFLATION_COMPOSITE'],
                    name='Inflation Composite',
                    line=dict(color='red'),
                    legendgroup='indicators'
                ),
                row=2, col=1
            )
        
        # 3. Regime Probability (simplified view)
        regime_numeric = pd.Categorical(df['Regime_Persistence'], 
                                      categories=['Recession', 'Stagflation', 'Overheating', 'Goldilocks'],
                                      ordered=True).codes
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=regime_numeric,
                mode='lines',
                fill='tozeroy',
                name='Regime State',
                line=dict(color='black', width=2),
                legendgroup='regime'
            ),
            row=3, col=1
        )
        
        # 4. Factor Leadership
        # Calculate rolling best performer
        window = 12
        rolling_returns = pd.DataFrame()
        
        for factor in factors:
            returns = df[factor].dropna()
            rolling_mean = returns.rolling(window).mean()
            rolling_returns[factor] = rolling_mean
        
        leading_factor = rolling_returns.idxmax(axis=1)
        factor_numeric = pd.Categorical(leading_factor, categories=factors).codes
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=factor_numeric,
                mode='lines',
                name='Leading Factor',
                line=dict(width=3),
                legendgroup='leadership'
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive Timeline with Regime Overlay (Persistence-Required)",
            height=1000,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Cumulative Return (Base=100)", row=1, col=1)
        fig.update_yaxes(title_text="Composite Score", row=2, col=1)
        fig.update_yaxes(title_text="Regime", row=3, col=1)
        fig.update_yaxes(title_text="Leading Factor", row=4, col=1)
        
        # Add regime labels to y-axis
        fig.update_yaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Recession', 'Stagflation', 'Overheating', 'Goldilocks'],
            row=3, col=1
        )
        
        fig.update_yaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=factors,
            row=4, col=1
        )
        
        fig.write_html(self.output_dir / "interactive_timeline_regime_overlay.html")
    
    def create_momentum_persistence_analysis(self, df):
        """Create momentum persistence analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Momentum Win Rate by Regime',
                'Momentum Persistence Periods',
                'Momentum vs Other Factors',
                'Momentum Drawdown Analysis'
            ]
        )
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # 1. Momentum Win Rate by Regime
        win_rates = []
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            regime_data = df[regime_mask]
            
            if len(regime_data) > 1:
                # Calculate win rate (momentum beats other factors)
                wins = 0
                total = 0
                
                for _, row in regime_data.iterrows():
                    factor_returns = [row[f] for f in factors if pd.notna(row[f])]
                    if len(factor_returns) == len(factors):
                        if row['Momentum'] == max(factor_returns):
                            wins += 1
                        total += 1
                
                if total > 0:
                    win_rates.append({
                        'Regime': regime,
                        'Win_Rate': wins / total * 100,
                        'Observations': total
                    })
        
        win_rate_df = pd.DataFrame(win_rates)
        
        fig.add_trace(
            go.Bar(
                x=win_rate_df['Regime'],
                y=win_rate_df['Win_Rate'],
                marker_color=[self.regime_colors[r] for r in win_rate_df['Regime']],
                text=win_rate_df['Win_Rate'].round(1),
                texttemplate='%{text}%',
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Momentum Persistence Periods
        # Calculate consecutive outperformance periods
        momentum_returns = df['Momentum'].dropna()
        avg_returns = df[factors].mean(axis=1)
        momentum_outperform = momentum_returns > avg_returns
        
        # Find streaks
        streaks = []
        current_streak = 0
        streak_start = None
        
        for date, outperform in momentum_outperform.items():
            if outperform:
                if current_streak == 0:
                    streak_start = date
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append({
                        'Start': streak_start,
                        'End': date,
                        'Length': current_streak,
                        'Regime': df.loc[streak_start:date, 'Regime_Persistence'].mode()[0]
                    })
                current_streak = 0
        
        if current_streak > 0:
            streaks.append({
                'Start': streak_start,
                'End': df.index[-1],
                'Length': current_streak,
                'Regime': df.loc[streak_start:, 'Regime_Persistence'].mode()[0]
            })
        
        # Plot histogram of streak lengths by regime
        streak_df = pd.DataFrame(streaks)
        if not streak_df.empty:
            for regime in self.regime_colors.keys():
                regime_streaks = streak_df[streak_df['Regime'] == regime]['Length']
                if len(regime_streaks) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=regime_streaks,
                            name=regime,
                            marker_color=self.regime_colors[regime],
                            opacity=0.7,
                            nbinsx=20
                        ),
                        row=1, col=2
                    )
        
        # 3. Momentum vs Other Factors
        scatter_data = []
        for factor in ['Value', 'Quality', 'MinVol']:
            # Calculate correlation with momentum
            corr = df[factor].corr(df['Momentum'])
            
            # Calculate average returns
            factor_return = (1 + df[factor].mean()) ** 12 - 1
            momentum_return = (1 + df['Momentum'].mean()) ** 12 - 1
            
            scatter_data.append({
                'Factor': factor,
                'Factor_Return': factor_return * 100,
                'Momentum_Return': momentum_return * 100,
                'Correlation': corr
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig.add_trace(
            go.Scatter(
                x=scatter_df['Factor_Return'],
                y=scatter_df['Momentum_Return'],
                mode='markers+text',
                text=scatter_df['Factor'],
                textposition='top center',
                marker=dict(
                    size=50,
                    color=scatter_df['Correlation'],
                    colorscale='RdBu',
                    colorbar=dict(title='Correlation'),
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # 4. Momentum Drawdown Analysis
        # Calculate drawdowns
        momentum_cumulative = (1 + momentum_returns).cumprod()
        rolling_max = momentum_cumulative.expanding().max()
        drawdown = (momentum_cumulative - rolling_max) / rolling_max * 100
        
        # Color by regime
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            regime_dd = drawdown[regime_mask]
            
            if len(regime_dd) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=regime_dd.index,
                        y=regime_dd.values,
                        mode='lines',
                        name=regime,
                        line=dict(color=self.regime_colors[regime]),
                        fill='tozeroy',
                        fillcolor=self.regime_colors[regime],
                        opacity=0.3
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Momentum Persistence Analysis (Persistence-Required)",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Streak Length (Months)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Factor Annual Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Momentum Annual Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
        
        fig.write_html(self.output_dir / "momentum_persistence_analysis.html")
    
    def create_relative_performance_heatmap(self, df):
        """Create relative performance heatmap"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Calculate relative performance matrix for each regime
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.regime_colors.keys()),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, regime in enumerate(self.regime_colors.keys()):
            row, col = positions[idx]
            regime_mask = df['Regime_Persistence'] == regime
            regime_data = df[regime_mask]
            
            if len(regime_data) > 10:  # Need sufficient data
                # Calculate average returns for each factor
                avg_returns = {}
                for factor in factors:
                    returns = regime_data[factor].dropna()
                    if len(returns) > 0:
                        avg_returns[factor] = (1 + returns.mean()) ** 12 - 1
                
                # Create relative performance matrix
                rel_perf = np.zeros((len(factors), len(factors)))
                
                for i, factor1 in enumerate(factors):
                    for j, factor2 in enumerate(factors):
                        if factor1 in avg_returns and factor2 in avg_returns:
                            if avg_returns[factor2] != 0:
                                rel_perf[i, j] = (avg_returns[factor1] - avg_returns[factor2]) / abs(avg_returns[factor2]) * 100
                            else:
                                rel_perf[i, j] = 0
                
                fig.add_trace(
                    go.Heatmap(
                        z=rel_perf,
                        x=factors,
                        y=factors,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=rel_perf.round(1),
                        texttemplate='%{text}%',
                        showscale=(idx == 0),
                        colorbar=dict(title='Relative<br>Perf %')
                    ),
                    row=row, col=col
                )
                
                # Add regime info
                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    text=f"n={regime_mask.sum()} months",
                    xref=f"x{idx+1} domain" if idx > 0 else "x domain",
                    yref=f"y{idx+1} domain" if idx > 0 else "y domain",
                    showarrow=False,
                    font=dict(size=10),
                    xshift=0,
                    yshift=0
                )
        
        fig.update_layout(
            title="Relative Performance Heatmap by Regime (Row vs Column)",
            height=800
        )
        
        fig.write_html(self.output_dir / "relative_performance_heatmap.html")
    
    def create_risk_adjusted_heatmap(self, df):
        """Create risk-adjusted performance heatmap"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        # Calculate metrics for each factor-regime combination
        results = {}
        
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            results[regime] = {}
            
            for factor in factors:
                factor_data = df.loc[regime_mask, factor].dropna()
                
                if len(factor_data) > 12:  # Need at least 1 year of data
                    returns = factor_data
                    
                    # Calculate metrics
                    monthly_mean = returns.mean()
                    monthly_std = returns.std()
                    downside_returns = returns[returns < 0]
                    downside_std = downside_returns.std() if len(downside_returns) > 0 else monthly_std
                    
                    # Annualize
                    annual_return = (1 + monthly_mean) ** 12 - 1
                    annual_vol = monthly_std * np.sqrt(12)
                    annual_downside_vol = downside_std * np.sqrt(12)
                    
                    # Risk-free adjustment
                    monthly_rf = (1 + self.rf_rate) ** (1/12) - 1
                    excess_return = monthly_mean - monthly_rf
                    annual_excess = (1 + excess_return) ** 12 - 1
                    
                    # Sharpe Ratio
                    sharpe = annual_excess / annual_vol if annual_vol > 0 else 0
                    
                    # Sortino Ratio
                    sortino = annual_excess / annual_downside_vol if annual_downside_vol > 0 else 0
                    
                    # Calmar Ratio (return / max drawdown)
                    cumulative = (1 + returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_dd = abs(drawdown.min())
                    calmar = annual_return / max_dd if max_dd > 0 else 0
                    
                    # Information Ratio (vs equal-weight portfolio)
                    benchmark_returns = df.loc[regime_mask, factors].mean(axis=1)
                    active_returns = returns - benchmark_returns[returns.index]
                    tracking_error = active_returns.std() * np.sqrt(12)
                    info_ratio = (active_returns.mean() * 12) / tracking_error if tracking_error > 0 else 0
                    
                    results[regime][factor] = {
                        'Sharpe Ratio': sharpe,
                        'Sortino Ratio': sortino,
                        'Calmar Ratio': calmar,
                        'Information Ratio': info_ratio
                    }
        
        # Create heatmaps for each metric
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # Create matrix
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
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f'{val:.2f}' for val in row] for row in matrix],
                    texttemplate='%{text}',
                    showscale=(idx == 0)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Risk-Adjusted Performance Metrics (RF={self.rf_rate*100:.1f}%)",
            height=800
        )
        
        fig.write_html(self.output_dir / "risk_adjusted_heatmap.html")
    
    def create_rolling_regime_analysis(self, df):
        """Create rolling regime analysis"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=[
                'Rolling 3-Year Sharpe Ratios',
                'Rolling Regime Stability',
                'Rolling Factor Correlations'
            ]
        )
        
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        window = 36  # 3-year rolling window
        
        # 1. Rolling Sharpe Ratios
        for factor in factors:
            returns = df[factor].dropna()
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            
            # Calculate rolling Sharpe with risk-free rate
            monthly_rf = (1 + self.rf_rate) ** (1/12) - 1
            rolling_excess = rolling_mean - monthly_rf
            rolling_sharpe = (rolling_excess * 12) / (rolling_std * np.sqrt(12))
            
            # Color by current regime
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_sharpe,
                    name=factor,
                    line=dict(color=self.factor_colors[factor], width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Add regime shading
        for regime in self.regime_colors.keys():
            regime_mask = df['Regime_Persistence'] == regime
            regime_periods = []
            
            in_regime = False
            start = None
            
            for i, (date, is_regime) in enumerate(regime_mask.items()):
                if is_regime and not in_regime:
                    start = date
                    in_regime = True
                elif not is_regime and in_regime:
                    regime_periods.append((start, df.index[i-1]))
                    in_regime = False
            
            if in_regime:
                regime_periods.append((start, df.index[-1]))
            
            for start, end in regime_periods:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=self.regime_colors[regime],
                    opacity=0.1,
                    layer="below",
                    line_width=0
                )
        
        # 2. Rolling Regime Stability
        # Calculate how long current regime has persisted
        regime_duration = []
        current_regime = df['Regime_Persistence'].iloc[0]
        duration_count = 1
        
        for i in range(1, len(df)):
            if df['Regime_Persistence'].iloc[i] == current_regime:
                duration_count += 1
            else:
                current_regime = df['Regime_Persistence'].iloc[i]
                duration_count = 1
            regime_duration.append(duration_count)
        
        regime_duration.insert(0, 1)  # First observation
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=regime_duration,
                mode='lines',
                fill='tozeroy',
                name='Regime Duration',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        # 3. Rolling Factor Correlations
        # Calculate average pairwise correlation
        rolling_corr = []
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i][factors]
            corr_matrix = window_data.corr()
            
            # Get average off-diagonal correlation
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            avg_corr = corr_matrix.values[mask].mean()
            rolling_corr.append(avg_corr)
        
        rolling_corr_series = pd.Series(rolling_corr, index=df.index[window:])
        
        fig.add_trace(
            go.Scatter(
                x=rolling_corr_series.index,
                y=rolling_corr_series.values,
                mode='lines',
                name='Avg Factor Correlation',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.2)'
            ),
            row=3, col=1
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title="Rolling Regime Analysis (3-Year Windows)",
            height=900,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Months in Regime", row=2, col=1)
        fig.update_yaxes(title_text="Average Correlation", row=3, col=1)
        
        fig.write_html(self.output_dir / "rolling_regime_analysis.html")
    
    def create_correlation_matrices_by_regime(self, df):
        """Create correlation matrices for each regime"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(self.regime_colors.keys()),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, regime in enumerate(self.regime_colors.keys()):
            row, col = positions[idx]
            regime_mask = df['Regime_Persistence'] == regime
            
            if regime_mask.sum() > 30:  # Need sufficient observations
                # Calculate returns for this regime
                regime_returns = df.loc[regime_mask, factors]
                
                # Calculate correlation matrix
                corr_matrix = regime_returns.corr()
                
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
                
                # Add observation count
                fig.add_annotation(
                    x=0.5,
                    y=-0.15,
                    text=f"n={regime_mask.sum()} months",
                    xref=f"x{idx+1} domain" if idx > 0 else "x domain",
                    yref=f"y{idx+1} domain" if idx > 0 else "y domain",
                    showarrow=False,
                    font=dict(size=10),
                    xshift=0,
                    yshift=0
                )
        
        fig.update_layout(
            title="Factor Correlation Matrices by Economic Regime",
            height=800
        )
        
        fig.write_html(self.output_dir / "correlation_matrices_by_regime.html")
    
    def run_all(self):
        """Run all visualization creation"""
        print("Creating missing visualizations for persistence-required analysis...")
        
        # Load data
        df = self.load_data()
        
        print("1. Creating comprehensive dashboard...")
        self.create_comprehensive_dashboard(df)
        
        print("2. Creating factor rotation wheel...")
        self.create_factor_rotation_wheel(df)
        
        print("3. Creating interactive timeline...")
        self.create_interactive_timeline(df)
        
        print("4. Creating momentum persistence analysis...")
        self.create_momentum_persistence_analysis(df)
        
        print("5. Creating relative performance heatmap...")
        self.create_relative_performance_heatmap(df)
        
        print("6. Creating risk-adjusted heatmap...")
        self.create_risk_adjusted_heatmap(df)
        
        print("7. Creating rolling regime analysis...")
        self.create_rolling_regime_analysis(df)
        
        print("8. Creating correlation matrices by regime...")
        self.create_correlation_matrices_by_regime(df)
        
        print(f"\nAll visualizations created in: {self.output_dir}")

if __name__ == "__main__":
    creator = MissingVisualizationsCreator()
    creator.run_all()