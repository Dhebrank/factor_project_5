#!/usr/bin/env python3
"""
Create a simple, single risk-return scatter plot
Clean institutional-style visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import json
from datetime import datetime

class SimpleRiskReturnScatterCreator:
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
        
    def load_data(self):
        """Load and prepare data"""
        df = pd.read_csv(self.aligned_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Get risk-free rate
        if 'DGS2' in df.columns:
            self.rf_rate = df['DGS2'].dropna().mean() / 100
        else:
            self.rf_rate = 0.0235
        
        # Convert SP500 from price to returns if needed
        if 'SP500' in df.columns and df['SP500'].min() > 10:
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
    
    def calculate_metrics(self, returns):
        """Calculate risk-return metrics"""
        if len(returns) < 2:
            return None
            
        returns = returns.dropna()
        if len(returns) < 2:
            return None
        
        # Monthly statistics
        monthly_mean = returns.mean()
        monthly_std = returns.std()
        
        # Annualize
        annual_return = (1 + monthly_mean) ** 12 - 1
        annual_vol = monthly_std * np.sqrt(12)
        
        # Calculate Sharpe ratio with risk-free rate
        monthly_rf = (1 + self.rf_rate) ** (1/12) - 1
        excess_return = monthly_mean - monthly_rf
        annual_excess = (1 + excess_return) ** 12 - 1
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
    
    def create_simple_scatter_plot(self, df):
        """Create a simple, clean risk-return scatter plot"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500']
        
        # Calculate metrics for each factor
        scatter_data = []
        for factor in factors:
            returns = df[factor].dropna()
            metrics = self.calculate_metrics(returns)
            if metrics:
                scatter_data.append({
                    'Factor': factor,
                    'Return': metrics['annual_return'],
                    'Risk': metrics['annual_volatility'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'MaxDD': metrics['max_drawdown'],
                    'Obs': metrics['observations']
                })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points for each factor
        for _, row in scatter_df.iterrows():
            # Size based on Sharpe ratio (scaled for visibility)
            size = max(20, min(60, 30 + row['Sharpe'] * 20))
            
            fig.add_trace(
                go.Scatter(
                    x=[row['Risk']],
                    y=[row['Return']],
                    mode='markers+text',
                    name=row['Factor'],
                    text=[row['Factor']],
                    textposition='top center',
                    textfont=dict(size=12),
                    marker=dict(
                        size=size,
                        color=self.factor_colors[row['Factor']],
                        line=dict(width=2, color='white'),
                        symbol='circle'
                    ),
                    hovertemplate=(
                        f"<b>{row['Factor']}</b><br>" +
                        f"Annual Return: {row['Return']:.2f}%<br>" +
                        f"Annual Volatility: {row['Risk']:.2f}%<br>" +
                        f"Sharpe Ratio: {row['Sharpe']:.3f}<br>" +
                        f"Max Drawdown: {row['MaxDD']:.2f}%<br>" +
                        f"Observations: {row['Obs']}<br>" +
                        "<extra></extra>"
                    )
                )
            )
        
        # Add risk-free rate line
        fig.add_hline(
            y=self.rf_rate * 100,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Risk-Free Rate ({self.rf_rate*100:.1f}%)",
            annotation_position="right"
        )
        
        # Add Sharpe ratio reference lines
        risk_range = np.linspace(0, max(scatter_df['Risk']) * 1.2, 100)
        
        for sharpe, color, style in [(0.5, 'gray', 'dot'), (1.0, 'darkgray', 'dash')]:
            returns = self.rf_rate * 100 + sharpe * risk_range
            fig.add_trace(
                go.Scatter(
                    x=risk_range,
                    y=returns,
                    mode='lines',
                    line=dict(dash=style, color=color, width=1),
                    showlegend=False,
                    hovertemplate=f'Sharpe={sharpe}<br>Risk: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>'
                )
            )
            
            # Add annotation for Sharpe line
            x_pos = max(scatter_df['Risk']) * 0.8
            y_pos = self.rf_rate * 100 + sharpe * x_pos
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=f"Sharpe = {sharpe}",
                showarrow=False,
                font=dict(size=10, color=color),
                textangle=-20 if sharpe == 0.5 else -35
            )
        
        # Add quadrant labels
        mean_risk = scatter_df['Risk'].mean()
        mean_return = scatter_df['Return'].mean()
        
        quadrant_labels = [
            {"x": mean_risk * 0.3, "y": mean_return * 1.7, "text": "Low Risk<br>High Return", "color": "green"},
            {"x": mean_risk * 1.7, "y": mean_return * 1.7, "text": "High Risk<br>High Return", "color": "orange"},
            {"x": mean_risk * 0.3, "y": mean_return * 0.3, "text": "Low Risk<br>Low Return", "color": "gray"},
            {"x": mean_risk * 1.7, "y": mean_return * 0.3, "text": "High Risk<br>Low Return", "color": "red"}
        ]
        
        for label in quadrant_labels:
            fig.add_annotation(
                x=label["x"],
                y=label["y"],
                text=label["text"],
                showarrow=False,
                font=dict(size=9, color=label["color"]),
                opacity=0.5
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Risk-Return Profile<br><sub>Persistence-Required Analysis | Risk-Free Rate: {self.rf_rate*100:.1f}%</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title="Annual Volatility (%)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                range=[0, max(scatter_df['Risk']) * 1.3]
            ),
            yaxis=dict(
                title="Annual Return (%)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                range=[min(0, min(scatter_df['Return']) * 1.1), max(scatter_df['Return']) * 1.2]
            ),
            height=700,
            width=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        )
        
        # Add summary statistics table
        summary_text = "<b>Summary Statistics</b><br>"
        summary_text += "-" * 30 + "<br>"
        
        # Sort by Sharpe ratio
        scatter_df_sorted = scatter_df.sort_values('Sharpe', ascending=False)
        
        for _, row in scatter_df_sorted.iterrows():
            summary_text += f"<b>{row['Factor']}</b>: "
            summary_text += f"Return={row['Return']:.1f}%, "
            summary_text += f"Vol={row['Risk']:.1f}%, "
            summary_text += f"Sharpe={row['Sharpe']:.2f}<br>"
        
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=summary_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=10
        )
        
        # Save the plot
        fig.write_html(self.output_dir / "risk_return_simple_scatter.html")
        
        # Also save as static image for reports
        try:
            fig.write_image(self.output_dir / "risk_return_simple_scatter.png", 
                          width=900, height=700, scale=2)
            print("Also saved as PNG image")
        except:
            print("Could not save PNG (kaleido not installed)")
        
        return fig
    
    def run(self):
        """Run the analysis"""
        print("Creating simple risk-return scatter plot...")
        
        # Load data
        df = self.load_data()
        
        # Create scatter plot
        self.create_simple_scatter_plot(df)
        
        print(f"Simple risk-return scatter plot saved to: {self.output_dir}/risk_return_simple_scatter.html")

if __name__ == "__main__":
    creator = SimpleRiskReturnScatterCreator()
    creator.run()