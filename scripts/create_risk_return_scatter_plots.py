#!/usr/bin/env python3
"""
Create risk-return scatter plots (multiple views)
Matches risk_return_scatter_plots.html from business_cycle_analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

class RiskReturnScatterPlotsCreator:
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
    
    def calculate_metrics(self, returns, rf_rate=None):
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
        
        # Calculate Sharpe ratio
        if rf_rate is None:
            rf_rate = self.rf_rate
        monthly_rf = (1 + rf_rate) ** (1/12) - 1
        excess_return = monthly_mean - monthly_rf
        annual_excess = (1 + excess_return) ** 12 - 1
        sharpe_ratio = annual_excess / annual_vol if annual_vol > 0 else 0
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else annual_vol
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'downside_volatility': downside_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'observations': len(returns)
        }
    
    def create_risk_return_scatter_plots(self, df):
        """Create multiple risk-return scatter plot views"""
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Overall Risk-Return Profile',
                'By Economic Regime',
                'Risk vs Downside Risk',
                'Pre-2008 vs Post-2008',
                'Bull vs Bear Markets',
                'Rolling 5-Year Windows'
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        # 1. Overall Risk-Return Profile
        overall_data = []
        for factor in factors:
            returns = df[factor].dropna()
            metrics = self.calculate_metrics(returns)
            if metrics:
                overall_data.append({
                    'Factor': factor,
                    'Return': metrics['annual_return'],
                    'Risk': metrics['annual_volatility'],
                    'Sharpe': metrics['sharpe_ratio']
                })
        
        if overall_data:
            overall_df = pd.DataFrame(overall_data)
            
            # Add scatter points
            for _, row in overall_df.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row['Risk']],
                        y=[row['Return']],
                        mode='markers+text',
                        name=row['Factor'],
                        text=[row['Factor']],
                        textposition='top center',
                        marker=dict(
                            size=20,
                            color=self.factor_colors[row['Factor']],
                            line=dict(width=2, color='white')
                        ),
                        showlegend=True,
                        legendgroup=row['Factor'],
                        hovertemplate=f"{row['Factor']}<br>Return: {row['Return']:.1f}%<br>Risk: {row['Risk']:.1f}%<br>Sharpe: {row['Sharpe']:.2f}"
                    ),
                    row=1, col=1
                )
            
            # Add efficient frontier reference
            self._add_efficient_frontier(fig, 1, 1)
        
        # 2. By Economic Regime
        for regime, color in self.regime_colors.items():
            regime_mask = df['Regime_Persistence'] == regime
            regime_data = []
            
            for factor in factors:
                returns = df.loc[regime_mask, factor].dropna()
                if len(returns) > 12:  # At least 1 year of data
                    metrics = self.calculate_metrics(returns)
                    if metrics:
                        regime_data.append({
                            'Factor': factor,
                            'Return': metrics['annual_return'],
                            'Risk': metrics['annual_volatility'],
                            'Regime': regime
                        })
            
            if regime_data:
                regime_df = pd.DataFrame(regime_data)
                
                fig.add_trace(
                    go.Scatter(
                        x=regime_df['Risk'],
                        y=regime_df['Return'],
                        mode='markers',
                        name=regime,
                        marker=dict(
                            size=15,
                            color=color,
                            symbol='circle',
                            line=dict(width=1, color='white')
                        ),
                        text=[f"{row['Factor']}" for _, row in regime_df.iterrows()],
                        hovertemplate='%{text}<br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Risk vs Downside Risk
        downside_data = []
        for factor in factors:
            returns = df[factor].dropna()
            metrics = self.calculate_metrics(returns)
            if metrics:
                downside_data.append({
                    'Factor': factor,
                    'Total_Risk': metrics['annual_volatility'],
                    'Downside_Risk': metrics['downside_volatility'],
                    'Return': metrics['annual_return']
                })
        
        if downside_data:
            downside_df = pd.DataFrame(downside_data)
            
            for _, row in downside_df.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row['Downside_Risk']],
                        y=[row['Return']],
                        mode='markers+text',
                        name=row['Factor'],
                        text=[row['Factor']],
                        textposition='top center',
                        marker=dict(
                            size=row['Total_Risk'],  # Size represents total risk
                            color=self.factor_colors[row['Factor']],
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False,
                        legendgroup=row['Factor'],
                        hovertemplate=f"{row['Factor']}<br>Return: {row['Return']:.1f}%<br>Downside Risk: {row['Downside_Risk']:.1f}%<br>Total Risk: {row['Total_Risk']:.1f}%"
                    ),
                    row=1, col=3
                )
        
        # 4. Pre-2008 vs Post-2008
        crisis_date = pd.Timestamp('2008-09-15')  # Lehman Brothers collapse
        
        for period, marker_symbol in [('Pre-2008', 'circle'), ('Post-2008', 'square')]:
            if period == 'Pre-2008':
                period_mask = df.index < crisis_date
            else:
                period_mask = df.index >= crisis_date
            
            period_data = []
            for factor in factors:
                returns = df.loc[period_mask, factor].dropna()
                if len(returns) > 12:
                    metrics = self.calculate_metrics(returns)
                    if metrics:
                        period_data.append({
                            'Factor': factor,
                            'Return': metrics['annual_return'],
                            'Risk': metrics['annual_volatility'],
                            'Period': period
                        })
            
            if period_data:
                period_df = pd.DataFrame(period_data)
                
                for _, row in period_df.iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row['Risk']],
                            y=[row['Return']],
                            mode='markers',
                            name=f"{row['Factor']} ({period})",
                            marker=dict(
                                size=15,
                                color=self.factor_colors[row['Factor']],
                                symbol=marker_symbol,
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False,
                            legendgroup=row['Factor'],
                            hovertemplate=f"{row['Factor']} ({period})<br>Return: {row['Return']:.1f}%<br>Risk: {row['Risk']:.1f}%"
                        ),
                        row=2, col=1
                    )
        
        # 5. Bull vs Bear Markets
        # Define bull/bear based on SP500 performance
        sp500_returns = df['SP500'].dropna()
        if len(sp500_returns) > 6:
            sp500_6m = sp500_returns.rolling(6).mean()
            bull_market = sp500_6m > 0
            
            # Align bull_market index with df index
            bull_market = bull_market.reindex(df.index, fill_value=False)
            
            for market, marker_symbol in [('Bull', 'triangle-up'), ('Bear', 'triangle-down')]:
                if market == 'Bull':
                    market_mask = bull_market.fillna(False)
                else:
                    market_mask = ~bull_market.fillna(True)
                
                market_data = []
                for factor in factors:
                    # Get returns for this market condition
                    factor_returns = df[factor][market_mask].dropna()
                    if len(factor_returns) > 12:
                        metrics = self.calculate_metrics(factor_returns)
                        if metrics:
                            market_data.append({
                                'Factor': factor,
                                'Return': metrics['annual_return'],
                                'Risk': metrics['annual_volatility'],
                                'Market': market
                            })
                
                if market_data:
                    market_df = pd.DataFrame(market_data)
                    
                    for _, row in market_df.iterrows():
                        fig.add_trace(
                            go.Scatter(
                                x=[row['Risk']],
                                y=[row['Return']],
                                mode='markers',
                                name=f"{row['Factor']} ({market})",
                                marker=dict(
                                    size=15,
                                    color=self.factor_colors[row['Factor']],
                                    symbol=marker_symbol,
                                    line=dict(width=1, color='white')
                                ),
                                showlegend=False,
                                legendgroup=row['Factor'],
                                hovertemplate=f"{row['Factor']} ({market})<br>Return: {row['Return']:.1f}%<br>Risk: {row['Risk']:.1f}%"
                            ),
                            row=2, col=2
                        )
        
        # 6. Rolling 5-Year Windows
        window = 60  # 5 years in months
        
        # Calculate rolling metrics for each factor
        for factor in factors:
            rolling_returns = []
            rolling_risks = []
            rolling_dates = []
            
            for i in range(window, len(df)):
                window_returns = df[factor].iloc[i-window:i].dropna()
                if len(window_returns) >= window * 0.8:  # At least 80% data available
                    metrics = self.calculate_metrics(window_returns)
                    if metrics:
                        rolling_returns.append(metrics['annual_return'])
                        rolling_risks.append(metrics['annual_volatility'])
                        rolling_dates.append(df.index[i])
            
            if rolling_returns:
                # Create gradient color based on time
                colors = np.linspace(0, 1, len(rolling_returns))
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_risks,
                        y=rolling_returns,
                        mode='markers',
                        name=factor,
                        marker=dict(
                            size=8,
                            color=colors,
                            colorscale=[[0, self.factor_colors[factor]], [1, 'white']],
                            showscale=False,
                            line=dict(width=1, color='darkgray')
                        ),
                        showlegend=False,
                        legendgroup=factor,
                        text=[d.strftime('%Y-%m') for d in rolling_dates],
                        hovertemplate=f"{factor}<br>Date: %{{text}}<br>Return: %{{y:.1f}}%<br>Risk: %{{x:.1f}}%"
                    ),
                    row=2, col=3
                )
        
        # Update layout
        fig.update_layout(
            title=f"Risk-Return Scatter Plots - Multiple Perspectives (RF={self.rf_rate*100:.1f}%)",
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        for row in [1, 2]:
            for col in [1, 2, 3]:
                fig.update_xaxes(title_text="Annual Volatility (%)", row=row, col=col)
                fig.update_yaxes(title_text="Annual Return (%)", row=row, col=col)
                
                # Add zero lines
                fig.add_hline(y=self.rf_rate*100, line_dash="dash", line_color="gray", 
                            line_width=1, row=row, col=col)
        
        # Special axis labels
        fig.update_xaxes(title_text="Downside Volatility (%)", row=1, col=3)
        
        # Save the plot
        fig.write_html(self.output_dir / "risk_return_scatter_plots.html")
        
        return fig
    
    def _add_efficient_frontier(self, fig, row, col):
        """Add efficient frontier reference line"""
        # Add Sharpe ratio reference lines
        risk_range = np.linspace(0, 25, 100)
        
        for sharpe in [0.5, 1.0]:
            returns = self.rf_rate * 100 + sharpe * risk_range
            fig.add_trace(
                go.Scatter(
                    x=risk_range,
                    y=returns,
                    mode='lines',
                    line=dict(dash='dot', color='gray', width=1),
                    showlegend=False,
                    hovertemplate=f'Sharpe={sharpe}<br>Risk: %{{x:.1f}}%<br>Return: %{{y:.1f}}%'
                ),
                row=row, col=col
            )
        
        # Add annotations
        fig.add_annotation(
            x=20, y=self.rf_rate*100 + 0.5*20,
            text="Sharpe=0.5",
            showarrow=False,
            font=dict(size=10, color='gray'),
            row=row, col=col
        )
        
        fig.add_annotation(
            x=15, y=self.rf_rate*100 + 1.0*15,
            text="Sharpe=1.0",
            showarrow=False,
            font=dict(size=10, color='gray'),
            row=row, col=col
        )
    
    def run(self):
        """Run the analysis"""
        print("Creating risk-return scatter plots...")
        
        # Load data
        df = self.load_data()
        
        # Create scatter plots
        self.create_risk_return_scatter_plots(df)
        
        print(f"Risk-return scatter plots saved to: {self.output_dir}/risk_return_scatter_plots.html")

if __name__ == "__main__":
    creator = RiskReturnScatterPlotsCreator()
    creator.run()