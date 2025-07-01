#!/usr/bin/env python3
"""
Comprehensive Returns Visualization: Static Optimized OOS vs S&P 500
Creates three visualizations:
1. Static Optimized OOS strategy returns
2. S&P 500 benchmark returns  
3. Comparison of both strategies

Author: Claude Code
Date: July 1, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Tuple, Dict, Any

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ReturnsVisualizer:
    """Comprehensive returns visualization for factor strategies"""
    
    def __init__(self, data_path: str, results_path: str):
        """Initialize with data and results paths"""
        self.data_path = data_path
        self.results_path = results_path
        self.figures_path = os.path.join(os.path.dirname(results_path), 'figures')
        os.makedirs(self.figures_path, exist_ok=True)
        
    def load_strategy_data(self) -> Dict[str, Any]:
        """Load strategy results from JSON"""
        results_file = os.path.join(self.results_path, 'msci_validation_results_20250630_133908.json')
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def load_market_data(self) -> pd.DataFrame:
        """Load S&P 500 market data"""
        market_file = os.path.join(self.data_path, 'market_data.csv')
        df = pd.read_csv(market_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    
    def process_strategy_returns(self, strategy_data: Dict[str, Any]) -> pd.DataFrame:
        """Process strategy returns into DataFrame"""
        static_optimized = strategy_data['static_optimized_static']
        
        # Extract returns
        returns_dict = static_optimized['portfolio_returns']
        dates = [pd.to_datetime(date) for date in returns_dict.keys()]
        returns = list(returns_dict.values())
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Monthly_Return': returns
        }).set_index('Date')
        
        # Calculate cumulative returns (starting from 1.0)
        df['Cumulative_Return'] = (1 + df['Monthly_Return']).cumprod()
        
        return df
    
    def process_sp500_returns(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Process S&P 500 returns"""
        # Use SP500_Monthly_Return column
        sp500_df = market_df[['SP500_Monthly_Return']].copy()
        sp500_df.columns = ['Monthly_Return']
        
        # Calculate cumulative returns
        sp500_df['Cumulative_Return'] = (1 + sp500_df['Monthly_Return']).cumprod()
        
        return sp500_df
    
    def calculate_performance_metrics(self, returns_series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        annual_return = (1 + returns_series.mean()) ** 12 - 1
        annual_volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
    def create_strategy_visualization(self, strategy_df: pd.DataFrame, title: str = "Static Optimized OOS Strategy") -> plt.Figure:
        """Create visualization for strategy returns"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Cumulative Returns
        ax1.plot(strategy_df.index, strategy_df['Cumulative_Return'], 
                linewidth=2, color='#2E86AB', label='Static Optimized OOS')
        ax1.set_title(f'{title}: Cumulative Returns (26.5 Years)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return ($1 → $X)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add performance text
        final_value = strategy_df['Cumulative_Return'].iloc[-1]
        ax1.text(0.02, 0.95, f'Final Value: ${final_value:.2f}', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 2. Monthly Returns
        colors = ['green' if x > 0 else 'red' for x in strategy_df['Monthly_Return']]
        ax2.bar(strategy_df.index, strategy_df['Monthly_Return'], 
               color=colors, alpha=0.7, width=20)
        ax2.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Monthly Return (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling 12-Month Returns
        rolling_12m = strategy_df['Monthly_Return'].rolling(12).apply(lambda x: (1+x).prod() - 1)
        ax3.plot(strategy_df.index, rolling_12m * 100, 
                linewidth=1.5, color='#A23B72', alpha=0.8)
        ax3.set_title('Rolling 12-Month Returns', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Annual Return (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_sp500_visualization(self, sp500_df: pd.DataFrame, title: str = "S&P 500 Benchmark") -> plt.Figure:
        """Create visualization for S&P 500 returns"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Cumulative Returns
        ax1.plot(sp500_df.index, sp500_df['Cumulative_Return'], 
                linewidth=2, color='#F18F01', label='S&P 500')
        ax1.set_title(f'{title}: Cumulative Returns (26.5 Years)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return ($1 → $X)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add performance text
        final_value = sp500_df['Cumulative_Return'].iloc[-1]
        ax1.text(0.02, 0.95, f'Final Value: ${final_value:.2f}', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        # 2. Monthly Returns
        colors = ['green' if x > 0 else 'red' for x in sp500_df['Monthly_Return']]
        ax2.bar(sp500_df.index, sp500_df['Monthly_Return'], 
               color=colors, alpha=0.7, width=20)
        ax2.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Monthly Return (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling 12-Month Returns
        rolling_12m = sp500_df['Monthly_Return'].rolling(12).apply(lambda x: (1+x).prod() - 1)
        ax3.plot(sp500_df.index, rolling_12m * 100, 
                linewidth=1.5, color='#C73E1D', alpha=0.8)
        ax3.set_title('Rolling 12-Month Returns', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Annual Return (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_visualization(self, strategy_df: pd.DataFrame, sp500_df: pd.DataFrame) -> plt.Figure:
        """Create comprehensive comparison visualization"""
        fig = plt.figure(figsize=(16, 14))
        
        # Create a 3x2 subplot layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Main comparison: Cumulative Returns (spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(strategy_df.index, strategy_df['Cumulative_Return'], 
                linewidth=2.5, color='#2E86AB', label='Static Optimized OOS', alpha=0.9)
        ax1.plot(sp500_df.index, sp500_df['Cumulative_Return'], 
                linewidth=2.5, color='#F18F01', label='S&P 500', alpha=0.9)
        
        ax1.set_title('Strategy Performance Comparison: 26.5-Year Analysis (Dec 1998 - May 2025)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Cumulative Return ($1 → $X)', fontsize=13)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add performance comparison text
        strategy_final = strategy_df['Cumulative_Return'].iloc[-1]
        sp500_final = sp500_df['Cumulative_Return'].iloc[-1]
        outperformance = ((strategy_final / sp500_final) - 1) * 100
        
        ax1.text(0.02, 0.85, f'Static Optimized: ${strategy_final:.2f}\nS&P 500: ${sp500_final:.2f}\nOutperformance: {outperformance:+.1f}%', 
                transform=ax1.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # 2. Rolling 12-Month Returns Comparison
        ax2 = fig.add_subplot(gs[1, :])
        
        strategy_rolling = strategy_df['Monthly_Return'].rolling(12).apply(lambda x: (1+x).prod() - 1) * 100
        sp500_rolling = sp500_df['Monthly_Return'].rolling(12).apply(lambda x: (1+x).prod() - 1) * 100
        
        ax2.plot(strategy_df.index, strategy_rolling, linewidth=1.5, color='#2E86AB', 
                label='Static Optimized OOS', alpha=0.8)
        ax2.plot(sp500_df.index, sp500_rolling, linewidth=1.5, color='#F18F01', 
                label='S&P 500', alpha=0.8)
        
        ax2.set_title('Rolling 12-Month Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Annual Return (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Excess Returns (Strategy - S&P 500)
        ax3 = fig.add_subplot(gs[2, :])
        
        # Align the dataframes by date
        aligned_data = pd.concat([strategy_df['Monthly_Return'], sp500_df['Monthly_Return']], 
                                axis=1, join='inner')
        aligned_data.columns = ['Strategy', 'SP500']
        excess_returns = aligned_data['Strategy'] - aligned_data['SP500']
        
        colors = ['green' if x > 0 else 'red' for x in excess_returns]
        ax3.bar(excess_returns.index, excess_returns * 100, 
               color=colors, alpha=0.6, width=20, 
               label=f'Avg Monthly Excess: {excess_returns.mean()*100:.2f}%')
        
        ax3.set_title('Monthly Excess Returns (Strategy - S&P 500)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Excess Return (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        return fig
    
    def generate_performance_report(self, strategy_df: pd.DataFrame, sp500_df: pd.DataFrame) -> str:
        """Generate comprehensive performance report"""
        
        # Calculate metrics for both strategies
        strategy_metrics = self.calculate_performance_metrics(strategy_df['Monthly_Return'])
        sp500_metrics = self.calculate_performance_metrics(sp500_df['Monthly_Return'])
        
        # Align data for comparison
        aligned_data = pd.concat([strategy_df['Monthly_Return'], sp500_df['Monthly_Return']], 
                                axis=1, join='inner')
        aligned_data.columns = ['Strategy', 'SP500']
        
        # Calculate additional metrics
        correlation = aligned_data['Strategy'].corr(aligned_data['SP500'])
        excess_returns = aligned_data['Strategy'] - aligned_data['SP500']
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else 0
        
        report = f"""
COMPREHENSIVE PERFORMANCE ANALYSIS: STATIC OPTIMIZED OOS vs S&P 500
========================================================================
Analysis Period: {strategy_df.index[0].strftime('%B %Y')} - {strategy_df.index[-1].strftime('%B %Y')}
Total Months: {len(strategy_df)}

PERFORMANCE METRICS COMPARISON:
------------------------------
                        Static Optimized    S&P 500       Difference
Annual Return           {strategy_metrics['Annual Return']:8.2%}        {sp500_metrics['Annual Return']:8.2%}       {strategy_metrics['Annual Return'] - sp500_metrics['Annual Return']:+8.2%}
Annual Volatility       {strategy_metrics['Annual Volatility']:8.2%}        {sp500_metrics['Annual Volatility']:8.2%}       {strategy_metrics['Annual Volatility'] - sp500_metrics['Annual Volatility']:+8.2%}
Sharpe Ratio            {strategy_metrics['Sharpe Ratio']:8.3f}        {sp500_metrics['Sharpe Ratio']:8.3f}       {strategy_metrics['Sharpe Ratio'] - sp500_metrics['Sharpe Ratio']:+8.3f}
Max Drawdown            {strategy_metrics['Max Drawdown']:8.2%}        {sp500_metrics['Max Drawdown']:8.2%}       {strategy_metrics['Max Drawdown'] - sp500_metrics['Max Drawdown']:+8.2%}

ADDITIONAL METRICS:
------------------
Correlation with S&P 500:     {correlation:.3f}
Information Ratio:            {information_ratio:.3f}
Average Monthly Excess:       {excess_returns.mean():+.4f} ({excess_returns.mean()*100:+.2f}%)
Monthly Excess Volatility:    {excess_returns.std():.4f} ({excess_returns.std()*100:.2f}%)

CUMULATIVE PERFORMANCE:
----------------------
Static Optimized Final Value: ${strategy_df['Cumulative_Return'].iloc[-1]:.2f}
S&P 500 Final Value:          ${sp500_df['Cumulative_Return'].iloc[-1]:.2f}
Total Outperformance:         {((strategy_df['Cumulative_Return'].iloc[-1] / sp500_df['Cumulative_Return'].iloc[-1]) - 1) * 100:+.1f}%

WIN RATE ANALYSIS:
-----------------
Months Strategy Outperformed: {(excess_returns > 0).sum()} ({(excess_returns > 0).mean()*100:.1f}%)
Months Strategy Underperformed: {(excess_returns < 0).sum()} ({(excess_returns < 0).mean()*100:.1f}%)

RISK-ADJUSTED PERFORMANCE:
-------------------------
Strategy Risk-Adjusted Return: {strategy_metrics['Annual Return'] / strategy_metrics['Annual Volatility']:.3f}
S&P 500 Risk-Adjusted Return:  {sp500_metrics['Annual Return'] / sp500_metrics['Annual Volatility']:.3f}
Improvement Factor:            {(strategy_metrics['Annual Return'] / strategy_metrics['Annual Volatility']) / (sp500_metrics['Annual Return'] / sp500_metrics['Annual Volatility']):.2f}x

========================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report
    
    def run_comprehensive_analysis(self):
        """Run complete analysis and generate all visualizations"""
        print("Starting comprehensive returns visualization analysis...")
        
        # Load data
        print("Loading strategy and market data...")
        strategy_data = self.load_strategy_data()
        market_df = self.load_market_data()
        
        # Process returns
        print("Processing strategy returns...")
        strategy_df = self.process_strategy_returns(strategy_data)
        
        print("Processing S&P 500 returns...")
        sp500_df = self.process_sp500_returns(market_df)
        
        # Generate visualizations
        print("Creating static optimized strategy visualization...")
        fig1 = self.create_strategy_visualization(strategy_df)
        fig1.savefig(os.path.join(self.figures_path, 'static_optimized_oos_returns.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(self.figures_path, 'static_optimized_oos_returns.png')}")
        
        print("Creating S&P 500 benchmark visualization...")
        fig2 = self.create_sp500_visualization(sp500_df)
        fig2.savefig(os.path.join(self.figures_path, 'sp500_benchmark_returns.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(self.figures_path, 'sp500_benchmark_returns.png')}")
        
        print("Creating comprehensive comparison visualization...")
        fig3 = self.create_comparison_visualization(strategy_df, sp500_df)
        fig3.savefig(os.path.join(self.figures_path, 'comprehensive_strategy_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(self.figures_path, 'comprehensive_strategy_comparison.png')}")
        
        # Generate performance report
        print("Generating performance report...")
        report = self.generate_performance_report(strategy_df, sp500_df)
        
        report_file = os.path.join(self.figures_path, 'performance_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Performance report saved: {report_file}")
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY:")
        print("="*80)
        print(report)
        
        return strategy_df, sp500_df, report

if __name__ == "__main__":
    # Initialize paths
    base_path = "/home/dhebrank/HS/research/stock_research/factor_project_5"
    data_path = os.path.join(base_path, "data", "processed")
    results_path = os.path.join(base_path, "results", "long_term_performance")
    
    # Create visualizer and run analysis
    visualizer = ReturnsVisualizer(data_path, results_path)
    strategy_df, sp500_df, report = visualizer.run_comprehensive_analysis()
    
    print("\n" + "🎯 ANALYSIS COMPLETE! 🎯".center(80))
    print("All visualizations and reports have been generated successfully.")