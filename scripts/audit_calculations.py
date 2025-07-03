#!/usr/bin/env python3
"""
Audit script for persistence-required analysis calculations
Verifies Sharpe ratios and other performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class CalculationAuditor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.aligned_file = Path("results/business_cycle_analysis/_archive/aligned_master_dataset_FIXED.csv")
        self.results_file = Path("results/persistence_required_analysis/executive_summary.json")
        
    def load_data(self):
        """Load the aligned dataset and check data format"""
        print("Loading aligned dataset...")
        df = pd.read_csv(self.aligned_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Check data format
        factors = ['Value', 'Quality', 'MinVol', 'Momentum', 'SP500']
        print("\nChecking factor data format...")
        for factor in factors:
            print(f"\n{factor}:")
            print(f"  First 5 values: {df[factor].head().values}")
            print(f"  Min: {df[factor].min():.4f}, Max: {df[factor].max():.4f}")
            print(f"  Mean: {df[factor].mean():.4f}, Std: {df[factor].std():.4f}")
            
            # Check if data looks like returns or levels
            if df[factor].min() < -0.5 or df[factor].max() > 100:
                print(f"  WARNING: {factor} data might be in index/price format, not returns!")
        
        # Check risk-free rate availability
        print("\nChecking risk-free rate data...")
        rf_columns = ['FEDFUNDS', 'DGS2', 'DGS10']
        for col in rf_columns:
            if col in df.columns:
                print(f"  {col}: Available (Mean: {df[col].mean():.2f}%)")
        
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
    
    def calculate_performance_metrics(self, df, factor, regime_mask, rf_rate=None):
        """Calculate performance metrics with proper methodology"""
        factor_data = df.loc[regime_mask, factor]
        
        if len(factor_data) < 2:
            return None
        
        # Calculate returns based on data format
        # Check if data is already in return format (typical range -0.5 to 0.5)
        if factor_data.min() > -0.5 and factor_data.max() < 0.5:
            # Data appears to be in return format already
            returns = factor_data
            print(f"  {factor}: Using data as returns (range suggests return format)")
        else:
            # Data appears to be in level/index format
            returns = factor_data.pct_change().dropna()
            print(f"  {factor}: Converting from levels to returns")
        
        if len(returns) == 0:
            return None
        
        # Calculate metrics
        monthly_mean = returns.mean()
        monthly_std = returns.std()
        
        # Annualize
        annual_return = (1 + monthly_mean) ** 12 - 1
        annual_vol = monthly_std * np.sqrt(12)
        
        # Calculate Sharpe ratio
        if rf_rate is not None:
            # Convert annual rf_rate to monthly and then to decimal
            monthly_rf = (1 + rf_rate) ** (1/12) - 1
            excess_return = monthly_mean - monthly_rf
            annual_excess = (1 + excess_return) ** 12 - 1
            sharpe = annual_excess / annual_vol if annual_vol > 0 else 0
        else:
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            'annual_return': annual_return * 100,  # Convert to percentage
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'sharpe_ratio_no_rf': annual_return / annual_vol if annual_vol > 0 else 0,
            'max_drawdown': max_dd * 100,
            'observations': len(returns),
            'monthly_mean_return': monthly_mean * 100,
            'monthly_volatility': monthly_std * 100
        }
    
    def audit_calculations(self):
        """Audit all calculations"""
        # Load data
        df = self.load_data()
        
        # Apply persistence requirement
        df['Regime_Persistence'] = self.apply_persistence_requirement(df['ECONOMIC_REGIME'])
        
        # Get risk-free rate (use DGS2 or FEDFUNDS if available)
        if 'DGS2' in df.columns and not df['DGS2'].isna().all():
            # DGS2 is 2-year Treasury rate
            avg_rf_rate = df['DGS2'].dropna().mean() / 100  # Convert to decimal
            print(f"\nUsing average DGS2 (2Y Treasury) rate: {avg_rf_rate*100:.2f}%")
        elif 'FEDFUNDS' in df.columns and not df['FEDFUNDS'].isna().all():
            # FEDFUNDS is typically in percentage form (e.g., 5.0 for 5%)
            avg_rf_rate = df['FEDFUNDS'].dropna().mean() / 100  # Convert to decimal
            print(f"\nUsing average FEDFUNDS rate: {avg_rf_rate*100:.2f}%")
        else:
            avg_rf_rate = 0.02  # Default 2% if not available
            print(f"\nUsing default risk-free rate: {avg_rf_rate*100:.2f}%")
        
        # Define factors and regimes
        factors = ['Value', 'Quality', 'MinVol', 'Momentum']
        regimes = ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']
        
        # Recalculate all metrics
        print("\nRecalculating performance metrics...")
        results = {}
        
        for regime in regimes:
            print(f"\n{regime}:")
            regime_mask = df['Regime_Persistence'] == regime
            regime_count = regime_mask.sum()
            print(f"  Observations: {regime_count}")
            
            if regime_count > 0:
                results[regime] = {}
                
                for factor in factors:
                    metrics = self.calculate_performance_metrics(df, factor, regime_mask, avg_rf_rate)
                    if metrics:
                        results[regime][factor] = metrics
                        print(f"  {factor}:")
                        print(f"    Annual Return: {metrics['annual_return']:.1f}%")
                        print(f"    Sharpe (with RF): {metrics['sharpe_ratio']:.3f}")
                        print(f"    Sharpe (no RF): {metrics['sharpe_ratio_no_rf']:.3f}")
        
        # Special check for SP500
        print("\nSpecial SP500 Analysis:")
        sp500_sample = df['SP500'].head(20)
        print(f"SP500 first 20 values:\n{sp500_sample}")
        print(f"\nSP500 statistics:")
        print(f"  Min: {df['SP500'].min()}")
        print(f"  Max: {df['SP500'].max()}")
        print(f"  Mean: {df['SP500'].mean()}")
        
        # Check if SP500 might be price data
        if df['SP500'].min() > 100:
            print("  WARNING: SP500 appears to be price/index data, not returns!")
            sp500_returns = df['SP500'].pct_change().dropna()
            print(f"  SP500 return statistics:")
            print(f"    Monthly mean return: {sp500_returns.mean()*100:.2f}%")
            print(f"    Monthly volatility: {sp500_returns.std()*100:.2f}%")
            print(f"    Annualized return: {((1 + sp500_returns.mean())**12 - 1)*100:.2f}%")
        
        # Compare with original results
        print("\nComparing with original results...")
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                original = json.load(f)
            
            if 'key_findings' in original and 'sharpe_rankings' in original['key_findings']:
                print("\nOriginal best Sharpe ratios:")
                for factor, sharpe in original['key_findings']['sharpe_rankings'].items():
                    print(f"  {factor}: {sharpe:.3f}")
            else:
                print("\nOriginal Sharpe rankings not found in expected format")
        
        # Save audit results
        audit_results = {
            'audit_date': pd.Timestamp.now().isoformat(),
            'risk_free_rate_used': avg_rf_rate,
            'recalculated_metrics': results,
            'data_format_warnings': {
                'SP500': 'Appears to be in price/index format',
                'factors': 'Value, Quality, MinVol, Momentum appear to be in return format'
            }
        }
        
        output_file = self.project_root / "results" / "persistence_required_analysis" / "audit_results.json"
        with open(output_file, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        print(f"\nAudit complete! Results saved to: {output_file}")
        
        return results
    
    def create_comparison_report(self):
        """Create detailed comparison report"""
        try:
            # Load original executive summary
            with open(self.results_file, 'r') as f:
                original = json.load(f)
            
            # Load audit results
            audit_file = self.project_root / "results" / "persistence_required_analysis" / "audit_results.json"
            if not audit_file.exists():
                print("Audit results not found. Run audit_calculations() first.")
                return None
                
            with open(audit_file, 'r') as f:
                audit = json.load(f)
            
            # Create comparison based on what we have
            print("\n=== SHARPE RATIO COMPARISON ===")
            print("\nOriginal calculations (assuming 0% risk-free rate):")
            print("Factor    | Goldilocks | Overheating | Stagflation | Recession")
            print("-" * 65)
            
            # Print recalculated values
            print("\nRecalculated with proper risk-free rate:")
            for regime in ['Goldilocks', 'Overheating', 'Stagflation', 'Recession']:
                if regime in audit['recalculated_metrics']:
                    print(f"\n{regime}:")
                    for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                        if factor in audit['recalculated_metrics'][regime]:
                            metrics = audit['recalculated_metrics'][regime][factor]
                            print(f"  {factor}: Return={metrics['annual_return']:.1f}%, " +
                                  f"Sharpe(RF)={metrics['sharpe_ratio']:.3f}, " +
                                  f"Sharpe(NoRF)={metrics['sharpe_ratio_no_rf']:.3f}")
            
            # Calculate overall Sharpe ratios
            print("\n=== OVERALL SHARPE RATIOS (Full Period) ===")
            df = pd.read_csv(self.aligned_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            for factor in ['Value', 'Quality', 'MinVol', 'Momentum']:
                returns = df[factor]
                monthly_mean = returns.mean()
                monthly_std = returns.std()
                annual_return = (1 + monthly_mean) ** 12 - 1
                annual_vol = monthly_std * np.sqrt(12)
                sharpe_no_rf = annual_return / annual_vol if annual_vol > 0 else 0
                
                # With risk-free rate
                rf_rate = audit['risk_free_rate_used']
                monthly_rf = (1 + rf_rate) ** (1/12) - 1
                excess_return = monthly_mean - monthly_rf
                annual_excess = (1 + excess_return) ** 12 - 1
                sharpe_rf = annual_excess / annual_vol if annual_vol > 0 else 0
                
                print(f"{factor}: Sharpe(NoRF)={sharpe_no_rf:.3f}, Sharpe(RF={rf_rate*100:.1f}%)={sharpe_rf:.3f}")
            
            return None
            
        except Exception as e:
            print(f"Error creating comparison report: {e}")
            return None

if __name__ == "__main__":
    auditor = CalculationAuditor()
    results = auditor.audit_calculations()
    auditor.create_comparison_report()