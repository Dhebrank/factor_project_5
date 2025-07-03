# Sharpe Ratio Calculation Audit Report

## Executive Summary

A comprehensive audit of the persistence-required analysis revealed **critical calculation errors** in Sharpe ratios that significantly overstated risk-adjusted returns. This report documents the errors found and presents corrected calculations.

## Key Findings

### 1. Sharpe Ratio Formula Error
- **Original**: Sharpe = Annual Return / Annual Volatility (assumes 0% risk-free rate)
- **Corrected**: Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
- **Risk-Free Rate Used**: 2.35% (average 2-year Treasury yield over the period)

### 2. SP500 Data Format Issue
- **Problem**: SP500 data was in price format (ranging from 73.93 to 602.55)
- **Impact**: Showed impossible 256,852% annual return
- **Solution**: Converted prices to returns using pct_change()

## Corrected Overall Sharpe Ratios

| Factor | Original (No RF) | Corrected (2.35% RF) | Difference |
|--------|-----------------|---------------------|------------|
| MinVol | 0.731 | 0.553 | -24.3% |
| Momentum | 0.696 | 0.576 | -17.2% |
| Quality | 0.642 | 0.503 | -21.7% |
| Value | 0.555 | 0.436 | -21.4% |

## Regime-Specific Corrections

### Goldilocks Regime
| Factor | Return | Original Sharpe | Corrected Sharpe |
|--------|--------|----------------|------------------|
| MinVol | 12.9% | 1.379 | 1.104 |
| Momentum | 12.0% | 0.970 | 0.763 |
| Quality | 11.2% | 0.886 | 0.686 |
| Value | 10.0% | 0.657 | 0.492 |

### Stagflation Regime (Most Dramatic Change)
| Factor | Return | Original Sharpe | Corrected Sharpe |
|--------|--------|----------------|------------------|
| Value | 2.2% | 0.108 | **-0.006** |
| Quality | 1.6% | 0.088 | **-0.044** |
| MinVol | 0.5% | 0.037 | **-0.121** |
| Momentum | 1.0% | 0.052 | **-0.074** |

**Critical Insight**: All factors show NEGATIVE Sharpe ratios in Stagflation when properly accounting for risk-free rate!

### Recession Regime (Best Performance)
| Factor | Return | Original Sharpe | Corrected Sharpe |
|--------|--------|----------------|------------------|
| Momentum | 27.4% | 1.651 | 1.478 |
| Quality | 19.6% | 1.250 | 1.077 |
| MinVol | 12.8% | 0.988 | 0.791 |
| Value | 16.0% | 0.817 | 0.683 |

## Impact on Investment Decisions

### Before Correction
1. All factors appeared attractive in all regimes
2. Stagflation seemed manageable with positive Sharpe ratios
3. Factor ranking: MinVol > Momentum > Quality > Value

### After Correction
1. Stagflation reveals as a "no-win" regime - all factors destroy value after risk adjustment
2. Momentum in Recession shows truly exceptional performance (1.478 Sharpe)
3. Factor ranking unchanged but magnitudes significantly lower
4. Risk-free rate hurdle is material - many strategies barely beat Treasury bills

## Methodology Notes

### Correct Calculation Process
```python
# Monthly statistics
monthly_mean = returns.mean()
monthly_std = returns.std()

# Annualize
annual_return = (1 + monthly_mean) ** 12 - 1
annual_vol = monthly_std * np.sqrt(12)

# Risk-free adjustment
monthly_rf = (1 + 0.0235) ** (1/12) - 1
excess_return = monthly_mean - monthly_rf
annual_excess = (1 + excess_return) ** 12 - 1

# Sharpe ratio
sharpe_ratio = annual_excess / annual_vol
```

### Data Quality Checks
- Factor data confirmed in return format (monthly returns -18% to +16%)
- SP500 required conversion from price to return format
- Risk-free rate from 2-year Treasury (DGS2) with NaN handling

## Recommendations

1. **Always include risk-free rate** in Sharpe calculations - assuming 0% significantly overstates attractiveness
2. **Verify data format** before calculations - price vs return format can cause massive errors
3. **Re-evaluate strategies** - some factors barely compensate for risk-free alternative
4. **Stagflation preparation** - consider cash/Treasury positions when this regime is detected
5. **Momentum in crisis** - exceptional Recession performance suggests crisis alpha potential

## Files Updated

- Created `persistence_required_analysis_corrected.py` with proper calculations
- New results in `results/persistence_required_analysis_corrected/`
- Original results preserved in `results/persistence_required_analysis/`
- Audit trail in `results/persistence_required_analysis/audit_results.json`

## Conclusion

The original analysis significantly overstated the attractiveness of factor investing by omitting the risk-free rate from Sharpe ratio calculations. The corrected analysis reveals a more nuanced picture where:
- Factor investing still adds value but with more modest risk-adjusted returns
- Regime awareness is critical - Stagflation is particularly challenging
- Proper benchmarking against risk-free alternatives is essential

This audit emphasizes the importance of rigorous calculation verification in quantitative analysis.