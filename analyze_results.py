"""
Portfolio Results Analysis Script

This script analyzes the CSV results to identify the most successful 
allocation strategy for achieving 3x returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_optimal_strategies(csv_file='portfolio_optimization_results.csv'):
    """
    Analyze CSV results and identify optimal strategies.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file with optimization results
    """
    print("="*70)
    print("PORTFOLIO STRATEGY ANALYSIS")
    print("="*70)
    
    # Read CSV
    print(f"\nLoading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} allocation combinations")
    
    # Calculate additional metrics (auto-detect weight columns)
    weight_cols = [c for c in df.columns if c.endswith('_Weight')]
    # Identify equities as all weights except Bonds and Gold where present
    equity_cols = [c for c in weight_cols if not (c.startswith('Bonds') or c.startswith('Gold'))]
    df['Total_Equities'] = df[equity_cols].sum(axis=1)
    df['Total_Risk_Assets'] = df[equity_cols + ([c for c in weight_cols if c.startswith('Gold')])] .sum(axis=1)
    df['Conservative_Score'] = df.get('Bonds_Weight', pd.Series(0, index=df.index)) * 0.4 \
        + df['Median_Terminal_Wealth'] / 1_000_000 * 0.6
    
    # Strategy 1: Highest 3x Success Rate
    print("\n" + "="*70)
    print("STRATEGY 1: HIGHEST 3X SUCCESS RATE")
    print("="*70)
    best_3x = df.loc[df['Success_3x_Pct'].idxmax()]
    print(f"Success Rate (3x): {best_3x['Success_3x_Pct']:.2f}%")
    print(f"\nAllocation:")
    for c in weight_cols:
        label = c.replace('_', ' ').replace('Weight', '').strip()
        print(f"  {label}: {best_3x[c]:.1%}")
    print(f"\nResults:")
    print(f"  Mean Terminal Wealth: ${best_3x['Mean_Terminal_Wealth']:,.0f}")
    print(f"  Median Terminal Wealth: ${best_3x['Median_Terminal_Wealth']:,.0f}")
    print(f"  Bottom 5% Outcome: ${best_3x['5th_Percentile_Outcome']:,.0f}")
    print(f"  Sharpe Ratio: {best_3x['Sharpe_Ratio']:.2f}")
    
    # Strategy 2: Top 10 by 3x success rate
    print("\n" + "="*70)
    print("TOP 10 STRATEGIES BY 3X SUCCESS RATE")
    print("="*70)
    cols_for_table = weight_cols + ['Success_3x_Pct', 'Mean_Terminal_Wealth', 'Median_Terminal_Wealth', '5th_Percentile_Outcome', 'Sharpe_Ratio']
    top_10 = df.nlargest(10, 'Success_3x_Pct')[cols_for_table].round(2)
    print("\n" + top_10.to_string(index=False))
    
    # Strategy 3: Balance high returns with reasonable risk
    print("\n" + "="*70)
    print("STRATEGY 3: BALANCED APPROACH")
    print("(High 3x probability + High median wealth)")
    print("="*70)
    
    # Weighted score: 60% 3x success, 40% median wealth
    df['Balanced_Score'] = (df['Success_3x_Pct'] * 0.6) + (df['Median_Terminal_Wealth'] / 10000 * 0.4)
    best_balanced = df.loc[df['Balanced_Score'].idxmax()]
    
    print(f"Success Rate (3x): {best_balanced['Success_3x_Pct']:.2f}%")
    print(f"\nAllocation:")
    for c in weight_cols:
        label = c.replace('_', ' ').replace('Weight', '').strip()
        print(f"  {label}: {best_balanced[c]:.1%}")
    print(f"\nResults:")
    print(f"  Mean Terminal Wealth: ${best_balanced['Mean_Terminal_Wealth']:,.0f}")
    print(f"  Median Terminal Wealth: ${best_balanced['Median_Terminal_Wealth']:,.0f}")
    print(f"  Bottom 5% Outcome: ${best_balanced['5th_Percentile_Outcome']:,.0f}")
    print(f"  Sharpe Ratio: {best_balanced['Sharpe_Ratio']:.2f}")
    
    # Strategy 4: Conservative with decent 3x chance
    print("\n" + "="*70)
    print("STRATEGY 4: CONSERVATIVE BUT SOLID")
    print("(Some cash for safety, but still decent 3x chance)")
    print("="*70)
    
    # Filter for strategies with >5% cash
    conservative = df[df.get('Bonds_Weight', 0) >= 0.10].copy()  # example: at least 10% bonds
    if len(conservative) > 0:
        conservative_balanced = conservative.loc[conservative['Balanced_Score'].idxmax()]
        
        print(f"Success Rate (3x): {conservative_balanced['Success_3x_Pct']:.2f}%")
        print(f"\nAllocation:")
        for c in weight_cols:
            label = c.replace('_', ' ').replace('Weight', '').strip()
            print(f"  {label}: {conservative_balanced[c]:.1%}")
        print(f"\nResults:")
        print(f"  Mean Terminal Wealth: ${conservative_balanced['Mean_Terminal_Wealth']:,.0f}")
        print(f"  Median Terminal Wealth: ${conservative_balanced['Median_Terminal_Wealth']:,.0f}")
        print(f"  Bottom 5% Outcome: ${conservative_balanced['5th_Percentile_Outcome']:,.0f}")
        print(f"  Sharpe Ratio: {conservative_balanced['Sharpe_Ratio']:.2f}")
    
    # Analysis by equity allocation
    print("\n" + "="*70)
    print("ANALYSIS BY TOTAL EQUITY ALLOCATION")
    print("="*70)
    
    df['Equity_Bin'] = pd.cut(df['Total_Equities'], bins=[0, 0.3, 0.6, 0.8, 1.0], 
                              labels=['Low (0-30%)', 'Medium (30-60%)', 'High (60-80%)', 'Very High (80-100%)'])
    
    equity_analysis = df.groupby('Equity_Bin').agg({
        'Success_3x_Pct': 'mean',
        'Mean_Terminal_Wealth': 'mean',
        'Median_Terminal_Wealth': 'mean',
        '5th_Percentile_Outcome': 'mean',
        'Sharpe_Ratio': 'mean'
    }).round(2)
    
    print("\nAverage metrics by equity allocation:")
    print(equity_analysis.to_string())
    
    # Save detailed analysis
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save top 50 strategies
    top_50 = df.nlargest(50, 'Success_3x_Pct')
    top_50_file = 'analysis_top_50.csv'
    top_50.to_csv(top_50_file, index=False)
    print(f"✓ Saved top 50 strategies to {top_50_file}")
    
    # Save analysis summary
    summary_dict = {
        'Strategy': ['Highest 3x Success', 'Balanced Approach'] + (['Conservative'] if len(conservative) > 0 else [])
    }
    for c in weight_cols:
        key = c
        vals = [best_3x[c], best_balanced[c]] + ([conservative_balanced[c]] if len(conservative) > 0 else [])
        summary_dict[key] = vals
    summary_dict.update({
        '3x_Success_Pct': [best_3x['Success_3x_Pct'], best_balanced['Success_3x_Pct']] + ([conservative_balanced['Success_3x_Pct']] if len(conservative) > 0 else []),
        'Mean_Terminal': [best_3x['Mean_Terminal_Wealth'], best_balanced['Mean_Terminal_Wealth']] + ([conservative_balanced['Mean_Terminal_Wealth']] if len(conservative) > 0 else []),
        'Median_Terminal': [best_3x['Median_Terminal_Wealth'], best_balanced['Median_Terminal_Wealth']] + ([conservative_balanced['Median_Terminal_Wealth']] if len(conservative) > 0 else []),
        'Bottom5Pct': [best_3x['5th_Percentile_Outcome'], best_balanced['5th_Percentile_Outcome']] + ([conservative_balanced['5th_Percentile_Outcome']] if len(conservative) > 0 else [])
    })
    summary = pd.DataFrame(summary_dict)
    
    summary_file = 'strategy_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"✓ Saved strategy summary to {summary_file}")
    
    return df


if __name__ == "__main__":
    df = analyze_optimal_strategies()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nFiles created:")
    print("- analysis_top_50.csv: Best 50 allocation combinations")
    print("- strategy_summary.csv: Summary of recommended strategies")

