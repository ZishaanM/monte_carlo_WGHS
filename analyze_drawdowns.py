"""
Analyze drawdown statistics from portfolio optimization results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_drawdowns(csv_file='portfolio_optimization_results.csv'):
    """
    Analyze drawdown statistics from portfolio optimization results.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with optimization results
    """
    
    # Load results
    df = pd.read_csv(csv_file)
    
    print("="*70)
    print("DRAWDOWN ANALYSIS")
    print("="*70)
    
    # Basic statistics
    print(f"\nDrawdown Statistics Across All Allocations:")
    print(f"  Mean Max Drawdown: {df['Mean_Max_Drawdown'].mean():.2%}")
    print(f"  Median Max Drawdown: {df['Median_Max_Drawdown'].median():.2%}")
    print(f"  Best (Least Negative) Mean Drawdown: {df['Mean_Max_Drawdown'].max():.2%}")
    print(f"  Worst (Most Negative) Mean Drawdown: {df['Mean_Max_Drawdown'].min():.2%}")
    
    # Find allocations with best/worst drawdowns
    best_dd_idx = df['Mean_Max_Drawdown'].idxmax()
    worst_dd_idx = df['Mean_Max_Drawdown'].idxmin()
    
    print(f"\nAllocation with Best Drawdown Profile:")
    best_row = df.iloc[best_dd_idx]
    print(f"  Emerging Markets: {best_row['Emerging_Markets_Weight']:.1%}")
    print(f"  Japan: {best_row['Japan_Weight']:.1%}")
    print(f"  Intl Developed: {best_row['Intl_Developed_Weight']:.1%}")
    print(f"  Bonds: {best_row['Bonds_Weight']:.1%}")
    print(f"  Gold: {best_row['Gold_Weight']:.1%}")
    print(f"  US Equities: {best_row['US_Equities_Weight']:.1%}")
    print(f"  Mean Max Drawdown: {best_row['Mean_Max_Drawdown']:.2%}")
    print(f"  Mean Terminal Wealth: ${best_row['Mean_Terminal_Wealth']:,.0f}")
    
    print(f"\nAllocation with Worst Drawdown Profile:")
    worst_row = df.iloc[worst_dd_idx]
    print(f"  Emerging Markets: {worst_row['Emerging_Markets_Weight']:.1%}")
    print(f"  Japan: {worst_row['Japan_Weight']:.1%}")
    print(f"  Intl Developed: {worst_row['Intl_Developed_Weight']:.1%}")
    print(f"  Bonds: {worst_row['Bonds_Weight']:.1%}")
    print(f"  Gold: {worst_row['Gold_Weight']:.1%}")
    print(f"  US Equities: {worst_row['US_Equities_Weight']:.1%}")
    print(f"  Mean Max Drawdown: {worst_row['Mean_Max_Drawdown']:.2%}")
    print(f"  Mean Terminal Wealth: ${worst_row['Mean_Terminal_Wealth']:,.0f}")
    
    # Correlation between drawdown and returns
    correlation = df['Mean_Max_Drawdown'].corr(df['Mean_Terminal_Wealth'])
    print(f"\nCorrelation between Max Drawdown and Terminal Wealth: {correlation:.3f}")
    
    # Top 10 allocations by drawdown performance
    print(f"\nTop 10 Allocations by Drawdown (Least Negative):")
    top_dd = df.nlargest(10, 'Mean_Max_Drawdown')
    for idx, row in top_dd.iterrows():
        print(f"  DD: {row['Mean_Max_Drawdown']:.2%} | "
              f"Wealth: ${row['Mean_Terminal_Wealth']:,.0f} | "
              f"EM:{row['Emerging_Markets_Weight']:.0%} "
              f"JP:{row['Japan_Weight']:.0%} "
              f"INTL:{row['Intl_Developed_Weight']:.0%} "
              f"B:{row['Bonds_Weight']:.0%} "
              f"G:{row['Gold_Weight']:.0%} "
              f"US:{row['US_Equities_Weight']:.0%}")
    
    return df


def plot_drawdown_vs_returns(df):
    """
    Plot drawdown vs terminal wealth scatter plot.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Mean_Terminal_Wealth'], df['Mean_Max_Drawdown'] * 100, 
                alpha=0.5, s=50)
    plt.xlabel('Mean Terminal Wealth ($)')
    plt.ylabel('Mean Max Drawdown (%)')
    plt.title('Drawdown vs Terminal Wealth Trade-off')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = analyze_drawdowns()
    
    # Uncomment to plot
    # plot_drawdown_vs_returns(df)

