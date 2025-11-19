"""
Portfolio Optimization Script

This script runs Monte Carlo simulations for all possible allocation combinations
(0.05 increments) and saves results to CSV for analysis.
"""

import numpy as np
import pandas as pd
from monte_carlo_portfolio import PortfolioSimulator
from tqdm import tqdm
import itertools


def generate_allocation_combinations():
    """
    Generate all possible allocation combinations (0.10 intervals)
    across six assets where weights sum to 1.0.
    Returns list of 6-tuples.
    """
    combos = []
    for a in range(0, 11):
        remaining_a = 10 - a
        for b in range(0, remaining_a + 1):
            remaining_b = remaining_a - b
            for c in range(0, remaining_b + 1):
                remaining_c = remaining_b - c
                for d in range(0, remaining_c + 1):
                    remaining_d = remaining_c - d
                    for e in range(0, remaining_d + 1):
                        f = remaining_d - e
                        w = (a/10.0, b/10.0, c/10.0, d/10.0, e/10.0, f/10.0)
                        combos.append(w)
    return combos


def run_simulation_for_allocation(weights,
                                  asset_names, expected_returns, 
                                  volatilities, correlation_matrix,
                                  initial_value, annual_withdrawal,
                                  years, iterations):
    """
    Run a single Monte Carlo simulation for given allocation weights.
    
    Returns:
    --------
    Dict with results
    """
    allocation_weights = list(weights)
    
    # Initialize simulator
    simulator = PortfolioSimulator(
        asset_names=asset_names,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        initial_value=initial_value,
        allocation_weights=allocation_weights,
        annual_withdrawal=annual_withdrawal,
        dividend_tax_rate=0.20,
        capital_gains_tax_rate=0.20,
        years=years,
        iterations=iterations
    )
    
    # Run simulation
    results_df = simulator.run_monte_carlo()
    
    # Calculate statistics
    final_values = results_df.iloc[:, -1]
    
    # Calculate required metrics
    mean_terminal = final_values.mean()
    median_terminal = final_values.median()
    percentile_5 = np.percentile(final_values, 5)
    # Success threshold set to $1,500,000
    target_final_value = 1_500_000
    prob_3x = (final_values > target_final_value).mean()
    
    # Calculate Sharpe Ratio
    avg_return = simulator._calculate_average_annual_return()
    sharpe_ratio = simulator._calculate_sharpe_ratio()
    
    return {
        'Emerging_Markets_Weight': allocation_weights[0],
        'Japan_Weight': allocation_weights[1],
        'Intl_Developed_Weight': allocation_weights[2],
        'Bonds_Weight': allocation_weights[3],
        'Gold_Weight': allocation_weights[4],
        'US_Equities_Weight': allocation_weights[5],
        'Mean_Terminal_Wealth': mean_terminal,
        'Median_Terminal_Wealth': median_terminal,
        '5th_Percentile_Outcome': percentile_5,
        'Sharpe_Ratio': sharpe_ratio,
        'Success_3x_Pct': prob_3x * 100  # Convert to percentage
    }


def main():
    """
    Main optimization function
    """
    print("="*70)
    print("PORTFOLIO ALLOCATION OPTIMIZER")
    print("="*70)
    
    # Define asset parameters (6 assets)
    # Order: Emerging Markets, Japan, International Developed, Bonds, Gold, US Equities
    asset_names = ['Emerging_Markets', 'Japan', 'Intl_Developed', 'Bonds', 'Gold', 'US_Equities']
    # Expected returns (as decimals)
    expected_returns = [0.1181, 0.0956, 0.0982, 0.0393, 0.0816, 0.1060]
    # Volatilities (as decimals)
    volatilities = [0.21, 0.18, 0.15, 0.089, 0.173, 0.157]
    
    # Correlation matrix (sensible defaults)
    # Equities moderately/highly correlated with each other, low-to-negative with Bonds/Gold
    correlation_matrix = np.array([
        [1.00, 0.70, 0.75, -0.10, -0.05, 0.65],  # Emerging Markets
        [0.70, 1.00, 0.80, -0.10, -0.05, 0.60],  # Japan
        [0.75, 0.80, 1.00, -0.10, -0.05, 0.75],  # International Developed
        [-0.10, -0.10, -0.10, 1.00, -0.30, -0.10],  # Bonds
        [-0.05, -0.05, -0.05, -0.30, 1.00, -0.10],  # Gold
        [0.65, 0.60, 0.75, -0.10, -0.10, 1.00]   # US Equities
    ])
    
    # Portfolio parameters
    initial_value = 500_000
    annual_withdrawal = 12_000
    years = 10
    iterations = 500  # Increased iterations for higher precision
    
    # Generate all allocation combinations at 5% steps (including 0% and 100%), 6 assets
    # Constraints: EM >= 40%; Japan/Intl/Bonds/Gold <= 25%; US free; weights sum to 100%
    print("\nGenerating allocation combinations (6 assets, 5% steps, constrained)...")
    combinations = []
    step_units = 20  # 20 units of 5% to make 100%
    # Order: EM(a), JP(b), INTL(c), BONDS(d), GOLD(e), US(f)
    for a in range(8, step_units + 1):  # EM >= 40%
        remaining_a = step_units - a
        for b in range(0, min(5, remaining_a) + 1):  # Japan <= 25%
            remaining_b = remaining_a - b
            for c in range(0, min(5, remaining_b) + 1):  # Intl <= 25%
                remaining_c = remaining_b - c
                for d in range(0, min(5, remaining_c) + 1):  # Bonds <= 25%
                    remaining_d = remaining_c - d
                    for e in range(0, min(5, remaining_d) + 1):  # Gold <= 25%
                        f = remaining_d - e  # US free
                        if f < 0:
                            continue
                        weights = [a, b, c, d, e, f]
                        combinations.append(tuple(w / 20.0 for w in weights))
    print(f"Total combinations to test: {len(combinations)}")
    
    # Store results
    results = []
    
    # Run simulations for each combination
    print("\nRunning simulations...")
    for combo in tqdm(combinations, desc="Progress"):
        try:
            result = run_simulation_for_allocation(
                combo,
                asset_names, expected_returns, volatilities,
                correlation_matrix, initial_value, annual_withdrawal,
                years, iterations
            )
            results.append(result)
        except Exception as e:
            print(f"\nError with allocation {combo}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'portfolio_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    print(f"Total combinations tested: {len(results)}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nMax Success Rate (3x): {results_df['Success_3x_Pct'].max():.2f}%")
    print(f"Max Mean Terminal Wealth: ${results_df['Mean_Terminal_Wealth'].max():,.0f}")
    print(f"Max Median Terminal Wealth: ${results_df['Median_Terminal_Wealth'].max():,.0f}")
    
    return results_df


if __name__ == "__main__":
    results_df = main()
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Analyze the CSV file for optimal strategies")
    print("2. Use analyze_results.py to identify best allocation for 3x returns")

