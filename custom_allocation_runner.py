"""
Run Monte Carlo for specific allocation sets and report success rates.
Success threshold is aligned with portfolio_optimizer.py (>= $1,580,000).
"""

import numpy as np
import pandas as pd
from monte_carlo_portfolio import PortfolioSimulator


ASSET_NAMES = ['Emerging_Markets', 'Japan', 'Intl_Developed', 'Bonds', 'Gold', 'US_Equities']
EXPECTED_RETURNS = [0.1181, 0.0956, 0.0982, 0.0393, 0.0816, 0.1060]
VOLATILITIES = [0.21, 0.18, 0.15, 0.089, 0.173, 0.157]
CORRELATION_MATRIX = np.array([
    [1.00, 0.70, 0.75, -0.10, -0.05, 0.65],
    [0.70, 1.00, 0.80, -0.10, -0.05, 0.60],
    [0.75, 0.80, 1.00, -0.10, -0.05, 0.75],
    [-0.10, -0.10, -0.10, 1.00, -0.30, -0.10],
    [-0.05, -0.05, -0.05, -0.30, 1.00, -0.10],
    [0.65, 0.60, 0.75, -0.10, -0.10, 1.00]
])


def run_for_allocation(weights, iterations=1000, years=10,
                       initial_value=500_000, annual_withdrawal=12_000,
                       target=1_500_000):
    simulator = PortfolioSimulator(
        asset_names=ASSET_NAMES,
        expected_returns=EXPECTED_RETURNS,
        volatilities=VOLATILITIES,
        correlation_matrix=CORRELATION_MATRIX,
        initial_value=initial_value,
        allocation_weights=list(weights),
        annual_withdrawal=annual_withdrawal,
        dividend_tax_rate=0.20,
        capital_gains_tax_rate=0.20,
        years=years,
        iterations=iterations
    )
    results_df = simulator.run_monte_carlo()
    final_values = results_df.iloc[:, -1]
    success_rate = (final_values >= target).mean()
    return success_rate, results_df


def main():
    allocations = [
        (0.45, 0.00, 0.30, 0.00, 0.10, 0.15),
        (0.60, 0.05, 0.05, 0.00, 0.05, 0.25),
        (0.70, 0.00, 0.10, 0.00, 0.10, 0.10),
        (0.80, 0.00, 0.10, 0.00, 0.05, 0.05),
        (0.60, 0.00, 0.30, 0.00, 0.00, 0.10),
        (0.75, 0.05, 0.05, 0.00, 0.10, 0.05),
        (0.65, 0.00, 0.20, 0.00, 0.10, 0.05),
        (0.70, 0.00, 0.25, 0.00, 0.05, 0.00),
        (0.80, 0.20, 0.00, 0.00, 0.00, 0.00),
        (0.85, 0.00, 0.00, 0.00, 0.10, 0.05),
    ]

    print("=" * 70)
    print("CUSTOM ALLOCATION SUCCESS RATES (target >= $1,580,000)")
    print("=" * 70)

    rows = []
    for idx, weights in enumerate(allocations, start=1):
        print(f"\n[{idx}/10] Running {weights} ...")
        success_rate, _ = run_for_allocation(weights, iterations=1000)
        row = {
            'Emerging_Markets_Weight': weights[0],
            'Japan_Weight': weights[1],
            'Intl_Developed_Weight': weights[2],
            'Bonds_Weight': weights[3],
            'Gold_Weight': weights[4],
            'US_Equities_Weight': weights[5],
            'Success_Pct_>=1.58M': success_rate * 100,
        }
        rows.append(row)
        print(f"Success rate: {success_rate*100:.2f}%")

    out = pd.DataFrame(rows)
    out_file = 'custom_allocation_success.csv'
    out.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")
    return out


if __name__ == '__main__':
    main()


