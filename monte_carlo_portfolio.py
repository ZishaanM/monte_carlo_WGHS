"""
Monte Carlo Portfolio Simulation Framework

This script provides a comprehensive framework for simulating portfolio returns
over multiple years using Monte Carlo methods. It supports multiple asset classes,
correlated returns, annual rebalancing, withdrawals, and tax considerations.

Author: Portfolio Simulation Framework
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PortfolioSimulator:
    """
    A comprehensive Monte Carlo portfolio simulation framework.
    
    This class handles portfolio simulation with multiple asset classes,
    correlated returns, rebalancing, withdrawals, and tax calculations.
    """
    
    def __init__(self, 
                 asset_names: List[str],
                 expected_returns: List[float],
                 volatilities: List[float],
                 correlation_matrix: np.ndarray,
                 initial_value: float,
                 allocation_weights: List[float],
                 annual_withdrawal: float,
                 dividend_tax_rate: float = 0.15,
                 capital_gains_tax_rate: float = 0.20,
                 years: int = 10,
                 iterations: int = 10000):
        """
        Initialize the portfolio simulator with all required parameters.
        
        Parameters:
        -----------
        asset_names : List[str]
            Names of asset classes (e.g., ['US_Equities', 'Intl_Equities', 'Gold', 'Cash'])
        expected_returns : List[float]
            Expected annual returns for each asset (as decimals, e.g., 0.08 for 8%)
        volatilities : List[float]
            Annual volatilities (standard deviations) for each asset (as decimals)
        correlation_matrix : np.ndarray
            Correlation matrix between asset classes
        initial_value : float
            Initial portfolio value
        allocation_weights : List[float]
            Target allocation weights for each asset (should sum to 1.0)
        annual_withdrawal : float
            Annual withdrawal amount
        dividend_tax_rate : float
            Tax rate on dividend income (default 0.15)
        capital_gains_tax_rate : float
            Tax rate on capital gains (default 0.20)
        years : int
            Number of years to simulate (default 10)
        iterations : int
            Number of Monte Carlo iterations (default 10000)
        """
        
        # Validate inputs
        self._validate_inputs(asset_names, expected_returns, volatilities, 
                             correlation_matrix, allocation_weights)
        
        # Store parameters
        self.asset_names = asset_names
        self.expected_returns = np.array(expected_returns)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = correlation_matrix
        self.initial_value = initial_value
        self.allocation_weights = np.array(allocation_weights)
        self.annual_withdrawal = annual_withdrawal
        self.dividend_tax_rate = dividend_tax_rate
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.years = years
        self.iterations = iterations
        
        # Calculate covariance matrix from correlation matrix and volatilities
        self.covariance_matrix = self._calculate_covariance_matrix()
        
        # Initialize results storage
        self.simulation_results = None
        self.summary_stats = None
        
    def _validate_inputs(self, asset_names, expected_returns, volatilities, 
                        correlation_matrix, allocation_weights):
        """Validate input parameters for consistency."""
        
        n_assets = len(asset_names)
        
        # Check all lists have same length
        if not all(len(x) == n_assets for x in [expected_returns, volatilities, allocation_weights]):
            raise ValueError("All parameter lists must have the same length")
            
        # Check correlation matrix dimensions
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Correlation matrix must be {n_assets}x{n_assets}")
            
        # Check allocation weights sum to 1
        if not np.isclose(np.sum(allocation_weights), 1.0, atol=1e-6):
            raise ValueError("Allocation weights must sum to 1.0")
            
        # Check volatilities are positive
        if any(v <= 0 for v in volatilities):
            raise ValueError("All volatilities must be positive")
            
        print(f"✓ Input validation passed for {n_assets} asset classes")
    
    def _calculate_covariance_matrix(self) -> np.ndarray:
        """
        Calculate covariance matrix from correlation matrix and volatilities.
        
        Returns:
        --------
        np.ndarray
            Covariance matrix
        """
        # Convert correlation matrix to covariance matrix
        # Cov(i,j) = Corr(i,j) * Std(i) * Std(j)
        std_matrix = np.outer(self.volatilities, self.volatilities)
        covariance_matrix = self.correlation_matrix * std_matrix
        
        return covariance_matrix
    
    def simulate_portfolio(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate a single portfolio path over the specified number of years.
        
        Parameters:
        -----------
        random_seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Array of portfolio values for each year (length = years + 1)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize portfolio value
        portfolio_values = np.zeros(self.years + 1)
        portfolio_values[0] = self.initial_value
        
        # Initialize asset values
        asset_values = np.zeros((self.years + 1, len(self.asset_names)))
        asset_values[0] = self.initial_value * self.allocation_weights
        
        # Simulate each year
        for year in range(1, self.years + 1):
            # Generate correlated random returns using multivariate normal distribution
            random_returns = np.random.multivariate_normal(
                self.expected_returns, 
                self.covariance_matrix
            )
            
            # Apply returns to asset values
            asset_values[year] = asset_values[year - 1] * (1 + random_returns)
            
            # Calculate total portfolio value
            total_value = np.sum(asset_values[year])
            
            # Apply annual withdrawal starting at end of year 3
            if year >= 3 and total_value > self.annual_withdrawal:
                total_value -= self.annual_withdrawal
                
                # Apply taxes (simplified model)
                # TODO: Implement more sophisticated tax logic here
                # For now, assume 2% dividend yield and tax on dividends
                dividend_yield = 0.02
                dividend_income = total_value * dividend_yield
                dividend_tax = dividend_income * self.dividend_tax_rate
                total_value -= dividend_tax
                
                # Rebalance to target allocation
                asset_values[year] = total_value * self.allocation_weights
            elif year >= 3 and total_value <= self.annual_withdrawal:
                # Portfolio depleted when withdrawals start
                asset_values[year] = 0
                total_value = 0
            
            portfolio_values[year] = total_value
            
            # Stop simulation if portfolio is depleted
            if total_value <= 0:
                portfolio_values[year:] = 0
                break
        
        return portfolio_values
    
    def run_monte_carlo(self) -> pd.DataFrame:
        """
        Run Monte Carlo simulation for all iterations.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio values for all simulations
            Shape: (iterations, years + 1)
        """
        print(f"Running {self.iterations:,} Monte Carlo simulations...")
        
        # Initialize results array
        results = np.zeros((self.iterations, self.years + 1))
        
        # Run simulations
        for i in range(self.iterations):
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1:,} simulations...")
            
            results[i] = self.simulate_portfolio()
        
        # Convert to DataFrame
        columns = [f'Year_{i}' for i in range(self.years + 1)]
        self.simulation_results = pd.DataFrame(results, columns=columns)
        
        print(f"✓ Completed {self.iterations:,} simulations")
        return self.simulation_results
    
    def summarize_results(self) -> Dict:
        """
        Calculate summary statistics from simulation results.
        
        Returns:
        --------
        Dict
            Dictionary containing various summary statistics
        """
        if self.simulation_results is None:
            raise ValueError("Must run Monte Carlo simulation first")
        
        final_values = self.simulation_results.iloc[:, -1]
        
        # Calculate statistics
        summary = {
            'mean_final_value': final_values.mean(),
            'median_final_value': final_values.median(),
            'std_final_value': final_values.std(),
            'min_final_value': final_values.min(),
            'max_final_value': final_values.max(),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'probability_of_failure': (final_values <= 0).mean(),
            'probability_of_depletion': (final_values < self.annual_withdrawal).mean(),
            'success_rate': (final_values > self.initial_value).mean(),
            'average_annual_return': self._calculate_average_annual_return(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'prob_of_3x': (final_values > 3 * self.initial_value).mean()
        }
        
        self.summary_stats = summary
        return summary
    
    def _calculate_average_annual_return(self) -> float:
        """Calculate average annual return across all simulations."""
        if self.simulation_results is None:
            return 0
        
        # Calculate annual returns for each simulation
        annual_returns = []
        for i in range(self.iterations):
            values = self.simulation_results.iloc[i].values
            if values[-1] > 0:  # Only include non-depleted portfolios
                total_return = (values[-1] / values[0]) ** (1 / self.years) - 1
                annual_returns.append(total_return)
        
        return np.mean(annual_returns) if annual_returns else 0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio using risk-free rate."""
        avg_return = self._calculate_average_annual_return()
        if avg_return == 0:
            return 0
        
        # Use standard deviation of final values as proxy for risk
        final_values = self.simulation_results.iloc[:, -1]
        risk = final_values.std() / final_values.mean()
        
        return (avg_return - risk_free_rate) / risk if risk > 0 else 0
    
    def plot_simulation_paths(self, num_paths: int = 50, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot sample simulation paths with withdrawal timing indicators.
        
        Parameters:
        -----------
        num_paths : int
            Number of random paths to plot
        figsize : Tuple[int, int]
            Figure size
        """
        if self.simulation_results is None:
            raise ValueError("Must run Monte Carlo simulation first")
        
        plt.figure(figsize=figsize)
        
        # Plot sample paths
        sample_indices = np.random.choice(self.iterations, num_paths, replace=False)
        years = range(self.years + 1)
        
        for idx in sample_indices:
            plt.plot(years, self.simulation_results.iloc[idx], alpha=0.3, color='blue')
        
        # Plot median path
        median_path = self.simulation_results.median()
        plt.plot(years, median_path, color='red', linewidth=2, label='Median Path')
        
        # Plot percentiles
        percentile_5 = self.simulation_results.quantile(0.05)
        percentile_95 = self.simulation_results.quantile(0.95)
        plt.fill_between(years, percentile_5, percentile_95, alpha=0.2, color='gray', 
                        label='5th-95th Percentile Range')
        
        # Add vertical line to indicate when withdrawals start (end of year 3)
        plt.axvline(x=3, color='orange', linestyle='--', linewidth=2, 
                   label=f'Withdrawals Start (Year 3)')
        
        # Add vertical lines for each withdrawal year
        withdrawal_years = range(3, self.years + 1)
        for year in withdrawal_years:
            plt.axvline(x=year, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.xlabel('Years')
        plt.ylabel('Portfolio Value ($)')
        plt.title(f'Monte Carlo Simulation Paths\n({num_paths} Sample Paths)\nWithdrawals Start at End of Year 3')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_final_value_distribution(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot distribution of final portfolio values.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        """
        if self.simulation_results is None:
            raise ValueError("Must run Monte Carlo simulation first")
        
        final_values = self.simulation_results.iloc[:, -1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(final_values.mean(), color='red', linestyle='--', 
                   label=f'Mean: ${final_values.mean():,.0f}')
        ax1.axvline(final_values.median(), color='green', linestyle='--', 
                   label=f'Median: ${final_values.median():,.0f}')
        ax1.set_xlabel('Final Portfolio Value ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Portfolio Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(final_values, vert=True)
        ax2.set_ylabel('Final Portfolio Value ($)')
        ax2.set_title('Box Plot of Final Portfolio Values')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        if self.summary_stats is None:
            self.summarize_results()
        
        print("\n" + "="*60)
        print("PORTFOLIO SIMULATION SUMMARY REPORT")
        print("="*60)
        
        print(f"\nSimulation Parameters:")
        print(f"  Initial Portfolio Value: ${self.initial_value:,.0f}")
        print(f"  Annual Withdrawal: ${self.annual_withdrawal:,.0f} (starting at end of year 3)")
        print(f"  Simulation Period: {self.years} years")
        print(f"  Number of Simulations: {self.iterations:,}")
        
        print(f"\nAsset Allocation:")
        for name, weight in zip(self.asset_names, self.allocation_weights):
            print(f"  {name}: {weight:.1%}")
        
        print(f"\nExpected Returns:")
        for name, ret in zip(self.asset_names, self.expected_returns):
            print(f"  {name}: {ret:.1%}")
        
        print(f"\nResults Summary:")
        print(f"  Mean Final Value: ${self.summary_stats['mean_final_value']:,.0f}")
        print(f"  Median Final Value: ${self.summary_stats['median_final_value']:,.0f}")
        print(f"  5th Percentile: ${self.summary_stats['percentile_5']:,.0f}")
        print(f"  95th Percentile: ${self.summary_stats['percentile_95']:,.0f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Probability of Failure: {self.summary_stats['probability_of_failure']:.1%}")
        print(f"  Probability of Depletion: {self.summary_stats['probability_of_depletion']:.1%}")
        print(f"  Success Rate (>Initial): {self.summary_stats['success_rate']:.1%}")
        print(f"  Probability of 3x Return: {self.summary_stats['prob_of_3x']:.1%}")
        print(f"  Average Annual Return: {self.summary_stats['average_annual_return']:.1%}")
        print(f"  Sharpe Ratio: {self.summary_stats['sharpe_ratio']:.2f}")
        
        print("\n" + "="*60)


def create_sample_data() -> Tuple[List[str], List[float], List[float], np.ndarray, 
                                 float, List[float], float]:
    """
    Create sample data for demonstration purposes.
    
    Returns:
    --------
    Tuple containing all required parameters for PortfolioSimulator
    """
    
    # Asset names
    asset_names = ['US_Equities', 'Intl_Equities', 'Gold', 'Cash']
    
    # Expected annual returns (as decimals).
    expected_returns = [0.09, 0.07, 0.06, 0.02]
    
    # Annual volatilities (as decimals).
    volatilities = [0.175, 0.16, 0.16, 0.01]
    
    # Correlation matrix (mild positive correlations)
    correlation_matrix = np.array([
        [1.00, 0.70, 0.15, 0.05],  # US Equities
        [0.70, 1.00, 0.10, 0.01],  # International Equities
        [0.15, 0.10, 1.00, -0.40],  # Gold
        [0.05, 0.01, -0.40, 1.00]   # Cash
    ])
    
    # Portfolio parameters
    initial_value = 500_000
    allocation_weights = [0.1, 0.5, 0.10, 0.30]  # 50%, 30%, 10%, 10%
    annual_withdrawal = 12_000
    
    return (asset_names, expected_returns, volatilities, correlation_matrix,
            initial_value, allocation_weights, annual_withdrawal)


def main():
    """
    Main function to demonstrate the Monte Carlo portfolio simulation.
    """
    print("Monte Carlo Portfolio Simulation Framework")
    print("=" * 50)
    
    # Create sample data
    (asset_names, expected_returns, volatilities, correlation_matrix,
     initial_value, allocation_weights, annual_withdrawal) = create_sample_data()
    
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
        years=10,
        iterations=10000
    )
    
    # Run simulation
    results_df = simulator.run_monte_carlo()
    
    # Generate summary statistics
    summary = simulator.summarize_results()
    
    # Print comprehensive report
    simulator.print_summary_report()
    
    # Create visualizations
    # print("\nGenerating visualizations...")
    # simulator.plot_simulation_paths(num_paths=100)
    # simulator.plot_final_value_distribution()
    
    # Return results for further analysis
    return simulator, results_df, summary


if __name__ == "__main__":
    # Run the simulation
    simulator, results, summary = main()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nThe simulation results are stored in:")
    print("- simulator.simulation_results: DataFrame with all simulation paths")
    print("- simulator.summary_stats: Dictionary with summary statistics")
    print("\nYou can now:")
    print("1. Modify the sample data in create_sample_data() for your specific needs")
    print("2. Adjust tax logic in simulate_portfolio() method")
    print("3. Add more sophisticated rebalancing strategies")
    print("4. Include additional asset classes or constraints")
    print("5. Export results to CSV or other formats")