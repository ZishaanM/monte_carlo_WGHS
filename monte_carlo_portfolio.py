"""
Monte Carlo Portfolio Simulation Framework with Rolling Window Covariance Matrices

This script provides a comprehensive framework for simulating portfolio returns
over multiple periods using Monte Carlo methods with rolling historical covariance
matrices and mean returns. It tracks max drawdown for each simulation.

Modified to use:
- Rolling window covariance matrices from historical data
- Random sampling from historical windows at each simulation step
- Max drawdown calculation for each simulation path

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
    A comprehensive Monte Carlo portfolio simulation framework using rolling window
    covariance matrices from historical data.
    
    This class handles portfolio simulation with multiple asset classes,
    correlated returns based on historical rolling windows, and max drawdown tracking.
    """
    
    def __init__(self, 
                 returns_df: pd.DataFrame,
                 initial_value: float,
                 allocation_weights: List[float],
                 annual_withdrawal: float = 0,
                 window_size: int = 36,
                 years: int = 10,
                 iterations: int = 10000,
                 dividend_tax_rate: float = 0.15,
                 capital_gains_tax_rate: float = 0.20,
                 rebalance_enabled: bool = True,
                 rebalance_check_frequency: int = 3,
                 rebalance_threshold: float = 0.05):
        """
        Initialize the portfolio simulator with historical returns data.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical monthly returns with columns = asset classes, index = date
            Shape: (time periods x assets)
        initial_value : float
            Initial portfolio value
        allocation_weights : List[float]
            Target allocation weights for each asset (should sum to 1.0)
        annual_withdrawal : float
            Annual withdrawal amount (default 0)
        window_size : int
            Rolling window size in months for covariance/mean calculation (default 36)
        years : int
            Number of years to simulate (default 10)
        iterations : int
            Number of Monte Carlo iterations (default 10000)
        dividend_tax_rate : float
            Tax rate on dividend income (default 0.15)
        capital_gains_tax_rate : float
            Tax rate on capital gains (default 0.20)
        rebalance_enabled : bool
            Whether to enable portfolio rebalancing (default True)
        rebalance_check_frequency : int
            How often to check if rebalancing is needed, in months (default 3)
        rebalance_threshold : float
            Drift threshold that triggers rebalancing (default 0.05 = 5%)
        """
        
        # Validate inputs
        self._validate_inputs(returns_df, allocation_weights, window_size)
        
        # Store parameters
        self.returns_df = returns_df.copy()
        self.asset_names = list(returns_df.columns)
        self.n_assets = len(self.asset_names)
        self.initial_value = initial_value
        self.allocation_weights = np.array(allocation_weights)
        self.annual_withdrawal = annual_withdrawal
        self.window_size = window_size
        self.years = years
        self.months = years * 12  # Convert years to months for monthly simulation
        self.iterations = iterations
        self.dividend_tax_rate = dividend_tax_rate
        self.capital_gains_tax_rate = capital_gains_tax_rate
        self.rebalance_enabled = rebalance_enabled
        self.rebalance_check_frequency = rebalance_check_frequency
        self.rebalance_threshold = rebalance_threshold
        
        # Pre-compute rolling window covariance matrices and mean returns
        print(f"Computing rolling window statistics (window size: {window_size} months)...")
        self.rolling_covariances, self.rolling_means = self._compute_rolling_statistics()
        print(f"✓ Computed {len(self.rolling_covariances)} rolling windows")
        
        # Initialize results storage
        self.simulation_results = None
        self.max_drawdowns = None
        self.summary_stats = None
        
    def _validate_inputs(self, returns_df: pd.DataFrame, 
                        allocation_weights: List[float], 
                        window_size: int):
        """
        Validate input parameters for consistency.
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Historical returns DataFrame
        allocation_weights : List[float]
            Allocation weights list
        window_size : int
            Rolling window size
        """
        
        n_assets = len(returns_df.columns)
        
        # Check returns_df has enough data
        if len(returns_df) < window_size:
            raise ValueError(f"returns_df must have at least {window_size} periods, "
                           f"but has {len(returns_df)}")
        
        # Check allocation weights match number of assets
        if len(allocation_weights) != n_assets:
            raise ValueError(f"allocation_weights length ({len(allocation_weights)}) "
                           f"must match number of assets ({n_assets})")
        
        # Check allocation weights sum to 1
        if not np.isclose(np.sum(allocation_weights), 1.0, atol=1e-6):
            raise ValueError("Allocation weights must sum to 1.0")
        
        # Check for missing values
        if returns_df.isnull().any().any():
            raise ValueError("returns_df contains missing values. Please handle missing data first.")
        
        print(f"✓ Input validation passed for {n_assets} asset classes with {len(returns_df)} periods")
    
    def _compute_rolling_statistics(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute rolling covariance matrices and mean returns for all possible windows.
        
        Returns:
        --------
        Tuple[List[np.ndarray], List[np.ndarray]]
            List of covariance matrices and list of mean return vectors
            Each list has length = (len(returns_df) - window_size + 1)
        """
        
        rolling_covariances = []
        rolling_means = []
        
        # Iterate through all possible rolling windows
        for i in range(len(self.returns_df) - self.window_size + 1):
            # Extract window data
            window_data = self.returns_df.iloc[i:i + self.window_size]
            
            # Calculate mean returns over the window (monthly means)
            mean_returns = window_data.mean().values
            
            # Calculate covariance matrix over the window (monthly covariance)
            covariance_matrix = window_data.cov().values
            
            # Ensure covariance matrix is positive semi-definite (add small regularization if needed)
            # This handles potential numerical issues with small windows
            min_eigenvalue = np.linalg.eigvals(covariance_matrix).min()
            if min_eigenvalue < 1e-8:
                covariance_matrix += np.eye(self.n_assets) * 1e-8
            
            rolling_covariances.append(covariance_matrix)
            rolling_means.append(mean_returns)
        
        return rolling_covariances, rolling_means
    
    def _needs_rebalancing(self, current_weights: np.ndarray) -> bool:
        """
        Check if portfolio needs rebalancing based on drift threshold.
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current allocation weights
            
        Returns:
        --------
        bool
            True if any asset has drifted beyond threshold
        """
        if not self.rebalance_enabled:
            return False
        
        # Calculate drift for each asset
        drifts = np.abs(current_weights - self.allocation_weights)
        
        # Rebalance if any asset has drifted beyond threshold
        return np.any(drifts > self.rebalance_threshold)
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown for a portfolio value path.
        
        Maximum drawdown is the maximum decline from peak to trough.
        Formula: MDD = min((portfolio_value / running_max) - 1)
        
        Parameters:
        -----------
        portfolio_values : np.ndarray
            Array of portfolio values over time
            
        Returns:
        --------
        float
            Maximum drawdown (negative value, e.g., -0.25 means 25% drawdown)
        """
        
        if len(portfolio_values) == 0 or np.all(portfolio_values <= 0):
            return 0.0
        
        # Compute running maximum of portfolio value
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown at each step: (value / running_max) - 1
        # This gives negative values for drawdowns
        drawdowns = (portfolio_values / running_max) - 1
        
        # Maximum drawdown is the most negative value (largest decline)
        max_drawdown = np.min(drawdowns)
        
        return max_drawdown
    
    def simulate_portfolio(self, random_seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Simulate a single portfolio path over the specified number of months.
        
        At each step, randomly selects a historical rolling window and uses its
        covariance matrix and mean returns to generate asset returns.
        
        Parameters:
        -----------
        random_seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Array of portfolio values for each month (length = months + 1)
            and maximum drawdown for this simulation
        """
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize portfolio value array (one for each month + initial value)
        portfolio_values = np.zeros(self.months + 1)
        portfolio_values[0] = self.initial_value
        
        # Initialize asset values
        asset_values = np.zeros((self.months + 1, self.n_assets))
        asset_values[0] = self.initial_value * self.allocation_weights
        
        # Number of available rolling windows
        n_windows = len(self.rolling_covariances)
        
        # Simulate each month
        for month in range(1, self.months + 1):
            # Randomly select a historical rolling window
            window_idx = np.random.randint(0, n_windows)
            
            # Get mean returns and covariance matrix for selected window
            mean_returns = self.rolling_means[window_idx].copy()
            covariance_matrix = self.rolling_covariances[window_idx].copy()
            
            # Generate correlated random returns using multivariate normal distribution
            # Using monthly returns from historical data
            try:
                monthly_returns = np.random.multivariate_normal(
                    mean_returns, 
                    covariance_matrix
                )
            except np.linalg.LinAlgError:
                # Handle case where covariance matrix might be singular
                # Add small regularization and retry
                covariance_matrix += np.eye(self.n_assets) * 1e-6
                monthly_returns = np.random.multivariate_normal(
                    mean_returns, 
                    covariance_matrix
                )
            
            # Apply returns to asset values
            asset_values[month] = asset_values[month - 1] * (1 + monthly_returns)
            
            # Calculate total portfolio value
            total_value = np.sum(asset_values[month])
            
            # Apply annual withdrawal (converted to monthly, applied at year end)
            withdrawal_occurred = False
            if month % 12 == 0 and month >= 36:  # Start withdrawals after 3 years (36 months)
                if total_value > self.annual_withdrawal:
                    total_value -= self.annual_withdrawal
                    withdrawal_occurred = True
                    
                    # Apply taxes (simplified model)
                    # Assume 2% annual dividend yield (monthly = 2%/12)
                    monthly_dividend_yield = 0.02 / 12
                    dividend_income = total_value * monthly_dividend_yield
                    dividend_tax = dividend_income * self.dividend_tax_rate
                    total_value -= dividend_tax
                    
                    # Update asset values proportionally after withdrawal
                    asset_values[month] = asset_values[month] * (total_value / np.sum(asset_values[month]))
                elif total_value <= self.annual_withdrawal:
                    # Portfolio depleted
                    asset_values[month] = 0
                    total_value = 0
            
            # Hybrid rebalancing: check periodically and rebalance if drift exceeds threshold
            if self.rebalance_enabled and month % self.rebalance_check_frequency == 0 and total_value > 0:
                # Calculate current weights
                current_weights = asset_values[month] / total_value
                
                # Check if rebalancing is needed
                if self._needs_rebalancing(current_weights):
                    # Rebalance to target allocation
                    asset_values[month] = total_value * self.allocation_weights
            
            portfolio_values[month] = total_value
            
            # Stop simulation if portfolio is depleted
            if total_value <= 0:
                portfolio_values[month:] = 0
                break
        
        # Calculate max drawdown for this simulation
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return portfolio_values, max_drawdown
    
    def run_monte_carlo(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Run Monte Carlo simulation for all iterations.
        
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            DataFrame with portfolio values for all simulations (shape: iterations x months+1)
            Array of max drawdowns for each simulation (shape: iterations,)
        """
        
        print(f"Running {self.iterations:,} Monte Carlo simulations ({self.months} months each)...")
        
        # Initialize results array
        results = np.zeros((self.iterations, self.months + 1))
        max_drawdowns = np.zeros(self.iterations)
        
        # Run simulations
        for i in range(self.iterations):
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1:,} simulations...")
            
            portfolio_path, mdd = self.simulate_portfolio()
            results[i] = portfolio_path
            max_drawdowns[i] = mdd
        
        # Convert to DataFrame
        columns = [f'Month_{i}' for i in range(self.months + 1)]
        self.simulation_results = pd.DataFrame(results, columns=columns)
        self.max_drawdowns = max_drawdowns
        
        print(f"✓ Completed {self.iterations:,} simulations")
        return self.simulation_results, self.max_drawdowns
    
    def summarize_results(self) -> Dict:
        """
        Calculate summary statistics from simulation results.
        
        Returns:
        --------
        Dict
            Dictionary containing various summary statistics including max drawdown stats
        """
        
        if self.simulation_results is None:
            raise ValueError("Must run Monte Carlo simulation first")
        
        # Get final values (last month of each simulation)
        final_values = self.simulation_results.iloc[:, -1]
        
        # Calculate statistics for final values
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
        
        # Add max drawdown statistics
        if self.max_drawdowns is not None:
            summary.update({
                'mean_max_drawdown': self.max_drawdowns.mean(),
                'median_max_drawdown': self.max_drawdowns.median(),
                'std_max_drawdown': self.max_drawdowns.std(),
                'min_max_drawdown': self.max_drawdowns.min(),
                'max_max_drawdown': self.max_drawdowns.max(),
                'percentile_5_max_drawdown': np.percentile(self.max_drawdowns, 5),
                'percentile_95_max_drawdown': np.percentile(self.max_drawdowns, 95),
                'prob_severe_drawdown': (self.max_drawdowns < -0.30).mean()  # >30% drawdown
            })
        
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
                # Convert monthly returns to annual
                total_return = (values[-1] / values[0]) ** (12 / self.months) - 1
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
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio using risk-free rate.
        Sortino ratio only considers downside deviation (negative returns).
        """
        
        avg_return = self._calculate_average_annual_return()
        if avg_return == 0:
            return 0
        
        # Calculate annual returns for each simulation
        annual_returns = []
        for i in range(self.iterations):
            values = self.simulation_results.iloc[i].values
            if values[-1] > 0:  # Only include non-depleted portfolios
                total_return = (values[-1] / values[0]) ** (12 / self.months) - 1
                annual_returns.append(total_return)
        
        if not annual_returns:
            return 0
        
        annual_returns = np.array(annual_returns)
        
        # Calculate downside deviation (only consider returns below risk-free rate)
        downside_returns = annual_returns[annual_returns < risk_free_rate] - risk_free_rate
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
        
        return (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    def plot_simulation_paths(self, num_paths: int = 50, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot sample simulation paths.
        
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
        months = range(self.months + 1)
        
        for idx in sample_indices:
            plt.plot(months, self.simulation_results.iloc[idx], alpha=0.3, color='blue')
        
        # Plot median path
        median_path = self.simulation_results.median()
        plt.plot(months, median_path, color='red', linewidth=2, label='Median Path')
        
        # Plot percentiles
        percentile_5 = self.simulation_results.quantile(0.05)
        percentile_95 = self.simulation_results.quantile(0.95)
        plt.fill_between(months, percentile_5, percentile_95, alpha=0.2, color='gray', 
                        label='5th-95th Percentile Range')
        
        # Add vertical line to indicate when withdrawals start (end of month 36)
        if self.annual_withdrawal > 0:
            plt.axvline(x=36, color='orange', linestyle='--', linewidth=2, 
                       label=f'Withdrawals Start (Month 36)')
        
        plt.xlabel('Months')
        plt.ylabel('Portfolio Value ($)')
        plt.title(f'Monte Carlo Simulation Paths\n({num_paths} Sample Paths)')
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
        