"""
Portfolio Optimization Script

This script runs Monte Carlo simulations for all possible allocation combinations
(0.05 increments) and saves results to CSV for analysis.

Updated to use rolling window covariance matrices from historical returns data.
"""

import numpy as np
import pandas as pd
from monte_carlo_portfolio import PortfolioSimulator
from tqdm import tqdm
import itertools
import time
import multiprocessing as mp
from functools import partial


# ============================================================================
# MODEL ASSUMPTIONS - Asset Class Parameters
# ============================================================================

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# Set to True to use real historical data from CSV file
# Set to False to use synthetic data based on model assumptions below
USE_REAL_DATA = False
HISTORICAL_DATA_FILE = 'historical_asset_returns.csv'

# Asset class names (order matters - must match CSV columns)
ASSET_NAMES = ['Emerging_Markets', 'Japan', 'Intl_Developed', 'Bonds', 'Gold', 'US_Equities']

# Expected monthly returns (as decimals) - used only if USE_REAL_DATA = False
# Order: Emerging Markets, Japan, Intl Developed, Bonds, Gold, US Equities
# Converted from annual using: monthly = (1 + annual)^(1/12) - 1
EXPECTED_RETURNS = [0.00934, 0.00764, 0.00784, 0.00322, 0.00656, 0.00844]

# Monthly volatilities (standard deviations, as decimals) - used only if USE_REAL_DATA = False
# Order: Emerging Markets, Japan, Intl Developed, Bonds, Gold, US Equities
VOLATILITIES = [0.0606, 0.0525, 0.0404, 0.0257, 0.0393, 0.0370]

# Correlation matrix between asset classes - used only if USE_REAL_DATA = False
# Rows/columns order: Emerging Markets, Japan, Intl Developed, Bonds, Gold, US Equities
CORRELATION_MATRIX = np.array([
    [1.00, 0.70, 0.75, -0.10, -0.05, 0.65],  # Emerging Markets
    [0.70, 1.00, 0.80, -0.10, -0.05, 0.60],  # Japan
    [0.75, 0.80, 1.00, -0.10, -0.05, 0.75],  # International Developed
    [-0.10, -0.10, -0.10, 1.00, -0.30, -0.10],  # Bonds
    [-0.05, -0.05, -0.05, -0.30, 1.00, -0.10],  # Gold
    [0.65, 0.60, 0.75, -0.10, -0.10, 1.00]   # US Equities
])


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Portfolio starting parameters
INITIAL_VALUE = 350_000          # Initial portfolio value ($)
ANNUAL_WITHDRAWAL = 12_000       # Annual withdrawal amount ($)

# Time horizon
YEARS = 10                       # Number of years to simulate
HISTORICAL_PERIODS = 120         # Number of months of historical data to generate

# Monte Carlo settings
ITERATIONS = 10000                   # Number of Monte Carlo simulations per allocation (reduced for speed)
WINDOW_SIZE = 24                 # Rolling window size in months for covariance calculation
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Use all CPU cores except one

# Tax rates
DIVIDEND_TAX_RATE = 0.20         # Tax rate on dividend income
CAPITAL_GAINS_TAX_RATE = 0.20    # Tax rate on capital gains

# Allocation constraints
# Order: Emerging Markets, Japan, Intl Developed, Bonds, Gold, US Equities
MIN_EMERGING_MARKETS = 0.0      # Minimum 40% in Emerging Markets
MAX_JAPAN = 0.40                 # Maximum 25% in Japan
MAX_INTL_DEVELOPED = 0.40        # Maximum 25% in International Developed
MAX_BONDS = 0.4                 # Maximum 25% in Bonds
MAX_GOLD = 1                  # Maximum 25% in Gold
# US Equities has no constraint (free variable)

# Allocation step size
ALLOCATION_STEP = 0.05           # 5% increments (0.05)

# Rebalancing settings (Hybrid approach: check frequently, rebalance only if needed)
REBALANCE_ENABLED = True         # Enable/disable rebalancing
REBALANCE_CHECK_FREQUENCY = 3    # Check for rebalancing every N months (3 = quarterly)
REBALANCE_THRESHOLD = 0.05       # Rebalance if any asset drifts ±5% from target allocation

# Output settings
TOP_N_STRATEGIES = 50            # Number of top strategies to save
TARGET_FINAL_VALUE = 1_350_000   # Success threshold for 3x metric ($)

# ============================================================================
# SPECIFIC ALLOCATIONS MODE
# ============================================================================
# Set to True to run simulations for specific allocations only (skips iteration)
RUN_SPECIFIC_ALLOCATIONS = True

# Define your specific allocations here (must sum to 1.0)
# Order: [Emerging_Markets, Japan, Intl_Developed, Bonds, Gold, US_Equities]
SPECIFIC_ALLOCATIONS = [
    [0.53623, 0.00, 0.00, 0.00, 0.00, 0.46377],  # Example: 35% EM, 5% JP, 5% INTL, 5% Bonds, 0% Gold, 50% US
]

def load_historical_returns(filepath):
    """
    Load historical returns from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with columns: Date, Asset1, Asset2, ...
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and asset returns as columns
    """
    print(f"\nLoading historical returns from: {filepath}")
    
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Convert first column to datetime index
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
        
        print(f"✓ Loaded {len(df)} months of historical data")
        print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        print(f"✓ Assets: {', '.join(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {filepath}")
        print(f"   Please create {filepath} with your historical data")
        print(f"   Expected format: Date, {', '.join(ASSET_NAMES)}")
        raise
    except Exception as e:
        print(f"❌ ERROR loading historical data: {e}")
        raise


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


def create_returns_df_from_parameters(asset_names, expected_returns, volatilities, 
                                     correlation_matrix, n_periods=120):
    """
    Create a historical returns DataFrame from expected returns, volatilities, and correlation matrix.
    
    This generates synthetic historical monthly returns that match the specified parameters.
    
    Parameters:
    -----------
    asset_names : List[str]
        Names of asset classes
    expected_returns : List[float]
        Expected monthly returns (as decimals)
    volatilities : List[float]
        Monthly volatilities (as decimals)
    correlation_matrix : np.ndarray
        Correlation matrix between assets
    n_periods : int
        Number of monthly periods to generate (default 120 = 10 years)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly returns, columns = asset classes, index = date
    """
    
    # Use monthly parameters directly (no conversion needed)
    monthly_means = np.array(expected_returns)
    monthly_vols = np.array(volatilities)
    
    # Create covariance matrix from correlation matrix and volatilities
    std_matrix = np.outer(monthly_vols, monthly_vols)
    covariance_matrix = correlation_matrix * std_matrix
    
    # Ensure covariance matrix is positive semi-definite
    min_eigenvalue = np.linalg.eigvals(covariance_matrix).min()
    if min_eigenvalue < 1e-8:
        covariance_matrix += np.eye(len(asset_names)) * 1e-8
    
    # Generate correlated monthly returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.multivariate_normal(monthly_means, covariance_matrix, size=n_periods)
    
    # Create date index (monthly)
    dates = pd.date_range(start='2014-01-01', periods=n_periods, freq='M')
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    return returns_df


# Global variable to store returns_df for multiprocessing workers
_global_returns_df = None

def _init_worker(returns_df_values, returns_df_index, returns_df_columns):
    """Initialize worker process with shared data."""
    global _global_returns_df
    _global_returns_df = pd.DataFrame(
        returns_df_values, 
        index=pd.to_datetime(returns_df_index), 
        columns=returns_df_columns
    )

def _worker_simulate(args):
    """Worker function for parallel simulation."""
    weights, initial_value, annual_withdrawal, years, iterations, window_size, \
        rebalance_enabled, rebalance_check_frequency, rebalance_threshold = args
    
    try:
        return run_simulation_for_allocation(
            weights=weights,
            returns_df=_global_returns_df,
            initial_value=initial_value,
            annual_withdrawal=annual_withdrawal,
            years=years,
            iterations=iterations,
            window_size=window_size,
            rebalance_enabled=rebalance_enabled,
            rebalance_check_frequency=rebalance_check_frequency,
            rebalance_threshold=rebalance_threshold
        )
    except Exception as e:
        print(f"\nError with allocation {weights}: {e}")
        return None


def run_simulation_for_allocation(weights,
                                  returns_df,
                                  initial_value, annual_withdrawal,
                                  years, iterations, window_size=36,
                                  rebalance_enabled=True,
                                  rebalance_check_frequency=3,
                                  rebalance_threshold=0.05,
                                  verbose=False):
    """
    Run a single Monte Carlo simulation for given allocation weights using rolling window covariances.
    
    Parameters:
    -----------
    weights : tuple
        Allocation weights for each asset
    returns_df : pd.DataFrame
        Historical monthly returns DataFrame
    initial_value : float
        Initial portfolio value
    annual_withdrawal : float
        Annual withdrawal amount
    years : int
        Number of years to simulate
    iterations : int
        Number of Monte Carlo iterations
    window_size : int
        Rolling window size in months (default 36)
    rebalance_enabled : bool
        Whether to enable rebalancing
    rebalance_check_frequency : int
        How often to check for rebalancing (in months)
    rebalance_threshold : float
        Drift threshold that triggers rebalancing
        
    Returns:
    --------
    Dict with results
    """
    allocation_weights = list(weights)
    
    # Initialize simulator with returns_df (new API)
    simulator = PortfolioSimulator(
        returns_df=returns_df,
        initial_value=initial_value,
        allocation_weights=allocation_weights,
        annual_withdrawal=annual_withdrawal,
        window_size=window_size,
        years=years,
        iterations=iterations,
        dividend_tax_rate=DIVIDEND_TAX_RATE,
        capital_gains_tax_rate=CAPITAL_GAINS_TAX_RATE,
        rebalance_enabled=rebalance_enabled,
        rebalance_check_frequency=rebalance_check_frequency,
        rebalance_threshold=rebalance_threshold,
        verbose=verbose
    )
    
    # Run simulation (now returns tuple of results_df and max_drawdowns)
    results_df, max_drawdowns = simulator.run_monte_carlo()
    
    # Calculate statistics
    final_values = results_df.iloc[:, -1]
    
    # Calculate required metrics
    mean_terminal = final_values.mean()
    median_terminal = final_values.median()
    percentile_5 = np.percentile(final_values, 5)
    
    # Success metrics
    prob_success = (final_values > initial_value).mean()  # Probability of ending above initial value
    prob_3x = (final_values > TARGET_FINAL_VALUE).mean()  # Probability of hitting target
    
    # Return and risk metrics
    avg_annual_return = simulator._calculate_average_annual_return()
    sharpe_ratio = simulator._calculate_sharpe_ratio()
    sortino_ratio = simulator._calculate_sortino_ratio()
    
    # Volatility (standard deviation of annual returns across simulations)
    annual_returns = []
    for i in range(iterations):
        values = results_df.iloc[i].values
        if values[-1] > 0:  # Only include non-depleted portfolios
            total_return = (values[-1] / values[0]) ** (12 / (years * 12)) - 1
            annual_returns.append(total_return)
    portfolio_volatility = np.std(annual_returns) if annual_returns else 0
    
    # Max drawdown (mean and worst case)
    mean_max_drawdown = max_drawdowns.mean()
    worst_drawdown = max_drawdowns.min()  # Most negative = worst drawdown
    
    return {
        'Emerging_Markets_Weight': allocation_weights[0],
        'Japan_Weight': allocation_weights[1],
        'Intl_Developed_Weight': allocation_weights[2],
        'Bonds_Weight': allocation_weights[3],
        'Gold_Weight': allocation_weights[4],
        'US_Equities_Weight': allocation_weights[5],
        'Expected_Annual_Return': avg_annual_return * 100,  # Convert to percentage
        'Volatility': portfolio_volatility * 100,  # Convert to percentage
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Mean_Terminal_Wealth': mean_terminal,
        'Median_Terminal_Wealth': median_terminal,
        '5th_Percentile_Outcome': percentile_5,
        'Probability_of_Success_Pct': prob_success * 100,  # Ending above initial value
        'Success_3x_Pct': prob_3x * 100,  # Convert to percentage
        'Mean_Max_Drawdown': mean_max_drawdown * 100,  # Convert to percentage
        'Worst_Max_Drawdown': worst_drawdown * 100  # Convert to percentage
    }


def run_specific_allocations_simulation():
    """
    Run simulations for specific allocations defined in SPECIFIC_ALLOCATIONS.
    Saves results to specificSIM.csv
    """
    print("="*70)
    print("SPECIFIC ALLOCATIONS SIMULATION")
    print("="*70)
    print("\nCONFIGURATION:")
    print(f"  Data Source: {'Real Historical Data' if USE_REAL_DATA else 'Synthetic Data'}")
    if USE_REAL_DATA:
        print(f"  Data File: {HISTORICAL_DATA_FILE}")
    print(f"  Assets: {', '.join(ASSET_NAMES)}")
    print(f"  Initial Value: ${INITIAL_VALUE:,}")
    print(f"  Annual Withdrawal: ${ANNUAL_WITHDRAWAL:,}")
    print(f"  Time Horizon: {YEARS} years")
    print(f"  Monte Carlo Iterations: {ITERATIONS:,}")
    print(f"  Rolling Window Size: {WINDOW_SIZE} months")
    print(f"  Rebalancing: {'Enabled' if REBALANCE_ENABLED else 'Disabled'}")
    if REBALANCE_ENABLED:
        print(f"    Check every {REBALANCE_CHECK_FREQUENCY} months, rebalance if drift >{REBALANCE_THRESHOLD*100:.0f}%")
    print(f"\n  Number of specific allocations: {len(SPECIFIC_ALLOCATIONS)}")
    print("="*70)
    
    # Validate allocations
    print("\nValidating allocations...")
    valid_allocations = []
    for i, alloc in enumerate(SPECIFIC_ALLOCATIONS):
        if len(alloc) != 6:
            print(f"  ❌ Allocation {i+1}: Wrong number of assets (got {len(alloc)}, expected 6)")
            continue
        total = sum(alloc)
        if abs(total - 1.0) > 0.001:
            print(f"  ❌ Allocation {i+1}: Weights sum to {total:.4f}, not 1.0")
            continue
        if any(w < 0 for w in alloc):
            print(f"  ❌ Allocation {i+1}: Contains negative weights")
            continue
        valid_allocations.append(tuple(alloc))
        print(f"  ✓ Allocation {i+1}: EM={alloc[0]:.0%}, JP={alloc[1]:.0%}, INTL={alloc[2]:.0%}, "
              f"Bonds={alloc[3]:.0%}, Gold={alloc[4]:.0%}, US={alloc[5]:.0%}")
    
    if not valid_allocations:
        print("\n❌ No valid allocations to simulate!")
        return None
    
    print(f"\n✓ {len(valid_allocations)} valid allocations to simulate")
    
    # Load or generate historical returns DataFrame
    if USE_REAL_DATA:
        returns_df = load_historical_returns(HISTORICAL_DATA_FILE)
        if list(returns_df.columns) != ASSET_NAMES:
            print(f"⚠ WARNING: CSV columns ({list(returns_df.columns)}) don't match ASSET_NAMES ({ASSET_NAMES})")
    else:
        print("\nGenerating synthetic historical returns from model assumptions...")
        returns_df = create_returns_df_from_parameters(
            asset_names=ASSET_NAMES,
            expected_returns=EXPECTED_RETURNS,
            volatilities=VOLATILITIES,
            correlation_matrix=CORRELATION_MATRIX,
            n_periods=HISTORICAL_PERIODS
        )
        print(f"✓ Created {len(returns_df)} months of synthetic returns data")
    
    # Prepare arguments for parallel processing
    args_list = [
        (combo, INITIAL_VALUE, ANNUAL_WITHDRAWAL, YEARS, ITERATIONS, WINDOW_SIZE,
         REBALANCE_ENABLED, REBALANCE_CHECK_FREQUENCY, REBALANCE_THRESHOLD)
        for combo in valid_allocations
    ]
    
    # Run simulations
    print(f"\nRunning simulations with {NUM_PROCESSES} parallel processes...")
    print(f"(Using {ITERATIONS:,} iterations per allocation)")
    start_time = time.time()
    
    # Prepare data for workers
    returns_values = returns_df.values
    returns_index = returns_df.index.astype(str).tolist()
    returns_columns = returns_df.columns.tolist()
    
    with mp.Pool(
        processes=NUM_PROCESSES,
        initializer=_init_worker,
        initargs=(returns_values, returns_index, returns_columns)
    ) as pool:
        results = []
        for result in tqdm(
            pool.imap(_worker_simulate, args_list),
            total=len(args_list),
            desc="Progress",
            unit="alloc",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            if result is not None:
                results.append(result)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed_time:.1f} seconds")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'Emerging_Markets_Weight',
        'Japan_Weight',
        'Intl_Developed_Weight',
        'Bonds_Weight',
        'Gold_Weight',
        'US_Equities_Weight',
        'Expected_Annual_Return',
        'Volatility',
        'Sharpe_Ratio',
        'Sortino_Ratio',
        'Mean_Terminal_Wealth',
        'Median_Terminal_Wealth',
        '5th_Percentile_Outcome',
        'Probability_of_Success_Pct',
        'Success_3x_Pct',
        'Mean_Max_Drawdown',
        'Worst_Max_Drawdown'
    ]
    results_df = results_df[[col for col in column_order if col in results_df.columns]]
    
    # Save to CSV
    output_file = 'specificSIM.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for idx, row in results_df.iterrows():
        print(f"\nAllocation {idx+1}:")
        print(f"  Weights: EM={row['Emerging_Markets_Weight']:.0%}, JP={row['Japan_Weight']:.0%}, "
              f"INTL={row['Intl_Developed_Weight']:.0%}, Bonds={row['Bonds_Weight']:.0%}, "
              f"Gold={row['Gold_Weight']:.0%}, US={row['US_Equities_Weight']:.0%}")
        print(f"  Mean Terminal Wealth: ${row['Mean_Terminal_Wealth']:,.0f}")
        print(f"  Success Rate (3x): {row['Success_3x_Pct']:.2f}%")
        print(f"  Sharpe Ratio: {row['Sharpe_Ratio']:.3f}")
        print(f"  Mean Max Drawdown: {row['Mean_Max_Drawdown']:.2f}%")
    
    return results_df


def main():
    """
    Main optimization function
    """
    # Check if running specific allocations mode
    if RUN_SPECIFIC_ALLOCATIONS:
        return run_specific_allocations_simulation()
    
    print("="*70)
    print("PORTFOLIO ALLOCATION OPTIMIZER")
    print("="*70)
    print("\nCONFIGURATION:")
    print(f"  Data Source: {'Real Historical Data' if USE_REAL_DATA else 'Synthetic Data'}")
    if USE_REAL_DATA:
        print(f"  Data File: {HISTORICAL_DATA_FILE}")
    print(f"  Assets: {', '.join(ASSET_NAMES)}")
    print(f"  Initial Value: ${INITIAL_VALUE:,}")
    print(f"  Annual Withdrawal: ${ANNUAL_WITHDRAWAL:,}")
    print(f"  Time Horizon: {YEARS} years")
    print(f"  Monte Carlo Iterations: {ITERATIONS:,}")
    print(f"  Rolling Window Size: {WINDOW_SIZE} months")
    print(f"  Allocation Step: {ALLOCATION_STEP*100:.0f}%")
    print(f"  Rebalancing: {'Enabled' if REBALANCE_ENABLED else 'Disabled'}")
    if REBALANCE_ENABLED:
        print(f"    Check every {REBALANCE_CHECK_FREQUENCY} months, rebalance if drift >{REBALANCE_THRESHOLD*100:.0f}%")
    print(f"  Constraints: EM>={MIN_EMERGING_MARKETS*100:.0f}%, JP<={MAX_JAPAN*100:.0f}%, "
          f"INTL<={MAX_INTL_DEVELOPED*100:.0f}%, Bonds<={MAX_BONDS*100:.0f}%, "
          f"Gold<={MAX_GOLD*100:.0f}%")
    print("="*70)
    
    # Load or generate historical returns DataFrame
    if USE_REAL_DATA:
        # Load real historical data from CSV
        returns_df = load_historical_returns(HISTORICAL_DATA_FILE)
        
        # Verify columns match expected assets
        if list(returns_df.columns) != ASSET_NAMES:
            print(f"⚠ WARNING: CSV columns ({list(returns_df.columns)}) don't match ASSET_NAMES ({ASSET_NAMES})")
            print(f"   Using CSV columns as asset names")
    else:
        # Generate synthetic historical data from model assumptions
        print("\nGenerating synthetic historical returns from model assumptions...")
        returns_df = create_returns_df_from_parameters(
            asset_names=ASSET_NAMES,
            expected_returns=EXPECTED_RETURNS,
            volatilities=VOLATILITIES,
            correlation_matrix=CORRELATION_MATRIX,
            n_periods=HISTORICAL_PERIODS
        )
        print(f"✓ Created {len(returns_df)} months of synthetic returns data")
    
    # Generate all allocation combinations based on configured step size and constraints
    print("\nGenerating allocation combinations (6 assets, 5% steps, constrained)...")
    combinations = []
    step_units = int(1.0 / ALLOCATION_STEP)  # e.g., 20 units for 5% steps
    
    # Convert constraints to step units
    min_em_units = int(MIN_EMERGING_MARKETS / ALLOCATION_STEP)
    max_jp_units = int(MAX_JAPAN / ALLOCATION_STEP)
    max_intl_units = int(MAX_INTL_DEVELOPED / ALLOCATION_STEP)
    max_bonds_units = int(MAX_BONDS / ALLOCATION_STEP)
    max_gold_units = int(MAX_GOLD / ALLOCATION_STEP)
    
    # Order: EM(a), JP(b), INTL(c), BONDS(d), GOLD(e), US(f)
    for a in range(min_em_units, step_units + 1):  # EM >= MIN_EMERGING_MARKETS
        remaining_a = step_units - a
        for b in range(0, min(max_jp_units, remaining_a) + 1):  # Japan <= MAX_JAPAN
            remaining_b = remaining_a - b
            for c in range(0, min(max_intl_units, remaining_b) + 1):  # Intl <= MAX_INTL_DEVELOPED
                remaining_c = remaining_b - c
                for d in range(0, min(max_bonds_units, remaining_c) + 1):  # Bonds <= MAX_BONDS
                    remaining_d = remaining_c - d
                    for e in range(0, min(max_gold_units, remaining_d) + 1):  # Gold <= MAX_GOLD
                        f = remaining_d - e  # US free
                        if f < 0:
                            continue
                        weights = [a, b, c, d, e, f]
                        combinations.append(tuple(w / step_units for w in weights))
    print(f"Total combinations to test: {len(combinations)}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (combo, INITIAL_VALUE, ANNUAL_WITHDRAWAL, YEARS, ITERATIONS, WINDOW_SIZE,
         REBALANCE_ENABLED, REBALANCE_CHECK_FREQUENCY, REBALANCE_THRESHOLD)
        for combo in combinations
    ]
    
    # Run simulations in parallel
    print(f"\nRunning simulations with {NUM_PROCESSES} parallel processes...")
    print(f"(Using {ITERATIONS:,} iterations per allocation)")
    start_time = time.time()
    
    # Prepare data for workers (convert DataFrame to serializable format)
    returns_values = returns_df.values
    returns_index = returns_df.index.astype(str).tolist()
    returns_columns = returns_df.columns.tolist()
    
    with mp.Pool(
        processes=NUM_PROCESSES,
        initializer=_init_worker,
        initargs=(returns_values, returns_index, returns_columns)
    ) as pool:
        # Use imap for progress tracking
        results = []
        for result in tqdm(
            pool.imap(_worker_simulate, args_list),
            total=len(args_list),
            desc="Progress",
            unit="alloc",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            if result is not None:
                results.append(result)
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by 3x success rate (descending - highest first)
    results_df = results_df.sort_values('Success_3x_Pct', ascending=False)
    
    # Take only top N strategies
    results_df = results_df.head(TOP_N_STRATEGIES)
    
    # Reorder columns to match desired format
    column_order = [
        'Emerging_Markets_Weight',
        'Japan_Weight',
        'Intl_Developed_Weight',
        'Bonds_Weight',
        'Gold_Weight',
        'US_Equities_Weight',
        'Expected_Annual_Return',
        'Volatility',
        'Sharpe_Ratio',
        'Sortino_Ratio',
        'Mean_Terminal_Wealth',
        'Median_Terminal_Wealth',
        '5th_Percentile_Outcome',
        'Probability_of_Success_Pct',
        'Success_3x_Pct',
        'Mean_Max_Drawdown',
        'Worst_Max_Drawdown'
    ]
    
    # Reorder columns (only include columns that exist)
    results_df = results_df[[col for col in column_order if col in results_df.columns]]
    
    # Save to CSV (change filename for different constraint runs)
    output_file = f'top_{TOP_N_STRATEGIES}_LOOSEconstraints.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file} (top {TOP_N_STRATEGIES} strategies by 3x success rate)")
    print(f"Total combinations tested: {len(results)}")
    print(f"Top {TOP_N_STRATEGIES} strategies saved (sorted by Success_3x_Pct, highest first)")
    print(f"  Highest success rate: {results_df['Success_3x_Pct'].iloc[0]:.2f}%")
    print(f"  Lowest success rate in top {TOP_N_STRATEGIES}: {results_df['Success_3x_Pct'].iloc[-1]:.2f}%")
    
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
    print("Done!")