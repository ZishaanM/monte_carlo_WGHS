"""
Analyze rolling window covariance matrices used in the simulation.
"""

import numpy as np
import pandas as pd
from monte_carlo_portfolio import PortfolioSimulator
from portfolio_optimizer import create_returns_df_from_parameters


def analyze_rolling_covariances(returns_df, window_size=36):
    """
    Analyze and display the rolling covariance matrices used in simulations.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Historical monthly returns DataFrame
    window_size : int
        Rolling window size in months (default 36)
        
    Returns:
    --------
    Dict with analysis results
    """
    
    print("="*70)
    print("ROLLING WINDOW COVARIANCE ANALYSIS")
    print("="*70)
    
    # Create a temporary simulator to access rolling statistics
    # We'll use dummy weights since we only need to compute the statistics
    dummy_weights = [1.0 / len(returns_df.columns)] * len(returns_df.columns)
    
    simulator = PortfolioSimulator(
        returns_df=returns_df,
        initial_value=100000,
        allocation_weights=dummy_weights,
        window_size=window_size,
        years=1,
        iterations=1
    )
    
    # Access pre-computed rolling statistics
    rolling_covariances = simulator.rolling_covariances
    rolling_means = simulator.rolling_means
    
    print(f"\nTotal Number of Rolling Windows: {len(rolling_covariances)}")
    print(f"Window Size: {window_size} months")
    print(f"Asset Names: {list(returns_df.columns)}")
    
    # Analyze covariance statistics across all windows
    print(f"\nCovariance Matrix Statistics Across All Windows:")
    print(f"  (Mean of each covariance element across all windows)")
    
    # Calculate mean covariance matrix across all windows
    mean_covariance = np.mean(rolling_covariances, axis=0)
    mean_corr_matrix = np.corrcoef(returns_df.values.T)
    
    print(f"\nMean Covariance Matrix (across all {len(rolling_covariances)} windows):")
    cov_df = pd.DataFrame(mean_covariance, 
                         index=returns_df.columns, 
                         columns=returns_df.columns)
    print(cov_df)
    
    # Show correlation matrix (derived from covariance)
    print(f"\nMean Correlation Matrix (derived from covariances):")
    # Calculate correlation from mean covariance
    std_devs = np.sqrt(np.diag(mean_covariance))
    mean_corr = mean_covariance / np.outer(std_devs, std_devs)
    corr_df = pd.DataFrame(mean_corr,
                          index=returns_df.columns,
                          columns=returns_df.columns)
    print(corr_df.round(3))
    
    # Analyze volatility of covariances (how much they vary across windows)
    cov_std = np.std(rolling_covariances, axis=0)
    print(f"\nStandard Deviation of Covariances Across Windows:")
    print("  (Shows how much covariances vary over time)")
    cov_std_df = pd.DataFrame(cov_std,
                              index=returns_df.columns,
                              columns=returns_df.columns)
    print(cov_std_df.round(6))
    
    # Analyze mean returns across windows
    mean_returns_matrix = np.mean(rolling_means, axis=0)
    print(f"\nMean Returns Across All Windows (monthly):")
    mean_ret_df = pd.DataFrame(mean_returns_matrix,
                               index=returns_df.columns,
                               columns=['Mean Monthly Return'])
    print(mean_ret_df)
    
    # Show sample covariance matrices
    print(f"\nSample Covariance Matrices (first, middle, last windows):")
    
    indices = [0, len(rolling_covariances) // 2, len(rolling_covariances) - 1]
    window_labels = ['First Window', 'Middle Window', 'Last Window']
    
    for idx, label in zip(indices, window_labels):
        print(f"\n{label} (Window {idx + 1}):")
        sample_cov = pd.DataFrame(rolling_covariances[idx],
                                  index=returns_df.columns,
                                  columns=returns_df.columns)
        print(sample_cov)
        
        # Show corresponding mean returns for this window
        print(f"  Mean Returns for this window (monthly):")
        sample_means = pd.DataFrame(rolling_means[idx],
                                    index=returns_df.columns,
                                    columns=['Monthly Return'])
        print(sample_means)
    
    return {
        'rolling_covariances': rolling_covariances,
        'rolling_means': rolling_means,
        'mean_covariance': mean_covariance,
        'mean_correlation': mean_corr,
        'covariance_std': cov_std,
        'asset_names': list(returns_df.columns)
    }


if __name__ == "__main__":
    """
    To use this script with your portfolio optimizer data:
    
    1. Recreate the returns_df from your portfolio_optimizer parameters, OR
    2. Load a saved returns_df if you've exported it, OR
    3. Generate sample data matching your parameters
    """
    
    # Option 1: Recreate from portfolio_optimizer parameters
    from portfolio_optimizer import create_returns_df_from_parameters
    
    asset_names = ['Emerging_Markets', 'Japan', 'Intl_Developed', 'Bonds', 'Gold', 'US_Equities']
    expected_returns = [0.1181, 0.0956, 0.0982, 0.0393, 0.0816, 0.1060]
    volatilities = [0.21, 0.18, 0.15, 0.089, 0.173, 0.157]
    correlation_matrix = np.array([
        [1.00, 0.70, 0.75, -0.10, -0.05, 0.65],
        [0.70, 1.00, 0.80, -0.10, -0.05, 0.60],
        [0.75, 0.80, 1.00, -0.10, -0.05, 0.75],
        [-0.10, -0.10, -0.10, 1.00, -0.30, -0.10],
        [-0.05, -0.05, -0.05, -0.30, 1.00, -0.10],
        [0.65, 0.60, 0.75, -0.10, -0.10, 1.00]
    ])
    
    returns_df = create_returns_df_from_parameters(
        asset_names=asset_names,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        n_periods=120
    )
    
    results = analyze_rolling_covariances(returns_df, window_size=36)

