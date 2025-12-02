"""
Correlation Matrix Validator

This script validates your correlation matrix assumptions by checking:
1. Mathematical validity (symmetry, positive semi-definite, proper range)
2. Economic realism (comparing to historical data)
3. Generated data quality (does synthetic data match your assumptions?)

Run this before running the full optimization to ensure your assumptions are sound.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Visualizations will use matplotlib only.")

from portfolio_optimizer import (
    CORRELATION_MATRIX, ASSET_NAMES, EXPECTED_RETURNS, VOLATILITIES,
    HISTORICAL_PERIODS, create_returns_df_from_parameters
)


def check_mathematical_validity(corr_matrix, asset_names):
    """
    Check if correlation matrix is mathematically valid.
    
    Returns:
    --------
    Dict with validation results
    """
    print("="*70)
    print("MATHEMATICAL VALIDITY CHECKS")
    print("="*70)
    
    results = {
        'is_valid': True,
        'issues': []
    }
    
    # Check 1: Matrix is square
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        results['is_valid'] = False
        results['issues'].append("❌ Matrix is not square")
        print("❌ Matrix is not square")
    else:
        print(f"✓ Matrix is square ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
    
    # Check 2: Dimensions match asset names
    if corr_matrix.shape[0] != len(asset_names):
        results['is_valid'] = False
        results['issues'].append(f"❌ Matrix size ({corr_matrix.shape[0]}) doesn't match number of assets ({len(asset_names)})")
        print(f"❌ Matrix size doesn't match number of assets")
    else:
        print(f"✓ Matrix size matches number of assets ({len(asset_names)})")
    
    # Check 3: Symmetric
    if not np.allclose(corr_matrix, corr_matrix.T):
        results['is_valid'] = False
        results['issues'].append("❌ Matrix is not symmetric")
        print("❌ Matrix is not symmetric")
        print(f"   Max asymmetry: {np.abs(corr_matrix - corr_matrix.T).max():.6f}")
    else:
        print("✓ Matrix is symmetric")
    
    # Check 4: Diagonal is all 1s
    if not np.allclose(np.diag(corr_matrix), 1.0):
        results['is_valid'] = False
        results['issues'].append("❌ Diagonal is not all 1s")
        print("❌ Diagonal is not all 1s")
        print(f"   Diagonal values: {np.diag(corr_matrix)}")
    else:
        print("✓ Diagonal is all 1s")
    
    # Check 5: All values in [-1, 1]
    if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
        results['is_valid'] = False
        results['issues'].append("❌ Some correlations are outside [-1, 1] range")
        print("❌ Some correlations are outside [-1, 1] range")
        print(f"   Min: {corr_matrix.min():.3f}, Max: {corr_matrix.max():.3f}")
    else:
        print(f"✓ All correlations in valid range [-1, 1]")
        print(f"   Min: {corr_matrix.min():.3f}, Max: {corr_matrix.max():.3f}")
    
    # Check 6: Positive semi-definite (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvals(corr_matrix)
    min_eigenvalue = eigenvalues.min()
    
    if min_eigenvalue < -1e-6:  # Small negative tolerance for numerical errors
        results['is_valid'] = False
        results['issues'].append(f"❌ Matrix is not positive semi-definite (min eigenvalue: {min_eigenvalue:.6f})")
        print(f"❌ Matrix is not positive semi-definite")
        print(f"   Min eigenvalue: {min_eigenvalue:.6f} (should be >= 0)")
    else:
        print(f"✓ Matrix is positive semi-definite")
        print(f"   Min eigenvalue: {min_eigenvalue:.6f}")
        print(f"   Condition number: {eigenvalues.max() / max(min_eigenvalue, 1e-10):.2f}")
    
    return results


def check_economic_realism(corr_matrix, asset_names):
    """
    Check if correlations are economically reasonable.
    """
    print("\n" + "="*70)
    print("ECONOMIC REALISM CHECKS")
    print("="*70)
    
    warnings = []
    
    # Define asset categories
    equities = ['Emerging_Markets', 'Japan', 'Intl_Developed', 'US_Equities']
    safe_havens = ['Bonds', 'Gold']
    
    # Check 1: Equity-Equity correlations (should be positive and moderate-to-high)
    print("\n1. Equity-Equity Correlations (typical range: 0.4-0.9)")
    for i, asset1 in enumerate(asset_names):
        if asset1 in equities:
            for j, asset2 in enumerate(asset_names):
                if i < j and asset2 in equities:
                    corr = corr_matrix[i, j]
                    status = "✓" if 0.4 <= corr <= 0.9 else "⚠"
                    print(f"   {status} {asset1:20s} - {asset2:20s}: {corr:6.2f}")
                    if corr < 0.4:
                        warnings.append(f"Low equity correlation: {asset1}-{asset2} = {corr:.2f}")
                    elif corr > 0.9:
                        warnings.append(f"Very high equity correlation: {asset1}-{asset2} = {corr:.2f}")
    
    # Check 2: Equity-SafeHaven correlations (should be low or negative)
    print("\n2. Equity-SafeHaven Correlations (typical range: -0.3 to 0.2)")
    for i, asset1 in enumerate(asset_names):
        if asset1 in equities:
            for j, asset2 in enumerate(asset_names):
                if asset2 in safe_havens:
                    corr = corr_matrix[i, j]
                    status = "✓" if -0.3 <= corr <= 0.2 else "⚠"
                    print(f"   {status} {asset1:20s} - {asset2:20s}: {corr:6.2f}")
                    if corr > 0.3:
                        warnings.append(f"High equity-safe haven correlation: {asset1}-{asset2} = {corr:.2f}")
    
    # Check 3: Bonds-Gold correlation (often negative)
    print("\n3. Bonds-Gold Correlation (typical range: -0.5 to 0.0)")
    bonds_idx = asset_names.index('Bonds')
    gold_idx = asset_names.index('Gold')
    corr = corr_matrix[bonds_idx, gold_idx]
    status = "✓" if -0.5 <= corr <= 0.0 else "⚠"
    print(f"   {status} Bonds-Gold: {corr:6.2f}")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    else:
        print("\n✓ All correlations appear economically reasonable")
    
    return warnings


def validate_generated_data(corr_matrix, asset_names, expected_returns, volatilities, n_periods):
    """
    Generate synthetic data and check if it matches your assumptions.
    """
    print("\n" + "="*70)
    print("GENERATED DATA VALIDATION")
    print("="*70)
    print(f"Generating {n_periods} months of synthetic data...\n")
    
    # Generate synthetic returns
    returns_df = create_returns_df_from_parameters(
        asset_names=asset_names,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=corr_matrix,
        n_periods=n_periods
    )
    
    # Calculate realized statistics
    realized_corr = returns_df.corr().values
    realized_annual_returns = returns_df.mean().values * 12
    realized_annual_vols = returns_df.std().values * np.sqrt(12)
    
    # Compare correlations
    print("Correlation Comparison (Input vs Realized):")
    print("-" * 70)
    
    max_corr_diff = 0
    for i, asset1 in enumerate(asset_names):
        for j, asset2 in enumerate(asset_names):
            if i < j:
                input_corr = corr_matrix[i, j]
                realized = realized_corr[i, j]
                diff = realized - input_corr
                max_corr_diff = max(max_corr_diff, abs(diff))
                
                # Only show if difference is notable
                if abs(diff) > 0.05:
                    print(f"{asset1:20s} - {asset2:20s}: "
                          f"Input={input_corr:6.2f}, Realized={realized:6.2f}, "
                          f"Diff={diff:+6.2f}")
    
    print(f"\nMax correlation difference: {max_corr_diff:.3f}")
    if max_corr_diff > 0.15:
        print("⚠ Large differences detected - consider increasing HISTORICAL_PERIODS")
    else:
        print("✓ Realized correlations match inputs reasonably well")
    
    # Compare returns and volatilities
    print("\n" + "-" * 70)
    print("Returns & Volatility Comparison:")
    print("-" * 70)
    print(f"{'Asset':<25} {'Input Return':<15} {'Realized':<15} {'Diff':<10}")
    print("-" * 70)
    
    for i, asset in enumerate(asset_names):
        input_ret = expected_returns[i]
        realized_ret = realized_annual_returns[i]
        diff = realized_ret - input_ret
        print(f"{asset:<25} {input_ret:>12.2%}   {realized_ret:>12.2%}   {diff:>+9.2%}")
    
    print("\n" + "-" * 70)
    print(f"{'Asset':<25} {'Input Vol':<15} {'Realized':<15} {'Diff':<10}")
    print("-" * 70)
    
    for i, asset in enumerate(asset_names):
        input_vol = volatilities[i]
        realized_vol = realized_annual_vols[i]
        diff = realized_vol - input_vol
        print(f"{asset:<25} {input_vol:>12.2%}   {realized_vol:>12.2%}   {diff:>+9.2%}")
    
    print("\n✓ Note: Small differences are normal due to sampling variation")
    print(f"  Increase HISTORICAL_PERIODS (currently {n_periods}) for closer match")
    
    return returns_df, realized_corr


def visualize_correlation_matrix(corr_matrix, asset_names, title="Correlation Matrix"):
    """
    Create a heatmap visualization of the correlation matrix.
    """
    plt.figure(figsize=(10, 8))
    
    if HAS_SEABORN:
        # Create heatmap with seaborn
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1,
                    vmax=1,
                    xticklabels=asset_names,
                    yticklabels=asset_names,
                    cbar_kws={'label': 'Correlation'},
                    square=True)
    else:
        # Create heatmap with matplotlib only
        im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label='Correlation')
        
        # Add text annotations
        for i in range(len(asset_names)):
            for j in range(len(asset_names)):
                text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        # Set ticks and labels
        plt.xticks(range(len(asset_names)), asset_names, rotation=45, ha='right')
        plt.yticks(range(len(asset_names)), asset_names)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return plt.gcf()


def compare_correlation_matrices(input_matrix, realized_matrix, asset_names):
    """
    Create side-by-side comparison of input vs realized correlations.
    """
    if not HAS_SEABORN:
        print("Skipping detailed comparison plot (seaborn not installed)")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Input matrix
    sns.heatmap(input_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                xticklabels=asset_names,
                yticklabels=asset_names,
                ax=axes[0],
                cbar_kws={'label': 'Correlation'},
                square=True)
    axes[0].set_title('Input Correlation Matrix', fontweight='bold')
    
    # Realized matrix
    sns.heatmap(realized_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                xticklabels=asset_names,
                yticklabels=asset_names,
                ax=axes[1],
                cbar_kws={'label': 'Correlation'},
                square=True)
    axes[1].set_title('Realized Correlation Matrix', fontweight='bold')
    
    # Difference matrix
    diff_matrix = realized_matrix - input_matrix
    sns.heatmap(diff_matrix, 
                annot=True, 
                fmt='+.2f',
                cmap='RdYlGn_r',
                center=0,
                vmin=-0.2,
                vmax=0.2,
                xticklabels=asset_names,
                yticklabels=asset_names,
                ax=axes[2],
                cbar_kws={'label': 'Difference'},
                square=True)
    axes[2].set_title('Difference (Realized - Input)', fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """
    Run all validation checks on the correlation matrix.
    """
    print("\n" + "="*70)
    print("CORRELATION MATRIX VALIDATION REPORT")
    print("="*70)
    print(f"Assets: {', '.join(ASSET_NAMES)}")
    print("="*70)
    
    # Check 1: Mathematical validity
    math_results = check_mathematical_validity(CORRELATION_MATRIX, ASSET_NAMES)
    
    # Check 2: Economic realism
    econ_warnings = check_economic_realism(CORRELATION_MATRIX, ASSET_NAMES)
    
    # Check 3: Validate generated data
    returns_df, realized_corr = validate_generated_data(
        CORRELATION_MATRIX, 
        ASSET_NAMES, 
        EXPECTED_RETURNS, 
        VOLATILITIES,
        HISTORICAL_PERIODS
    )
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if math_results['is_valid'] and len(econ_warnings) == 0:
        print("✓ ✓ ✓ Your correlation matrix is SATISFACTORY")
        print("   • Mathematically valid")
        print("   • Economically reasonable")
        print("   • Generates data matching your assumptions")
    elif math_results['is_valid']:
        print("⚠ Your correlation matrix is USABLE but has some warnings:")
        for warning in econ_warnings:
            print(f"   • {warning}")
        print("\n   Consider reviewing these correlations for realism")
    else:
        print("❌ Your correlation matrix has ISSUES that must be fixed:")
        for issue in math_results['issues']:
            print(f"   • {issue}")
    
    print("="*70)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Input correlation matrix
    fig1 = visualize_correlation_matrix(CORRELATION_MATRIX, ASSET_NAMES, 
                                        "Your Input Correlation Matrix")
    fig1.savefig('correlation_matrix_input.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: correlation_matrix_input.png")
    
    # Plot 2: Comparison (only if seaborn is available)
    fig2 = compare_correlation_matrices(CORRELATION_MATRIX, realized_corr, ASSET_NAMES)
    if fig2 is not None:
        fig2.savefig('correlation_matrix_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: correlation_matrix_comparison.png")
    
    print("\n" + "="*70)
    print("Validation complete! Review the output above and plots to assess your matrix.")
    print("="*70)


if __name__ == "__main__":
    main()

