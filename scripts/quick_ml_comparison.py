"""
Quick ML Strategy Comparison - Faster version with reduced parameters
"""
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ml_strategy_comparison import EnhancedStrategyComparison

def main():
    """Quick comparison with reduced complexity"""
    print("Starting Quick ML Strategy Comparison...")
    print("(Using reduced parameters for faster execution)")
    print("="*80)

    comparator = EnhancedStrategyComparison()

    # Reduced symbol list and parameters for speed
    symbols = [
        "AAPL_TECH", "MSFT_TECH",
        "JPM_FIN", "BAC_FIN"
    ]

    # Run comparison with only Random Forest (fastest)
    results = comparator.run_comparison(
        symbols=symbols,
        include_ml=True,
        ml_strategies=['rf']  # Only Random Forest
    )

    # Print metrics table
    print("\n" + "="*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*80)
    metrics_table = comparator.calculate_metrics_table(results)
    print(metrics_table.to_string(index=False))

    # Generate detailed report
    detailed_report = comparator.generate_detailed_report(results)
    print("\n" + detailed_report)

    # Plot comparison
    try:
        print("\nGenerating comparison charts...")
        comparator.plot_comparison(results, "quick_ml_comparison.png")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Save report
    from datetime import datetime
    report_file = f"quick_ml_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(detailed_report)
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    main()
