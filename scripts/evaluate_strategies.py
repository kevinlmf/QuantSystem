"""
Example script demonstrating comprehensive strategy evaluation
Evaluates multiple strategies with full P&L breakdown and risk analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from evaluation import StrategyEvaluator, calculate_all_metrics, PnLAnalyzer


def generate_sample_strategy_data(
    name: str,
    n_days: int = 252,
    mean_return: float = 0.001,
    volatility: float = 0.015,
    sharpe_target: float = 2.0,
    win_rate: float = 0.55
) -> tuple:
    """Generate sample strategy returns and trades"""

    # Adjust parameters to hit target Sharpe
    adjusted_mean = sharpe_target * volatility / np.sqrt(252)

    # Generate returns with some autocorrelation (realistic for strategies)
    np.random.seed(hash(name) % 2**32)
    raw_returns = np.random.randn(n_days) * volatility

    # Add trend
    returns = raw_returns + adjusted_mean

    # Add autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]

    dates = pd.date_range(datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    returns_series = pd.Series(returns, index=dates)

    # Generate trades
    n_trades = int(n_days * 0.5)  # ~0.5 trades per day
    trade_returns = np.random.randn(n_trades) * volatility * 5

    # Adjust for target win rate
    winners = int(n_trades * win_rate)
    trade_returns[:winners] = np.abs(trade_returns[:winners])
    trade_returns[winners:] = -np.abs(trade_returns[winners:])
    np.random.shuffle(trade_returns)

    trades = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(n_trades) * 0.5),
        'quantity': np.random.randint(50, 200, n_trades),
        'pnl': trade_returns * 10000,  # Dollar P&L
        'duration': np.random.randint(1, 20, n_trades),
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA'], n_trades),
        'strategy': name
    })

    return returns_series, trades


def main():
    """Run comprehensive strategy evaluation"""

    print("=" * 80)
    print("COMPREHENSIVE STRATEGY EVALUATION DEMO")
    print("=" * 80)

    # Generate data for multiple strategies
    strategies = {
        'Market Making': {
            'mean_return': 0.0008,
            'volatility': 0.008,
            'sharpe_target': 2.5,
            'win_rate': 0.58
        },
        'Stat Arb': {
            'mean_return': 0.0006,
            'volatility': 0.010,
            'sharpe_target': 1.8,
            'win_rate': 0.52
        },
        'Momentum': {
            'mean_return': 0.0012,
            'volatility': 0.018,
            'sharpe_target': 1.5,
            'win_rate': 0.48
        },
        'Order Flow': {
            'mean_return': 0.0010,
            'volatility': 0.012,
            'sharpe_target': 2.0,
            'win_rate': 0.55
        }
    }

    # Create evaluator
    evaluator = StrategyEvaluator(
        risk_free_rate=0.02,
        periods_per_year=252,
        risk_limits={
            'max_drawdown': 0.15,
            'max_volatility': 0.25,
            'min_sharpe': 1.0,
            'max_var_95': 0.04,
            'min_win_rate': 0.45
        }
    )

    reports = []

    # Evaluate each strategy
    for strategy_name, params in strategies.items():
        print(f"\nEvaluating {strategy_name}...")

        # Generate data
        returns, trades = generate_sample_strategy_data(
            name=strategy_name,
            **params
        )

        # Evaluate
        report = evaluator.evaluate(
            strategy_name=strategy_name,
            returns=returns,
            trades=trades
        )

        reports.append(report)

        # Print individual report
        evaluator.print_evaluation_report(report, detailed=False)

    # Compare strategies
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    comparison_df = evaluator.compare_strategies(reports)
    print(comparison_df.to_string(index=False))

    # Detailed analysis of top strategy
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF TOP STRATEGY")
    print("=" * 80)

    top_strategy = reports[0]  # Already sorted by score
    evaluator.print_evaluation_report(top_strategy, detailed=True)

    # P&L breakdown for top strategy
    print("\n" + "=" * 80)
    print(f"P&L ATTRIBUTION: {top_strategy.strategy_name}")
    print("=" * 80)

    pnl_analyzer = PnLAnalyzer(
        commission_rate=0.0001,
        spread_cost=0.0001,
        market_impact_coef=0.1
    )

    # Get trades for top strategy
    top_strategy_name = top_strategy.strategy_name
    returns, trades = generate_sample_strategy_data(
        name=top_strategy_name,
        **strategies[top_strategy_name]
    )

    pnl_breakdown = pnl_analyzer.analyze_pnl(trades)
    pnl_analyzer.print_pnl_report(pnl_breakdown)

    # Risk summary
    print("\n" + "=" * 80)
    print("RISK SUMMARY")
    print("=" * 80)

    for report in reports:
        status = "✓" if all(report.risk_checks.values()) else "✗"
        passed = sum(report.risk_checks.values())
        total = len(report.risk_checks)

        print(f"\n{status} {report.strategy_name}")
        print(f"   Risk Checks: {passed}/{total} passed")
        print(f"   Max Drawdown: {report.max_drawdown:.2%}")
        print(f"   VaR (95%): {report.risk_metrics.var_95:.2%}")
        print(f"   Volatility: {report.performance.annualized_volatility:.2%}")

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Best overall
    best_overall = max(reports, key=lambda r: r.overall_score)
    print(f"\n🏆 Best Overall Strategy: {best_overall.strategy_name}")
    print(f"   Score: {best_overall.overall_score:.1f}/100")
    print(f"   Sharpe Ratio: {best_overall.sharpe_ratio:.2f}")
    print(f"   Annualized Return: {best_overall.annualized_return:.2%}")

    # Best risk-adjusted
    best_sharpe = max(reports, key=lambda r: r.sharpe_ratio)
    print(f"\n📊 Best Risk-Adjusted Returns: {best_sharpe.strategy_name}")
    print(f"   Sharpe Ratio: {best_sharpe.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {best_sharpe.performance.sortino_ratio:.2f}")

    # Lowest risk
    lowest_risk = min(reports, key=lambda r: abs(r.max_drawdown))
    print(f"\n🛡️  Lowest Risk Strategy: {lowest_risk.strategy_name}")
    print(f"   Max Drawdown: {lowest_risk.max_drawdown:.2%}")
    print(f"   Volatility: {lowest_risk.performance.annualized_volatility:.2%}")

    # Highest return
    highest_return = max(reports, key=lambda r: r.annualized_return)
    print(f"\n📈 Highest Returns: {highest_return.strategy_name}")
    print(f"   Annualized Return: {highest_return.annualized_return:.2%}")
    print(f"   Total Return: {highest_return.total_return:.2%}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    # Optional: Plot comparison
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        evaluator.plot_comparison(reports, save_path='strategy_comparison.png')
        print("\n✓ Comparison chart saved to: strategy_comparison.png")
    except Exception as e:
        print(f"\n✗ Could not generate comparison chart: {e}")


if __name__ == "__main__":
    main()
