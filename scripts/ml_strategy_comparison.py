"""
Enhanced Strategy Comparison Including ML Strategies
Compares traditional strategies (Momentum, Pairs, Mean-Variance) with ML strategies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Traditional strategies
from strategy.momentum_strategy import MomentumStrategy
from strategy.pairs_trading import PairsTradingStrategy
from strategy.mean_variance import MeanVarianceStrategy

# ML strategies
try:
    from strategy.ml_traditional import RandomForestStrategy, XGBoostStrategy, LightGBMStrategy
    from strategy.dl_strategies import LSTMStrategy, TransformerStrategy
    from strategy.rl_strategies import RLTradingStrategy
    from strategy.ml_base import PredictionType
    from strategy.ml_strategy_adapter import (
        MLStrategyAdapter, TraditionalStrategyAdapter,
        StrategySimulatorML, create_ml_adapter, create_traditional_adapter
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML strategies not available: {e}")
    ML_AVAILABLE = False

from scripts.strategy_comparison import FinancialMetrics, StrategySimulator


class EnhancedStrategyComparison:
    """
    Enhanced strategy comparison including ML strategies
    """

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = initial_capital
        self.simulator = StrategySimulator(initial_capital)
        if ML_AVAILABLE:
            self.ml_simulator = StrategySimulatorML(initial_capital)
        self.metrics = FinancialMetrics()

    def run_comparison(
        self,
        symbols: List[str] | None = None,
        include_ml: bool = True,
        ml_strategies: List[str] | None = None
    ) -> Dict:
        """
        Run comprehensive strategy comparison

        Args:
            symbols: List of symbols to trade
            include_ml: Whether to include ML strategies
            ml_strategies: List of ML strategies to include
                          ['rf', 'xgb', 'lgb', 'lstm', 'transformer', 'dqn']

        Returns:
            Dictionary of results for each strategy
        """
        if symbols is None:
            symbols = [
                "AAPL_TECH", "MSFT_TECH", "GOOGL_TECH",
                "JPM_FIN", "BAC_FIN",
                "XOM_ENERGY", "CVX_ENERGY",
                "JNJ_HEALTH", "PFE_HEALTH",
                "UTIL_POWER"
            ]

        print("Generating simulation data...")
        price_data = self.simulator.generate_sample_data(symbols, days=1000)

        results = {}

        # Traditional strategies
        print("\n" + "="*80)
        print("Running Traditional Strategies...")
        print("="*80)

        print("  - Momentum Strategy")
        momentum = self.simulator.simulate_momentum_strategy(price_data)
        results['momentum'] = {
            'returns': self.metrics.calculate_returns(momentum),
            'cumulative': momentum,
            'name': 'Momentum Strategy',
            'type': 'traditional'
        }

        print("  - Pairs Trading Strategy")
        pairs = self.simulator.simulate_pairs_strategy(price_data)
        results['pairs'] = {
            'returns': self.metrics.calculate_returns(pairs),
            'cumulative': pairs,
            'name': 'Pairs Trading Strategy',
            'type': 'traditional'
        }

        print("  - Mean-Variance Strategy")
        mean_var = self.simulator.simulate_mean_variance_strategy(price_data)
        results['mean_variance'] = {
            'returns': self.metrics.calculate_returns(mean_var),
            'cumulative': mean_var,
            'name': 'Mean-Variance Strategy',
            'type': 'traditional'
        }

        # ML strategies
        if include_ml and ML_AVAILABLE:
            print("\n" + "="*80)
            print("Running ML Strategies...")
            print("="*80)

            if ml_strategies is None:
                ml_strategies = ['rf', 'xgb', 'lgb']  # Default to tree-based models

            # Random Forest
            if 'rf' in ml_strategies:
                print("  - Random Forest Strategy")
                try:
                    rf_result = self._run_ml_strategy(
                        RandomForestStrategy,
                        'Random Forest',
                        price_data
                    )
                    results['random_forest'] = rf_result
                except Exception as e:
                    print(f"    Error: {e}")

            # XGBoost
            if 'xgb' in ml_strategies:
                print("  - XGBoost Strategy")
                try:
                    xgb_result = self._run_ml_strategy(
                        XGBoostStrategy,
                        'XGBoost',
                        price_data
                    )
                    results['xgboost'] = xgb_result
                except Exception as e:
                    print(f"    Error: {e}")

            # LightGBM
            if 'lgb' in ml_strategies:
                print("  - LightGBM Strategy")
                try:
                    lgb_result = self._run_ml_strategy(
                        LightGBMStrategy,
                        'LightGBM',
                        price_data
                    )
                    results['lightgbm'] = lgb_result
                except Exception as e:
                    print(f"    Error: {e}")

            # LSTM (if requested)
            if 'lstm' in ml_strategies:
                print("  - LSTM Strategy")
                try:
                    lstm_result = self._run_ml_strategy(
                        LSTMStrategy,
                        'LSTM',
                        price_data,
                        epochs=20  # Reduced for faster training
                    )
                    results['lstm'] = lstm_result
                except Exception as e:
                    print(f"    Error: {e}")

        # Benchmark
        print("\n" + "="*80)
        print("Calculating Benchmark...")
        print("="*80)

        benchmark_prices = pd.DataFrame({sym: df["close"] for sym, df in price_data.items()})
        if results:
            any_curve = next(iter(results.values()))['cumulative']
            benchmark_prices = benchmark_prices.loc[any_curve.index]
        equal_weight = benchmark_prices.mean(axis=1)
        bench_rets = self.metrics.calculate_returns(equal_weight)

        results['benchmark'] = {
            'returns': bench_rets,
            'cumulative': equal_weight,
            'name': 'Equal Weight Benchmark',
            'type': 'benchmark'
        }

        return self._align_results(results)

    def _run_ml_strategy(
        self,
        strategy_class,
        strategy_name: str,
        price_data: Dict[str, pd.DataFrame],
        **strategy_kwargs
    ) -> Dict:
        """
        Run a single ML strategy

        Args:
            strategy_class: ML strategy class
            strategy_name: Name for display
            price_data: Price data dictionary
            **strategy_kwargs: Additional strategy parameters

        Returns:
            Result dictionary
        """
        # Create strategy
        ml_strategy = strategy_class(
            prediction_type=PredictionType.CLASSIFICATION,
            **strategy_kwargs
        )

        # Backtest with training
        portfolio_series, training_info = self.ml_simulator.backtest_ml_strategy(
            ml_strategy,
            price_data,
            train_ratio=0.6,
            val_ratio=0.2,
            rebalance_freq=10,
            warmup_period=100
        )

        return {
            'returns': self.metrics.calculate_returns(portfolio_series),
            'cumulative': portfolio_series,
            'name': strategy_name,
            'type': 'ml',
            'training_info': training_info
        }

    def _align_results(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Align all result series to common index"""
        series_dict = {k: v['cumulative'] for k, v in results.items()}

        # Only keep series with length >= 2
        clean = {k: v.dropna() for k, v in series_dict.items()
                if v is not None and len(v.dropna()) >= 2}

        if not clean:
            return results

        # Find common index
        common = None
        for s in clean.values():
            common = s.index if common is None else common.intersection(s.index)

        if common is None or len(common) == 0:
            return results

        # Align all results
        aligned_results = {}
        for key, data in results.items():
            if key in clean:
                aligned_results[key] = {
                    'returns': self.metrics.calculate_returns(data['cumulative'].loc[common]),
                    'cumulative': data['cumulative'].loc[common],
                    'name': data['name'],
                    'type': data['type']
                }
                if 'training_info' in data:
                    aligned_results[key]['training_info'] = data['training_info']

        return aligned_results

    def calculate_metrics_table(self, results: Dict) -> pd.DataFrame:
        """Calculate comprehensive metrics comparison table"""
        rows = []

        for key, data in results.items():
            returns = data["returns"]
            cumulative = data["cumulative"]

            # Total return
            total_ret = 0.0
            cum = pd.Series(cumulative).replace([np.inf, -np.inf], np.nan).dropna()
            if len(cum) >= 2 and cum.iloc[0] != 0:
                total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1.0)

            row = {
                "Strategy": data["name"],
                "Type": data["type"],
                "Annual Return": f"{self.metrics.annual_return(returns):.2%}",
                "Volatility": f"{self.metrics.volatility(returns):.2%}",
                "Sharpe Ratio": f"{self.metrics.sharpe_ratio(returns):.3f}",
                "Max Drawdown": f"{self.metrics.max_drawdown(cumulative):.2%}",
                "Calmar Ratio": f"{self.metrics.calmar_ratio(returns):.3f}",
                "Sortino Ratio": f"{self.metrics.sortino_ratio(returns):.3f}",
                "Win Rate": f"{self.metrics.win_rate(returns):.2%}",
                "Profit/Loss": f"{self.metrics.profit_loss_ratio(returns):.2f}",
                "Total Return": f"{total_ret:.2%}",
            }

            # Add training info for ML strategies
            if 'training_info' in data:
                test_metrics = data['training_info'].get('test_metrics', {})
                if 'accuracy' in test_metrics:
                    row['ML Accuracy'] = f"{test_metrics['accuracy']:.2%}"
                elif 'r2' in test_metrics:
                    row['ML R²'] = f"{test_metrics['r2']:.3f}"

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_comparison(self, results: Dict, save_path: str | None = None):
        """Plot enhanced comparison charts"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for key, data in results.items():
            cum = pd.Series(data["cumulative"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(cum) < 2:
                continue
            normed = cum / cum.iloc[0]
            linestyle = '--' if data['type'] == 'benchmark' else '-'
            linewidth = 2 if data['type'] == 'ml' else 1.5
            ax1.plot(normed.index, normed.values, label=data["name"],
                    linewidth=linewidth, linestyle=linestyle, alpha=0.8)
        ax1.set_title("Cumulative Returns Comparison", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Normalized Value")
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Risk-Return scatter
        ax2 = fig.add_subplot(gs[1, 0])
        for key, data in results.items():
            rets = data["returns"]
            ann_ret = self.metrics.annual_return(rets)
            vol = self.metrics.volatility(rets)

            color = 'green' if data['type'] == 'ml' else 'blue' if data['type'] == 'traditional' else 'gray'
            marker = 'o' if data['type'] == 'ml' else 's' if data['type'] == 'traditional' else '^'

            ax2.scatter(vol * 100, ann_ret * 100, s=100, alpha=0.6,
                       color=color, marker=marker, label=data['name'])

        ax2.set_xlabel("Volatility (%)")
        ax2.set_ylabel("Annual Return (%)")
        ax2.set_title("Risk-Return Profile", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')

        # 3. Sharpe ratios
        ax3 = fig.add_subplot(gs[1, 1])
        names = []
        sharpes = []
        colors = []

        for key, data in results.items():
            names.append(data['name'][:15])  # Truncate long names
            sharpes.append(self.metrics.sharpe_ratio(data['returns']))
            colors.append('green' if data['type'] == 'ml' else
                         'blue' if data['type'] == 'traditional' else 'gray')

        bars = ax3.barh(names, sharpes, color=colors, alpha=0.7)
        ax3.set_xlabel("Sharpe Ratio")
        ax3.set_title("Sharpe Ratio Comparison", fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Max Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        names = []
        drawdowns = []
        colors = []

        for key, data in results.items():
            names.append(data['name'][:15])
            drawdowns.append(abs(self.metrics.max_drawdown(data['cumulative'])) * 100)
            colors.append('green' if data['type'] == 'ml' else
                         'blue' if data['type'] == 'traditional' else 'gray')

        ax4.barh(names, drawdowns, color=colors, alpha=0.7)
        ax4.set_xlabel("Max Drawdown (%)")
        ax4.set_title("Maximum Drawdown", fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # 5. Rolling Sharpe
        ax5 = fig.add_subplot(gs[2, 0])
        for key, data in results.items():
            rets = pd.Series(data["returns"]).dropna()
            if len(rets) < 60:
                continue
            roll = rets.rolling(60).apply(
                lambda x: self.metrics.sharpe_ratio(pd.Series(x)), raw=False
            )
            linestyle = '--' if data['type'] == 'benchmark' else '-'
            ax5.plot(roll.index, roll.values, label=data["name"],
                    alpha=0.7, linestyle=linestyle)

        ax5.set_ylabel("Sharpe Ratio")
        ax5.set_title("60-Day Rolling Sharpe Ratio", fontweight='bold')
        ax5.legend(fontsize=8, loc='best')
        ax5.grid(True, alpha=0.3)

        # 6. Drawdown curves
        ax6 = fig.add_subplot(gs[2, 1])
        for key, data in results.items():
            cum = pd.Series(data["cumulative"]).dropna()
            if len(cum) < 2:
                continue
            peak = cum.cummax()
            dd = (cum - peak) / peak.replace(0, np.nan)
            dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            linestyle = '--' if data['type'] == 'benchmark' else '-'
            ax6.fill_between(dd.index, dd.values * 100, 0, alpha=0.4,
                           label=data["name"])

        ax6.set_ylabel("Drawdown (%)")
        ax6.set_title("Drawdown Curves", fontweight='bold')
        ax6.legend(fontsize=8, loc='best')
        ax6.grid(True, alpha=0.3)

        # 7. Return distribution comparison
        ax7 = fig.add_subplot(gs[2, 2])
        for key, data in results.items():
            rets = pd.Series(data["returns"]).dropna()
            if len(rets) == 0:
                continue
            ax7.hist(rets.values * 100, bins=50, alpha=0.5,
                    label=data["name"], density=True)

        ax7.set_xlabel("Daily Return (%)")
        ax7.set_ylabel("Density")
        ax7.set_title("Return Distribution", fontweight='bold')
        ax7.legend(fontsize=8, loc='best')
        ax7.grid(True, alpha=0.3)

        plt.suptitle("Enhanced Strategy Comparison: Traditional vs ML Strategies",
                    fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to: {save_path}")

        plt.show()

    def generate_detailed_report(self, results: Dict) -> str:
        """Generate detailed comparison report"""
        lines = []
        lines.append("=" * 100)
        lines.append("           ENHANCED STRATEGY COMPARISON REPORT")
        lines.append("           Traditional vs Machine Learning Strategies")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Strategy overview
        lines.append("STRATEGY OVERVIEW:")
        lines.append("")

        traditional_count = sum(1 for v in results.values() if v['type'] == 'traditional')
        ml_count = sum(1 for v in results.values() if v['type'] == 'ml')

        lines.append(f"Total Strategies Compared: {len(results)}")
        lines.append(f"  - Traditional Strategies: {traditional_count}")
        lines.append(f"  - ML Strategies: {ml_count}")
        lines.append(f"  - Benchmark: 1")
        lines.append("")

        # Metrics table
        metrics_df = self.calculate_metrics_table(results)
        lines.append("PERFORMANCE METRICS:")
        lines.append("")
        lines.append(metrics_df.to_string(index=False))
        lines.append("")

        # Best performers
        lines.append("TOP PERFORMERS:")
        lines.append("")

        # Best Sharpe
        best_sharpe = max(
            results.items(),
            key=lambda x: self.metrics.sharpe_ratio(x[1]['returns'])
        )
        lines.append(f"  Best Sharpe Ratio: {best_sharpe[1]['name']} "
                    f"({self.metrics.sharpe_ratio(best_sharpe[1]['returns']):.3f})")

        # Best Return
        best_return = max(
            results.items(),
            key=lambda x: self.metrics.annual_return(x[1]['returns'])
        )
        lines.append(f"  Best Annual Return: {best_return[1]['name']} "
                    f"({self.metrics.annual_return(best_return[1]['returns']):.2%})")

        # Lowest Drawdown
        best_dd = min(
            results.items(),
            key=lambda x: abs(self.metrics.max_drawdown(x[1]['cumulative']))
        )
        lines.append(f"  Lowest Drawdown: {best_dd[1]['name']} "
                    f"({self.metrics.max_drawdown(best_dd[1]['cumulative']):.2%})")

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)


def main():
    """Main function"""
    print("Starting Enhanced Strategy Comparison...")
    print("="*80)

    comparator = EnhancedStrategyComparison()

    # Run comparison with ML strategies
    results = comparator.run_comparison(
        include_ml=True,
        ml_strategies=['rf', 'xgb', 'lgb']  # Can add 'lstm', 'transformer', 'dqn'
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
        comparator.plot_comparison(results, "ml_strategy_comparison_charts.png")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Save report
    report_file = f"ml_strategy_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(detailed_report)
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
