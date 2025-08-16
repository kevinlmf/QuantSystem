# -*- coding: utf-8 -*-
"""
金融策略对比分析脚本
对比动量策略、配对交易策略和均值方差策略的各项金融指标
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# 确保能 import 到 strategy 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.momentum_strategy import MomentumStrategy
from strategy.pairs_trading import PairsTradingStrategy
from strategy.mean_variance import MeanVarianceStrategy


# =====================================================================================
# 金融指标
# =====================================================================================
class FinancialMetrics:
    """计算各种金融指标（含容错）"""

    @staticmethod
    def _safe_series(x: pd.Series | np.ndarray | list) -> pd.Series:
        s = pd.Series(x).astype(float)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return s

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """计算简单收益率（容错）"""
        prices = FinancialMetrics._safe_series(prices)
        if len(prices) < 2:
            return pd.Series([], dtype=float)
        rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return rets

    @staticmethod
    def annual_return(returns: pd.Series, trading_days: int = 252) -> float:
        """年化收益率（容错）"""
        returns = FinancialMetrics._safe_series(returns)
        if len(returns) == 0:
            return 0.0
        total_return = float((1.0 + returns).prod() - 1.0)
        years = len(returns) / float(trading_days)
        if years <= 0:
            return 0.0
        try:
            return float((1.0 + total_return) ** (1.0 / years) - 1.0)
        except Exception:
            return 0.0

    @staticmethod
    def volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """年化波动率"""
        returns = FinancialMetrics._safe_series(returns)
        if len(returns) == 0:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """夏普比率"""
        ar = FinancialMetrics.annual_return(returns, trading_days)
        vol = FinancialMetrics.volatility(returns, trading_days)
        return float((ar - risk_free_rate) / vol) if vol > 0 else 0.0

    @staticmethod
    def max_drawdown(cumulative: pd.Series) -> float:
        """最大回撤（基于净值曲线）"""
        cumulative = FinancialMetrics._safe_series(cumulative)
        if len(cumulative) == 0:
            return 0.0
        peak = cumulative.cummax()
        dd = (cumulative - peak) / peak.replace(0, np.nan)
        dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(dd.min())

    @staticmethod
    def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
        """卡玛比率 (年化收益率 / 最大回撤)"""
        ar = FinancialMetrics.annual_return(returns, trading_days)
        # calmar 用收益率的净值序列算最大回撤
        cum = (1.0 + FinancialMetrics._safe_series(returns)).cumprod()
        mdd = FinancialMetrics.max_drawdown(cum)
        return float(ar / abs(mdd)) if mdd < 0 else float("inf")

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """索提诺比率"""
        returns = FinancialMetrics._safe_series(returns)
        if len(returns) == 0:
            return 0.0
        ar = FinancialMetrics.annual_return(returns, trading_days)
        downside = returns[returns < 0]
        dvol = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 0 else 0.0
        return float((ar - risk_free_rate) / dvol) if dvol > 0 else 0.0

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """胜率"""
        returns = FinancialMetrics._safe_series(returns)
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).mean())

    @staticmethod
    def profit_loss_ratio(returns: pd.Series) -> float:
        """盈亏比"""
        returns = FinancialMetrics._safe_series(returns)
        if len(returns) == 0:
            return float("inf")
        pos = returns[returns > 0]
        neg = returns[returns < 0]
        avg_win = float(pos.mean()) if len(pos) > 0 else 0.0
        avg_loss = float(abs(neg.mean())) if len(neg) > 0 else 0.0
        return float(avg_win / avg_loss) if avg_loss > 0 else float("inf")


# =====================================================================================
# 策略模拟器（合成数据 + 三策略）
# =====================================================================================
class StrategySimulator:
    """策略模拟器"""

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = float(initial_capital)

    # ---------------- 合成数据 ----------------
    def generate_sample_data(self, symbols: List[str], days: int = 1000) -> Dict[str, pd.DataFrame]:
        """生成模拟价格数据（含 RSI/MACD/SMA 指标；用 min_periods 降低 NaN）"""
        rng = np.random.default_rng(42)
        data: Dict[str, pd.DataFrame] = {}

        start_dt = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_dt, periods=days, freq="D")

        for symbol in symbols:
            # 依据行业 标签 改变分布参数
            if "TECH" in symbol:
                mu, sigma = 0.001, 0.03
            elif "UTIL" in symbol:
                mu, sigma = 0.0003, 0.015
            else:
                mu, sigma = 0.0005, 0.02

            rets = rng.normal(mu, sigma, days)
            prices = 100.0 * np.cumprod(1.0 + rets)

            s_prices = pd.Series(prices, index=dates, dtype=float)
            sma_20 = s_prices.rolling(20, min_periods=5).mean()
            sma_50 = s_prices.rolling(50, min_periods=10).mean()
            rsi = self._calculate_rsi(s_prices, period=14)
            macd, macd_signal = self._calculate_macd(s_prices)

            df = pd.DataFrame(
                {
                    "open": s_prices * rng.normal(1.0, 0.005, days),
                    "high": s_prices * rng.normal(1.01, 0.005, days),
                    "low": s_prices * rng.normal(0.99, 0.005, days),
                    "close": s_prices,
                    "volume": rng.integers(100_000, 1_000_000, days),
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal,
                },
                index=dates,
            )
            data[symbol] = df

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI（使用 min_periods）"""
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.bfill().ffill()

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """计算MACD（使用 adjust=False 更贴近常见实现）"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    # ---------------- 策略：动量 ----------------
    def simulate_momentum_strategy(self, price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        strategy = MomentumStrategy()
        portfolio_values = [self.initial_capital]

        # 跳过前 60 天作为“热身期”
        any_df = next(iter(price_data.values()))
        dates = any_df.index[60:]
        if len(dates) == 0:
            return pd.Series([self.initial_capital], index=any_df.index[:1])

        for date in dates[::5]:  # 每 5 天再平衡
            current_data = {sym: df.loc[:date] for sym, df in price_data.items()}
            signals = strategy.generate_signals(current_data)
            positions = strategy.calculate_position_sizes(signals, portfolio_values[-1])

            daily_return = 0.0
            for sym, w in positions.items():
                df = price_data.get(sym)
                if df is None or date not in df.index:
                    continue
                sub = df.loc[:date, "close"]
                if len(sub) < 2:
                    continue
                sym_ret = sub.pct_change().iloc[-1]
                if np.isfinite(sym_ret):
                    daily_return += float(w) * float(sym_ret)

            portfolio_values.append(portfolio_values[-1] * (1.0 + daily_return))

        series = pd.Series(portfolio_values[1:], index=dates[::5], dtype=float)
        return series

    # ---------------- 策略：配对交易 ----------------
    def simulate_pairs_strategy(self, price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        strategy = PairsTradingStrategy()
        portfolio_values = [self.initial_capital]

        # 找可交易配对，无则回退基线序列（单点，后续对齐时会自动被 intersection 缩小）
        viable_pairs = strategy.find_pairs(price_data)
        any_df = next(iter(price_data.values()))
        if not viable_pairs:
            return pd.Series([self.initial_capital], index=any_df.index[:1], dtype=float)

        dates = any_df.index[300:]  # 需要更多历史
        if len(dates) == 0:
            return pd.Series([self.initial_capital], index=any_df.index[:1], dtype=float)

        for date in dates[::10]:  # 每 10 天再平衡
            # 价格截面
            current_prices = {sym: df.loc[date, "close"] for sym, df in price_data.items() if date in df.index}
            if len(current_prices) < 2:
                portfolio_values.append(portfolio_values[-1])
                continue

            # 更新配对统计并生成信号
            current_data = {sym: df.loc[:date] for sym, df in price_data.items()}
            updated_pairs = strategy.update_pair_statistics(current_data)  # {pair_id: pair_state}
            pair_states = list(updated_pairs.values()) if isinstance(updated_pairs, dict) else updated_pairs
            signals = strategy.generate_signals(pair_states, current_prices)

            # 累积当日收益
            daily_return = 0.0
            for sig in signals[:3]:  # 限制活跃配对
                positions = strategy.execute_pair_trade(sig, portfolio_values[-1])  # {symbol: weight}
                for sym, w in positions.items():
                    df = price_data.get(sym)
                    if df is None or date not in df.index:
                        continue
                    sub = df.loc[:date, "close"]
                    if len(sub) < 2:
                        continue
                    sym_ret = sub.pct_change().iloc[-1]
                    if np.isfinite(sym_ret):
                        daily_return += float(w) * float(sym_ret)

            # 降低影响（避免过度放大）
            portfolio_values.append(portfolio_values[-1] * (1.0 + daily_return * 0.1))

        series = pd.Series(portfolio_values[1:], index=dates[::10], dtype=float)
        return series

    # ---------------- 策略：均值方差 ----------------
    def simulate_mean_variance_strategy(self, price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        strategy = MeanVarianceStrategy(risk_aversion=2.0)
        portfolio_values = [self.initial_capital]

        any_df = next(iter(price_data.values()))
        dates = any_df.index[60:]
        if len(dates) == 0:
            return pd.Series([self.initial_capital], index=any_df.index[:1], dtype=float)

        for date in dates[::20]:  # 每 20 天再平衡
            price_matrix = pd.DataFrame({sym: df.loc[:date, "close"] for sym, df in price_data.items()})
            price_matrix = price_matrix.dropna(how="any")
            if len(price_matrix) < 30:
                portfolio_values.append(portfolio_values[-1])
                continue

            # 用最近 60 天计算权重
            positions = strategy.decide_position(price_matrix.tail(60), portfolio_values[-1])

            daily_return = 0.0
            total_weight = 0.0
            for sym, shares in positions.items():
                if sym not in price_data or date not in price_data[sym].index:
                    continue
                px = float(price_data[sym].loc[date, "close"])
                weight = float(shares * px / portfolio_values[-1]) if portfolio_values[-1] > 0 else 0.0
                total_weight += abs(weight)

                sub = price_data[sym].loc[:date, "close"]
                if len(sub) < 2:
                    continue
                sym_ret = sub.pct_change().iloc[-1]
                if np.isfinite(sym_ret):
                    daily_return += weight * float(sym_ret)

            if total_weight > 1.0 and total_weight > 0.0:
                daily_return /= total_weight

            portfolio_values.append(portfolio_values[-1] * (1.0 + daily_return))

        series = pd.Series(portfolio_values[1:], index=dates[::20], dtype=float)
        return series


# =====================================================================================
# 策略对比
# =====================================================================================
class StrategyComparison:
    """策略对比分析"""

    def __init__(self):
        self.simulator = StrategySimulator()
        self.metrics = FinancialMetrics()

    def _align_results(self, series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """对齐多条净值曲线，返回有共同索引的子集"""
        # 仅保留长度 >= 2 的序列
        clean = {k: v.dropna() for k, v in series_dict.items() if v is not None and len(v.dropna()) >= 2}
        if not clean:
            # 实在没有，返回原始（可能长度 1），后续会得到空的收益率
            return series_dict
        common = None
        for s in clean.values():
            common = s.index if common is None else common.intersection(s.index)
        # 仍可能为空；这时保留原始
        if common is None or len(common) == 0:
            return series_dict
        return {k: v.loc[common] for k, v in series_dict.items() if k in clean}

    def run_comparison(self, symbols: List[str] | None = None) -> Dict:
        """运行策略对比分析"""
        if symbols is None:
            symbols = ["AAPL_TECH", "MSFT_TECH", "JPM_FIN", "XOM_ENERGY", "JNJ_HEALTH", "UTIL_POWER"]

        print("生成模拟数据...")
        price_data = self.simulator.generate_sample_data(symbols, days=1000)

        print("运行策略模拟...")
        momentum = self.simulator.simulate_momentum_strategy(price_data)
        pairs = self.simulator.simulate_pairs_strategy(price_data)
        mean_var = self.simulator.simulate_mean_variance_strategy(price_data)

        # 对齐净值序列
        aligned = self._align_results(
            {"momentum": momentum, "pairs": pairs, "mean_variance": mean_var}
        )

        # 计算收益率
        out: Dict[str, Dict] = {}
        for name, curve in aligned.items():
            rets = self.metrics.calculate_returns(curve)
            cname = {"momentum": "动量策略", "pairs": "配对交易策略", "mean_variance": "均值方差策略"}[name]
            out[name] = {"returns": rets, "cumulative": curve, "name": cname}

        # 基准（等权持有）
        benchmark_prices = pd.DataFrame({sym: df["close"] for sym, df in price_data.items()})
        if aligned:
            any_curve = next(iter(aligned.values()))
            benchmark_prices = benchmark_prices.loc[any_curve.index]
        equal_weight = benchmark_prices.mean(axis=1)
        bench_rets = self.metrics.calculate_returns(equal_weight)
        out["benchmark"] = {"returns": bench_rets, "cumulative": equal_weight, "name": "等权重基准"}

        return out

    def calculate_metrics_table(self, results: Dict) -> pd.DataFrame:
        """计算金融指标对比表"""
        rows = []
        for key, data in results.items():
            returns = data["returns"]
            cumulative = data["cumulative"]

            # 总收益（容错）
            total_ret = 0.0
            cum = pd.Series(cumulative).replace([np.inf, -np.inf], np.nan).dropna()
            if len(cum) >= 2 and cum.iloc[0] != 0:
                total_ret = float(cum.iloc[-1] / cum.iloc[0] - 1.0)

            rows.append(
                {
                    "策略名称": data["name"],
                    "年化收益率": f"{self.metrics.annual_return(returns):.2%}",
                    "年化波动率": f"{self.metrics.volatility(returns):.2%}",
                    "夏普比率": f"{self.metrics.sharpe_ratio(returns):.3f}",
                    "最大回撤": f"{self.metrics.max_drawdown(cumulative):.2%}",
                    "卡玛比率": f"{self.metrics.calmar_ratio(returns):.3f}",
                    "索提诺比率": f"{self.metrics.sortino_ratio(returns):.3f}",
                    "胜率": f"{self.metrics.win_rate(returns):.2%}",
                    "盈亏比": f"{self.metrics.profit_loss_ratio(returns):.2f}",
                    "总收益率": f"{total_ret:.2%}",
                }
            )
        return pd.DataFrame(rows)

    def plot_comparison(self, results: Dict, save_path: str | None = None):
        """绘制策略对比图表（带容错）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1) 累积收益
        ax1 = axes[0, 0]
        for _, data in results.items():
            cum = pd.Series(data["cumulative"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(cum) < 2:
                continue
            normed = cum / cum.iloc[0]
            ax1.plot(normed.index, normed.values, label=data["name"], linewidth=2)
        ax1.set_title("累积收益率对比")
        ax1.set_ylabel("净值（归一化）")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) 滚动夏普
        ax2 = axes[0, 1]
        for _, data in results.items():
            rets = pd.Series(data["returns"]).dropna()
            if len(rets) < 60:
                continue
            roll = rets.rolling(60).apply(lambda x: self.metrics.sharpe_ratio(pd.Series(x)), raw=False)
            ax2.plot(roll.index, roll.values, label=data["name"], alpha=0.8)
        ax2.set_title("60日滚动夏普比率")
        ax2.set_ylabel("夏普比率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3) 回撤曲线
        ax3 = axes[1, 0]
        for _, data in results.items():
            cum = pd.Series(data["cumulative"]).dropna()
            if len(cum) < 2:
                continue
            peak = cum.cummax()
            dd = (cum - peak) / peak.replace(0, np.nan)
            dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ax3.fill_between(dd.index, dd.values, 0, alpha=0.6, label=data["name"])
        ax3.set_title("回撤曲线")
        ax3.set_ylabel("回撤")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4) 收益率分布
        ax4 = axes[1, 1]
        for _, data in results.items():
            rets = pd.Series(data["returns"]).dropna()
            if len(rets) == 0:
                continue
            ax4.hist(rets.values, bins=50, alpha=0.6, label=data["name"], density=True)
        ax4.set_title("收益率分布")
        ax4.set_xlabel("日收益率")
        ax4.set_ylabel("密度")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")
        plt.show()

    def generate_detailed_report(self, results: Dict) -> str:
        """生成详细的策略对比报告（带容错）"""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("           金融策略对比分析报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 交易日数
        try:
            n_days = max(len(v["returns"]) for v in results.values())
        except Exception:
            n_days = 0
        lines.append(f"模拟期间: {n_days} 个交易日")
        lines.append("")

        # 概述
        lines.append("策略概述:")
        lines.append("- 动量策略: 基于价格趋势和技术指标的多因子动量模型")
        lines.append("- 配对交易策略: 统计套利，利用协整/相关关系进行配对交易")
        lines.append("- 均值方差策略: 现代投资组合理论，追求最优风险收益比")
        lines.append("- 等权重基准: 简单的买入持有策略作为基准")
        lines.append("")

        # 指标表
        metrics_df = self.calculate_metrics_table(results)
        lines.append("关键金融指标对比:")
        try:
            lines.append(metrics_df.to_string(index=False))
        except Exception:
            lines.append(str(metrics_df))
        lines.append("")

        # 风险特征
        lines.append("风险特征分析:")
        for _, data in results.items():
            name = data["name"]
            rets = pd.Series(data["returns"]).dropna()
            lines.append(f"\n{name}:")
            if len(rets) == 0:
                lines.append("  - 无可用收益数据")
                continue
            lines.append(f"  - VaR (95%): {np.percentile(rets, 5):.4f}")
            lines.append(f"  - 偏度: {rets.skew():.3f}")
            lines.append(f"  - 峰度: {rets.kurtosis():.3f}")
            lines.append(f"  - 最大单日损失: {rets.min():.4f}")
            lines.append(f"  - 最大单日收益: {rets.max():.4f}")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


# =====================================================================================
# 入口
# =====================================================================================
def main():
    """主函数"""
    print("开始金融策略对比分析...")

    comparator = StrategyComparison()
    test_symbols = [
        "AAPL_TECH",
        "MSFT_TECH",
        "GOOGL_TECH",
        "JPM_FIN",
        "BAC_FIN",
        "XOM_ENERGY",
        "CVX_ENERGY",
        "JNJ_HEALTH",
        "PFE_HEALTH",
        "UTIL_POWER",
    ]

    results = comparator.run_comparison(test_symbols)

    print("\n金融指标对比表:")
    metrics_table = comparator.calculate_metrics_table(results)
    try:
        print(metrics_table.to_string(index=False))
    except Exception:
        print(metrics_table)

    detailed_report = comparator.generate_detailed_report(results)
    print("\n" + detailed_report)

    # 绘图
    try:
        print("\n生成对比图表...")
        comparator.plot_comparison(results, "strategy_comparison_charts.png")
    except Exception as e:
        print(f"图表生成失败: {e}")
        print("请确保已安装 matplotlib: pip install matplotlib")

    # 存报告
    report_file = f"strategy_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(detailed_report)
    print(f"\n详细报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
