"""
Trading EnvironmentPerformance monitoring module
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)

class TradingMetrics:
    """Trade指标calculate器"""
    
    @staticmethod
    def calculate_returns(values: List[float]) -> Dict[str, float]:
        """calculatereturn率指标"""
        if len(values) < 2:
            return {}
        
        values = np.array(values)
        returns = np.diff(values) / values[:-1]
        
        return {
            'total_return': (values[-1] - values[0]) / values[0],
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns)
        }
    
    @staticmethod
    def calculate_drawdown(values: List[float]) -> Dict[str, float]:
        """calculateDrawdown指标"""
        if len(values) < 2:
            return {'max_drawdown': 0, 'current_drawdown': 0}
        
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        return {
            'max_drawdown': abs(np.min(drawdown)),
            'current_drawdown': abs(drawdown[-1]),
            'drawdown_duration': len(values) - np.argmax(peak == values[-1]) if values[-1] < peak[-1] else 0
        }
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> Dict[str, float]:
        """calculateWin Rate指标"""
        if not trades:
            return {}
        
        profits = []
        for trade in trades:
            # 简化的P&Lcalculate
            if 'profit' in trade:
                profits.append(trade['profit'])
            elif 'proceeds' in trade and 'cost' in trade:
                profits.append(trade['proceeds'] - trade['cost'])
        
        if not profits:
            return {}
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        return {
            'win_rate': len(winning_trades) / len(profits),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'total_trades': len(profits),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

class PerformanceMonitor:
    """Performance monitor"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """reset监控器"""
        self.episode_start_time = time.time()
        self.step_times = deque(maxlen=self.max_history)
        self.portfolio_values = deque(maxlen=self.max_history)
        self.rewards = deque(maxlen=self.max_history)
        self.actions = deque(maxlen=self.max_history)
        self.trades = []
        self.step_count = 0
        self.episode_count = 0
        
    def log_step(self, action: Any, reward: float, portfolio_value: float, 
                info: Dict[str, Any]):
        """记录StepInfo"""
        step_start = time.time()
        
        self.step_count += 1
        self.actions.append(action)
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)
        
        # 记录Trade
        if 'trades' in info and info['trades']:
            self.trades.extend(info['trades'])
        
        step_end = time.time()
        self.step_times.append(step_end - step_start)
    
    def log_episode_end(self):
        """记录episodeDone"""
        self.episode_count += 1
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get性能统计"""
        if not self.portfolio_values:
            return {}
        
        # 基本统计
        current_time = time.time()
        episode_duration = current_time - self.episode_start_time
        
        stats = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'episode_duration': episode_duration,
            'steps_per_second': self.step_count / episode_duration if episode_duration > 0 else 0,
            'avg_step_time': np.mean(self.step_times) if self.step_times else 0
        }
        
        # return率指标
        portfolio_values = list(self.portfolio_values)
        stats.update(TradingMetrics.calculate_returns(portfolio_values))
        stats.update(TradingMetrics.calculate_drawdown(portfolio_values))
        
        # Trade指标
        stats.update(TradingMetrics.calculate_win_rate(self.trades))
        
        # Action分布
        if self.actions:
            action_counts = defaultdict(int)
            for action in self.actions:
                if isinstance(action, (int, np.integer)):
                    action_counts[int(action)] += 1
                else:
                    action_counts['continuous'] += 1
            
            stats['action_distribution'] = dict(action_counts)
        
        # Reward统计
        if self.rewards:
            stats['reward_stats'] = {
                'mean_reward': np.mean(self.rewards),
                'std_reward': np.std(self.rewards),
                'total_reward': sum(self.rewards),
                'positive_rewards': sum(1 for r in self.rewards if r > 0),
                'negative_rewards': sum(1 for r in self.rewards if r < 0)
            }
        
        return stats
    
    def print_performance_summary(self):
        """Print性能Summary"""
        stats = self.get_performance_stats()
        
        print("=" * 60)
        print("Trading Environment性能Summary")
        print("=" * 60)
        
        if not stats:
            print("暂无性能data")
            return
        
        # 基本Info
        print(f"EpisodeCount: {stats.get('episode_count', 0)}")
        print(f"总步数: {stats.get('step_count', 0)}")
        print(f"Run时长: {stats.get('episode_duration', 0):.2f}秒")
        print(f"步数/秒: {stats.get('steps_per_second', 0):.2f}")
        print(f"AverageStep用时: {stats.get('avg_step_time', 0)*1000:.2f}ms")
        
        # return指标
        print(f"\n📈 return指标:")
        print(f"总return率: {stats.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"max drawdown: {stats.get('max_drawdown', 0):.2%}")
        print(f"Current Drawdown: {stats.get('current_drawdown', 0):.2%}")
        
        # Trade指标
        if stats.get('total_trades', 0) > 0:
            print(f"\n📊 Trade指标:")
            print(f"总Trade次数: {stats.get('total_trades', 0)}")
            print(f"Win Rate: {stats.get('win_rate', 0):.2%}")
            print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
            print(f"Average盈利: {stats.get('avg_win', 0):.2f}")
            print(f"Average亏损: {stats.get('avg_loss', 0):.2f}")
        
        # Action分布
        if 'action_distribution' in stats:
            print(f"\n🎯 Action分布:")
            for action, count in stats['action_distribution'].items():
                percentage = count / stats['step_count'] * 100
                print(f"  Action {action}: {count}次 ({percentage:.1f}%)")
        
        # Reward统计
        if 'reward_stats' in stats:
            reward_stats = stats['reward_stats']
            print(f"\n🏆 Reward统计:")
            print(f"总Reward: {reward_stats.get('total_reward', 0):.2f}")
            print(f"AverageReward: {reward_stats.get('mean_reward', 0):.4f}")
            print(f"Reward标准差: {reward_stats.get('std_reward', 0):.4f}")
            print(f"正Reward次数: {reward_stats.get('positive_rewards', 0)}")
            print(f"负Reward次数: {reward_stats.get('negative_rewards', 0)}")

class EnvironmentBenchmark:
    """Environment benchmark"""
    
    def __init__(self, env_factory, test_configs: List[Dict]):
        """
        Args:
            env_factory: Environment工厂function
            test_configs: TestConfiguration列表
        """
        self.env_factory = env_factory
        self.test_configs = test_configs
        self.benchmark_results = []
    
    def run_benchmark(self, num_episodes: int = 10, max_steps: int = 1000):
        """Run基准Test"""
        logger.info(f"StartEnvironment benchmark: {len(self.test_configs)}个Configuration")
        
        for i, config in enumerate(self.test_configs):
            logger.info(f"TestConfiguration {i+1}/{len(self.test_configs)}: {config}")
            
            # CreateEnvironment
            env = self.env_factory(**config)
            monitor = PerformanceMonitor()
            
            # Run多个episode
            episode_results = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                monitor.reset()
                
                for step in range(max_steps):
                    # 随机Action（基准Test用）
                    action = env.action_space.sample()
                    
                    obs, reward, done, truncated, info = env.step(action)
                    
                    portfolio_value = info.get('portfolio_value', 0)
                    monitor.log_step(action, reward, portfolio_value, info)
                    
                    if done or truncated:
                        break
                
                monitor.log_episode_end()
                episode_stats = monitor.get_performance_stats()
                episode_results.append(episode_stats)
            
            # 汇总Results
            config_result = {
                'config': config,
                'episodes': episode_results,
                'summary': self._summarize_episodes(episode_results)
            }
            
            self.benchmark_results.append(config_result)
            
            env.close()
            logger.info(f"Configuration {i+1} TestCompleted")
        
        return self.benchmark_results
    
    def _summarize_episodes(self, episode_results: List[Dict]) -> Dict[str, float]:
        """汇总episodeResults"""
        if not episode_results:
            return {}
        
        # 提取关键指标
        total_returns = [r.get('total_return', 0) for r in episode_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in episode_results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in episode_results]
        steps_per_second = [r.get('steps_per_second', 0) for r in episode_results]
        
        return {
            'avg_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_steps_per_second': np.mean(steps_per_second),
            'episodes_completed': len(episode_results)
        }
    
    def save_benchmark_results(self, filename: str):
        """Save基准TestResults"""
        with open(filename, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
        logger.info(f"基准TestResultsSaved到: {filename}")
    
    def print_benchmark_summary(self):
        """Print基准TestSummary"""
        print("\n" + "="*80)
        print("Environment benchmarkResultsSummary")
        print("="*80)
        
        for i, result in enumerate(self.benchmark_results):
            config = result['config']
            summary = result['summary']
            
            print(f"\nConfiguration {i+1}: {config}")
            print(f"  Average总return率: {summary.get('avg_total_return', 0):.2%}")
            print(f"  AverageSharpe Ratio: {summary.get('avg_sharpe_ratio', 0):.2f}")
            print(f"  Averagemax drawdown: {summary.get('avg_max_drawdown', 0):.2%}")
            print(f"  AverageExecute速度: {summary.get('avg_steps_per_second', 0):.1f} 步/秒")
            print(f"  Completedepisode数: {summary.get('episodes_completed', 0)}")

# 便捷function
def create_env_monitor(env):
    """为EnvironmentCreate监控器"""
    monitor = PerformanceMonitor()
    
    # 包装Environment的stepmethod
    original_step = env.step
    
    def monitored_step(action):
        result = original_step(action)
        obs, reward, done, truncated, info = result
        
        portfolio_value = info.get('portfolio_value', getattr(env.state, 'portfolio_value', 0))
        monitor.log_step(action, reward, portfolio_value, info)
        
        if done or truncated:
            monitor.log_episode_end()
        
        return result
    
    env.step = monitored_step
    env.monitor = monitor
    
    return env