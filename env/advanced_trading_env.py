"""
工业级交易环境 - 集成新数据基础设施的高级交易模拟器
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from enum import Enum

# 导入数据基础设施
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.pipeline import data_pipeline
from data.database import data_storage

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """交易动作类型"""
    HOLD = 0
    BUY = 1
    SELL = 2
    # 扩展动作类型
    BUY_MARKET = 3
    SELL_MARKET = 4
    BUY_LIMIT = 5
    SELL_LIMIT = 6

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class TradingState:
    """交易状态管理"""
    def __init__(self):
        self.cash = 0.0
        self.positions = {}  # {symbol: quantity}
        self.portfolio_value = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0

class AdvancedTradingEnv(gym.Env):
    """
    工业级交易环境
    
    特性:
    - 集成新的数据基础设施
    - 支持多资产交易
    - 真实的交易费用和滑点模拟
    - 丰富的技术指标观察空间
    - 完整的交易统计和监控
    """
    
    def __init__(self, 
                 symbols: Union[str, List[str]] = 'SPY',
                 start_date: str = None,
                 end_date: str = None,
                 initial_balance: float = 100000,
                 window_size: int = 20,
                 transaction_fee: float = 0.001,  # 0.1% 手续费
                 slippage: float = 0.0005,        # 0.05% 滑点
                 max_position_size: float = 1.0,  # 最大仓位比例
                 enable_short: bool = True,
                 lookback_days: int = 252,        # 历史数据天数
                 **kwargs):
        """
        初始化高级交易环境
        
        Args:
            symbols: 交易标的，支持单个或多个
            start_date: 开始日期
            end_date: 结束日期  
            initial_balance: 初始资金
            window_size: 观察窗口大小
            transaction_fee: 交易手续费率
            slippage: 滑点率
            max_position_size: 最大仓位比例
            enable_short: 是否允许做空
            lookback_days: 历史数据回看天数
        """
        super().__init__()
        
        # 基本参数
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.enable_short = enable_short
        self.lookback_days = lookback_days
        
        # 加载数据
        self._load_market_data()
        
        # 定义动作空间
        if len(self.symbols) == 1:
            # 单资产：Hold, Buy, Sell
            self.action_space = gym.spaces.Discrete(3)
        else:
            # 多资产：每个资产的动作 + 资产选择
            # 简化版本：使用Box空间表示每个资产的权重 [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0 if enable_short else 0.0,
                high=1.0,
                shape=(len(self.symbols),),
                dtype=np.float32
            )
        
        # 定义观察空间
        self._setup_observation_space()
        
        # 初始化状态
        self.state = TradingState()
        self.reset()
        
        logger.info(f"AdvancedTradingEnv初始化: {self.symbols}, 数据: {len(self.data)}行")
    
    def _load_market_data(self):
        """加载市场数据"""
        try:
            # 如果没有指定日期，使用最近的数据
            if self.end_date is None:
                self.end_date = datetime.now().strftime('%Y-%m-%d')
            if self.start_date is None:
                start_dt = datetime.now() - timedelta(days=self.lookback_days)
                self.start_date = start_dt.strftime('%Y-%m-%d')
            
            # 为每个symbol加载数据
            data_dict = {}
            for symbol in self.symbols:
                # 优先从数据库获取，如果没有则在线获取
                df = data_pipeline.get_data_with_fallback(
                    symbol, self.start_date, self.end_date
                )
                
                if df.empty:
                    raise ValueError(f"无法获取 {symbol} 的数据")
                
                # 确保列名标准化
                df.columns = [col.lower() for col in df.columns]
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    # 尝试从数据源列映射
                    if 'Open' in df.columns:
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                                         'Close': 'close', 'Volume': 'volume'}, inplace=True)
                
                data_dict[symbol] = df[required_cols]
            
            # 合并数据
            if len(self.symbols) == 1:
                self.data = data_dict[self.symbols[0]]
                # 添加技术指标
                self._add_technical_indicators(self.data)
            else:
                # 多资产数据对齐
                self.data = self._align_multi_asset_data(data_dict)
                
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            # 回退到示例数据
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        logger.warning("使用示例数据")
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成模拟价格数据
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.data = pd.DataFrame({
            'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        self._add_technical_indicators(self.data)
    
    def _add_technical_indicators(self, df: pd.DataFrame):
        """添加技术指标"""
        # 简单移动平均
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # 指数移动平均
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 价格相对位置
        df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / \
                              (df['high'].rolling(14).max() - df['low'].rolling(14).min())
        
        # 填充NaN值
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
    
    def _align_multi_asset_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对齐多资产数据"""
        # 找到共同的日期范围
        common_dates = None
        for symbol, df in data_dict.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # 重新构造多资产数据
        multi_data = pd.DataFrame(index=common_dates)
        
        for symbol in self.symbols:
            df = data_dict[symbol].loc[common_dates]
            self._add_technical_indicators(df)
            
            # 添加symbol前缀到列名
            for col in df.columns:
                multi_data[f"{symbol}_{col}"] = df[col]
        
        return multi_data
    
    def _setup_observation_space(self):
        """设置观察空间"""
        if len(self.data) == 0:
            # 临时设置，实际会在数据加载后更新
            n_features = 20 * len(self.symbols)
        else:
            n_features = len(self.data.columns)
        
        # 市场数据特征 + 账户状态特征
        account_features = 4  # cash_ratio, position_ratio, unrealized_pnl, drawdown
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, n_features + account_features),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置交易状态
        self.state = TradingState()
        self.state.cash = self.initial_balance
        self.state.portfolio_value = self.initial_balance
        self.state.peak_value = self.initial_balance
        
        for symbol in self.symbols:
            self.state.positions[symbol] = 0.0
        
        # 重置时间步
        self.current_step = self.window_size
        self.episode_trades = []
        self.episode_rewards = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self):
        """获取当前观察"""
        if self.current_step < self.window_size:
            # 如果步数不足，用0填充
            market_obs = np.zeros((self.window_size, len(self.data.columns)))
        else:
            # 市场数据窗口
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            market_data = self.data.iloc[start_idx:end_idx].values
            
            # 标准化处理
            market_obs = self._normalize_market_data(market_data)
        
        # 账户状态特征
        account_features = self._get_account_features()
        
        # 扩展观察空间：每个时间步都包含账户特征
        account_obs = np.tile(account_features, (self.window_size, 1))
        
        # 合并观察
        obs = np.concatenate([market_obs, account_obs], axis=1)
        
        return obs.astype(np.float32)
    
    def _normalize_market_data(self, data: np.ndarray) -> np.ndarray:
        """标准化市场数据"""
        # 简单的标准化：使用窗口内的均值和标准差
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            col_data = data[:, i]
            mean = np.mean(col_data)
            std = np.std(col_data)
            if std > 0:
                normalized[:, i] = (col_data - mean) / std
            else:
                normalized[:, i] = col_data - mean
        
        return normalized
    
    def _get_account_features(self) -> np.ndarray:
        """获取账户特征"""
        total_value = self._calculate_portfolio_value()
        
        features = [
            self.state.cash / self.initial_balance,  # 现金比例
            self._get_total_position_ratio(),        # 总仓位比例
            (total_value - self.initial_balance) / self.initial_balance,  # 未实现盈亏比例
            self.state.max_drawdown                  # 最大回撤
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_total_position_ratio(self) -> float:
        """计算总仓位比例"""
        if len(self.symbols) == 1:
            symbol = self.symbols[0]
            current_price = self._get_current_price(symbol)
            position_value = abs(self.state.positions[symbol]) * current_price
            return position_value / self.state.portfolio_value
        else:
            total_position_value = 0
            for symbol in self.symbols:
                current_price = self._get_current_price(symbol)
                position_value = abs(self.state.positions[symbol]) * current_price
                total_position_value += position_value
            return total_position_value / self.state.portfolio_value
    
    def step(self, action):
        """执行动作"""
        reward = 0.0
        info = {}
        
        # 记录步骤开始时的投资组合价值
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # 执行交易动作
        if len(self.symbols) == 1:
            reward += self._execute_single_asset_action(action)
        else:
            reward += self._execute_multi_asset_action(action)
        
        # 推进时间步
        self.current_step += 1
        
        # 计算新的投资组合价值
        current_portfolio_value = self._calculate_portfolio_value()
        self.state.portfolio_value = current_portfolio_value
        
        # 更新统计信息
        self._update_statistics(current_portfolio_value)
        
        # 计算回报奖励
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_return * 100  # 放大奖励信号
        
        # 检查是否结束
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # 风险控制：如果损失过大，提前结束
        if current_portfolio_value < self.initial_balance * 0.1:  # 损失90%
            done = True
            reward -= 10.0  # 重大损失惩罚
            info['early_stop'] = 'risk_limit'
        
        obs = self._get_observation()
        info.update(self._get_info())
        
        self.episode_rewards.append(reward)
        
        return obs, reward, done, truncated, info
    
    def _execute_single_asset_action(self, action: int) -> float:
        """执行单资产交易动作"""
        symbol = self.symbols[0]
        current_price = self._get_current_price(symbol)
        reward = 0.0
        
        if action == ActionType.BUY.value:
            reward += self._execute_buy_order(symbol, current_price)
        elif action == ActionType.SELL.value:
            reward += self._execute_sell_order(symbol, current_price)
        # HOLD动作不执行交易
        
        return reward
    
    def _execute_multi_asset_action(self, action: np.ndarray) -> float:
        """执行多资产交易动作"""
        total_reward = 0.0
        current_portfolio_value = self._calculate_portfolio_value()
        
        for i, symbol in enumerate(self.symbols):
            target_weight = np.clip(action[i], -self.max_position_size, self.max_position_size)
            current_price = self._get_current_price(symbol)
            
            # 计算目标仓位
            target_value = current_portfolio_value * target_weight
            target_shares = target_value / current_price if current_price > 0 else 0
            
            # 计算需要交易的股数
            current_shares = self.state.positions.get(symbol, 0)
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) > 0.01:  # 只有当交易量足够大时才执行
                if trade_shares > 0:
                    total_reward += self._execute_buy_order(symbol, current_price, abs(trade_shares))
                else:
                    total_reward += self._execute_sell_order(symbol, current_price, abs(trade_shares))
        
        return total_reward
    
    def _execute_buy_order(self, symbol: str, price: float, quantity: float = None) -> float:
        """执行买入订单"""
        if quantity is None:
            # 单资产模式：使用可用现金的一定比例
            available_cash = self.state.cash * 0.95  # 保留5%现金
            quantity = available_cash / (price * (1 + self.slippage + self.transaction_fee))
        
        if quantity <= 0 or self.state.cash < quantity * price * (1 + self.slippage + self.transaction_fee):
            return -0.1  # 无效交易惩罚
        
        # 计算实际成本（包含滑点和手续费）
        execution_price = price * (1 + self.slippage)
        total_cost = quantity * execution_price * (1 + self.transaction_fee)
        
        # 更新状态
        self.state.cash -= total_cost
        self.state.positions[symbol] = self.state.positions.get(symbol, 0) + quantity
        self.state.total_trades += 1
        self.state.total_fees += quantity * execution_price * self.transaction_fee
        
        # 记录交易
        trade = {
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': execution_price,
            'cost': total_cost,
            'step': self.current_step
        }
        self.episode_trades.append(trade)
        
        return -self.transaction_fee  # 交易成本作为小幅负奖励
    
    def _execute_sell_order(self, symbol: str, price: float, quantity: float = None) -> float:
        """执行卖出订单"""
        current_position = self.state.positions.get(symbol, 0)
        
        if quantity is None:
            quantity = abs(current_position)
        
        if current_position == 0 and not self.enable_short:
            return -0.1  # 无持仓时卖出的惩罚
        
        if not self.enable_short and quantity > current_position:
            quantity = current_position  # 限制卖出量
        
        if quantity <= 0:
            return -0.1
        
        # 计算实际成交价格（包含滑点和手续费）
        execution_price = price * (1 - self.slippage)
        total_proceeds = quantity * execution_price * (1 - self.transaction_fee)
        
        # 更新状态
        self.state.cash += total_proceeds
        self.state.positions[symbol] = self.state.positions.get(symbol, 0) - quantity
        self.state.total_trades += 1
        self.state.total_fees += quantity * execution_price * self.transaction_fee
        
        # 记录交易
        trade = {
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': execution_price,
            'proceeds': total_proceeds,
            'step': self.current_step
        }
        self.episode_trades.append(trade)
        
        return -self.transaction_fee  # 交易成本作为小幅负奖励
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if len(self.symbols) == 1:
            return float(self.data.iloc[self.current_step]['close'])
        else:
            return float(self.data.iloc[self.current_step][f'{symbol}_close'])
    
    def _calculate_portfolio_value(self) -> float:
        """计算投资组合总价值"""
        total_value = self.state.cash
        
        for symbol in self.symbols:
            position = self.state.positions.get(symbol, 0)
            if position != 0:
                current_price = self._get_current_price(symbol)
                total_value += position * current_price
        
        return total_value
    
    def _update_statistics(self, current_value: float):
        """更新统计信息"""
        # 更新峰值
        if current_value > self.state.peak_value:
            self.state.peak_value = current_value
        
        # 更新最大回撤
        drawdown = (self.state.peak_value - current_value) / self.state.peak_value
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
    
    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        portfolio_value = self._calculate_portfolio_value()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        info = {
            'portfolio_value': portfolio_value,
            'cash': self.state.cash,
            'positions': dict(self.state.positions),
            'total_return': total_return,
            'total_trades': self.state.total_trades,
            'total_fees': self.state.total_fees,
            'max_drawdown': self.state.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'current_step': self.current_step,
            'data_length': len(self.data)
        }
        
        return info
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.episode_rewards) < 2:
            return 0.0
        
        returns = np.array(self.episode_rewards)
        excess_returns = returns - 0.02/252  # 假设无风险利率2%
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['current_step']}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Total Return: {info['total_return']:.2%}")
            print(f"Cash: ${info['cash']:.2f}")
            print(f"Positions: {info['positions']}")
            print(f"Max Drawdown: {info['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
            print(f"Total Trades: {info['total_trades']}")
            print(f"Total Fees: ${info['total_fees']:.2f}")
            print("-" * 50)

# 便捷函数
def create_trading_env(symbol='SPY', **kwargs):
    """创建交易环境的便捷函数"""
    return AdvancedTradingEnv(symbols=symbol, **kwargs)

def create_multi_asset_env(symbols=['SPY', 'QQQ'], **kwargs):
    """创建多资产交易环境"""
    return AdvancedTradingEnv(symbols=symbols, **kwargs)