"""
Industrial-gradeTrading Environment - Advanced Trading Simulator with integrated data infrastructure
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from enum import Enum

# Import data infrastructure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.pipeline import data_pipeline
from data.database import data_storage

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Trade action types"""
    HOLD = 0
    BUY = 1
    SELL = 2
    # Extended action types
    BUY_MARKET = 3
    SELL_MARKET = 4
    BUY_LIMIT = 5
    SELL_LIMIT = 6

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class TradingState:
    """TradeState管理"""
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
    Industrial-gradeTrading Environment
    
    Features:
    - 集成Newdata基础设施
    - 支持多AssetTrade
    - Real的Trade费用和SlippageSimulation
    - 丰富的Technical indicatorsObservation空间
    - Complete的Trade统计和监控
    """
    
    def __init__(self, 
                 symbols: Union[str, List[str]] = 'SPY',
                 start_date: str = None,
                 end_date: str = None,
                 initial_balance: float = 100000,
                 window_size: int = 20,
                 transaction_fee: float = 0.001,  # 0.1% Fee
                 slippage: float = 0.0005,        # 0.05% Slippage
                 max_position_size: float = 1.0,  # MaximumPositionProportion
                 enable_short: bool = True,
                 lookback_days: int = 252,        # Historicaldata天数
                 **kwargs):
        """
        initializeAdvancedTrading Environment
        
        Args:
            symbols: Trade标的，支持单个或多个
            start_date: Start date
            end_date: End date  
            initial_balance: 初始Capital
            window_size: Observation窗口大小
            transaction_fee: TradeFee率
            slippage: Slippage率
            max_position_size: MaximumPositionProportion
            enable_short: Is否允许Short
            lookback_days: Historicaldata回看天数
        """
        super().__init__()
        
        # 基本parameter
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
        
        # Loaddata
        self._load_market_data()
        
        # 定义Action空间
        if len(self.symbols) == 1:
            # 单Asset：Hold, Buy, Sell
            self.action_space = gym.spaces.Discrete(3)
        else:
            # 多Asset：每个Asset的Action + Asset选择
            # 简化版本：UsingBox空间表示每个Asset的Weight [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0 if enable_short else 0.0,
                high=1.0,
                shape=(len(self.symbols),),
                dtype=np.float32
            )
        
        # 定义Observation空间
        self._setup_observation_space()
        
        # initializeState
        self.state = TradingState()
        self.reset()
        
        logger.info(f"AdvancedTradingEnvinitialize: {self.symbols}, data: {len(self.data)}行")
    
    def _load_market_data(self):
        """Load市场data"""
        try:
            # If没有指定Date，Using最近的data
            if self.end_date is None:
                self.end_date = datetime.now().strftime('%Y-%m-%d')
            if self.start_date is None:
                start_dt = datetime.now() - timedelta(days=self.lookback_days)
                self.start_date = start_dt.strftime('%Y-%m-%d')
            
            # 为每个symbolLoaddata
            data_dict = {}
            for symbol in self.symbols:
                # 优先从data库Get，If没有则在线Get
                df = data_pipeline.get_data_with_fallback(
                    symbol, self.start_date, self.end_date
                )
                
                if df.empty:
                    raise ValueError(f"无法Get {symbol} 的data")
                
                # 确保列名Normalization
                df.columns = [col.lower() for col in df.columns]
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    # 尝试从data源列Map
                    if 'Open' in df.columns:
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 
                                         'Close': 'close', 'Volume': 'volume'}, inplace=True)
                
                data_dict[symbol] = df[required_cols]
            
            # Mergedata
            if len(self.symbols) == 1:
                self.data = data_dict[self.symbols[0]]
                # AddTechnical indicators
                self._add_technical_indicators(self.data)
            else:
                # 多AssetdataAlign
                self.data = self._align_multi_asset_data(data_dict)
                
        except Exception as e:
            logger.error(f"dataLoadfailure: {e}")
            # 回退到Exampledata
            self._create_sample_data()
    
    def _create_sample_data(self):
        """CreateExampledata"""
        logger.warning("UsingExampledata")
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # GenerateSimulationPricedata
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
        """AddTechnical indicators"""
        # SimpleMoving average
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # 指数Moving average
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
        
        # Fill量指标
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price相对位置
        df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / \
                              (df['high'].rolling(14).max() - df['low'].rolling(14).min())
        
        # 填充NaN值
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
    
    def _align_multi_asset_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align多Assetdata"""
        # 找到共同的Date范围
        common_dates = None
        for symbol, df in data_dict.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # 重新构造多Assetdata
        multi_data = pd.DataFrame(index=common_dates)
        
        for symbol in self.symbols:
            df = data_dict[symbol].loc[common_dates]
            self._add_technical_indicators(df)
            
            # Addsymbol前缀到列名
            for col in df.columns:
                multi_data[f"{symbol}_{col}"] = df[col]
        
        return multi_data
    
    def _setup_observation_space(self):
        """SettingsObservation空间"""
        if len(self.data) == 0:
            # 临时Settings，实际会在dataLoad后update
            n_features = 20 * len(self.symbols)
        else:
            n_features = len(self.data.columns)
        
        # 市场data特征 + AccountState特征
        account_features = 4  # cash_ratio, position_ratio, unrealized_pnl, drawdown
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, n_features + account_features),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """resetEnvironment"""
        super().reset(seed=seed)
        
        # resetTradeState
        self.state = TradingState()
        self.state.cash = self.initial_balance
        self.state.portfolio_value = self.initial_balance
        self.state.peak_value = self.initial_balance
        
        for symbol in self.symbols:
            self.state.positions[symbol] = 0.0
        
        # resetTime步
        self.current_step = self.window_size
        self.episode_trades = []
        self.episode_rewards = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self):
        """GetCurrentObservation"""
        if self.current_step < self.window_size:
            # If步数不足，用0填充
            market_obs = np.zeros((self.window_size, len(self.data.columns)))
        else:
            # 市场data窗口
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            market_data = self.data.iloc[start_idx:end_idx].values
            
            # NormalizationProcess
            market_obs = self._normalize_market_data(market_data)
        
        # AccountState特征
        account_features = self._get_account_features()
        
        # 扩展Observation空间：每个Time步都包含Account特征
        account_obs = np.tile(account_features, (self.window_size, 1))
        
        # MergeObservation
        obs = np.concatenate([market_obs, account_obs], axis=1)
        
        return obs.astype(np.float32)
    
    def _normalize_market_data(self, data: np.ndarray) -> np.ndarray:
        """Normalization市场data"""
        # Simple的Normalization：Using窗口内的均值和标准差
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
        """GetAccount特征"""
        total_value = self._calculate_portfolio_value()
        
        features = [
            self.state.cash / self.initial_balance,  # CashProportion
            self._get_total_position_ratio(),        # 总PositionProportion
            (total_value - self.initial_balance) / self.initial_balance,  # 未实现Profit/Loss Ratio例
            self.state.max_drawdown                  # max drawdown
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_total_position_ratio(self) -> float:
        """calculate总PositionProportion"""
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
        """ExecuteAction"""
        reward = 0.0
        info = {}
        
        # 记录StepStart时的Portfolio价值
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # ExecuteTradeAction
        if len(self.symbols) == 1:
            reward += self._execute_single_asset_action(action)
        else:
            reward += self._execute_multi_asset_action(action)
        
        # 推进Time步
        self.current_step += 1
        
        # calculateNewPortfolio价值
        current_portfolio_value = self._calculate_portfolio_value()
        self.state.portfolio_value = current_portfolio_value
        
        # update统计Info
        self._update_statistics(current_portfolio_value)
        
        # calculatereturnReward
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += portfolio_return * 100  # 放大Reward信号
        
        # checkIs否Done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # risk控制：IfLoss过大，提前Done
        if current_portfolio_value < self.initial_balance * 0.1:  # Loss90%
            done = True
            reward -= 10.0  # 重大Loss惩罚
            info['early_stop'] = 'risk_limit'
        
        obs = self._get_observation()
        info.update(self._get_info())
        
        self.episode_rewards.append(reward)
        
        return obs, reward, done, truncated, info
    
    def _execute_single_asset_action(self, action: int) -> float:
        """Execute单AssetTradeAction"""
        symbol = self.symbols[0]
        current_price = self._get_current_price(symbol)
        reward = 0.0
        
        if action == ActionType.BUY.value:
            reward += self._execute_buy_order(symbol, current_price)
        elif action == ActionType.SELL.value:
            reward += self._execute_sell_order(symbol, current_price)
        # HOLDAction不ExecuteTrade
        
        return reward
    
    def _execute_multi_asset_action(self, action: np.ndarray) -> float:
        """Execute多AssetTradeAction"""
        total_reward = 0.0
        current_portfolio_value = self._calculate_portfolio_value()
        
        for i, symbol in enumerate(self.symbols):
            target_weight = np.clip(action[i], -self.max_position_size, self.max_position_size)
            current_price = self._get_current_price(symbol)
            
            # calculate目标Position
            target_value = current_portfolio_value * target_weight
            target_shares = target_value / current_price if current_price > 0 else 0
            
            # calculate需要Trade的Shares
            current_shares = self.state.positions.get(symbol, 0)
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) > 0.01:  # 只有WhenTrade量足够大时才Execute
                if trade_shares > 0:
                    total_reward += self._execute_buy_order(symbol, current_price, abs(trade_shares))
                else:
                    total_reward += self._execute_sell_order(symbol, current_price, abs(trade_shares))
        
        return total_reward
    
    def _execute_buy_order(self, symbol: str, price: float, quantity: float = None) -> float:
        """ExecuteBuyOrder"""
        if quantity is None:
            # 单Asset模式：UsingAvailableCash的一定Proportion
            available_cash = self.state.cash * 0.95  # 保留5%Cash
            quantity = available_cash / (price * (1 + self.slippage + self.transaction_fee))
        
        if quantity <= 0 or self.state.cash < quantity * price * (1 + self.slippage + self.transaction_fee):
            return -0.1  # InvalidTrade惩罚
        
        # calculate实际Cost（包含Slippage和Fee）
        execution_price = price * (1 + self.slippage)
        total_cost = quantity * execution_price * (1 + self.transaction_fee)
        
        # updateState
        self.state.cash -= total_cost
        self.state.positions[symbol] = self.state.positions.get(symbol, 0) + quantity
        self.state.total_trades += 1
        self.state.total_fees += quantity * execution_price * self.transaction_fee
        
        # 记录Trade
        trade = {
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': execution_price,
            'cost': total_cost,
            'step': self.current_step
        }
        self.episode_trades.append(trade)
        
        return -self.transaction_fee  # TradeCost作为小幅负Reward
    
    def _execute_sell_order(self, symbol: str, price: float, quantity: float = None) -> float:
        """ExecuteSellOrder"""
        current_position = self.state.positions.get(symbol, 0)
        
        if quantity is None:
            quantity = abs(current_position)
        
        if current_position == 0 and not self.enable_short:
            return -0.1  # 无持仓时Sell的惩罚
        
        if not self.enable_short and quantity > current_position:
            quantity = current_position  # 限制Sell量
        
        if quantity <= 0:
            return -0.1
        
        # calculate实际FillPrice（包含Slippage和Fee）
        execution_price = price * (1 - self.slippage)
        total_proceeds = quantity * execution_price * (1 - self.transaction_fee)
        
        # updateState
        self.state.cash += total_proceeds
        self.state.positions[symbol] = self.state.positions.get(symbol, 0) - quantity
        self.state.total_trades += 1
        self.state.total_fees += quantity * execution_price * self.transaction_fee
        
        # 记录Trade
        trade = {
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': execution_price,
            'proceeds': total_proceeds,
            'step': self.current_step
        }
        self.episode_trades.append(trade)
        
        return -self.transaction_fee  # TradeCost作为小幅负Reward
    
    def _get_current_price(self, symbol: str) -> float:
        """GetCurrentPrice"""
        if len(self.symbols) == 1:
            return float(self.data.iloc[self.current_step]['close'])
        else:
            return float(self.data.iloc[self.current_step][f'{symbol}_close'])
    
    def _calculate_portfolio_value(self) -> float:
        """calculatePortfolio总价值"""
        total_value = self.state.cash
        
        for symbol in self.symbols:
            position = self.state.positions.get(symbol, 0)
            if position != 0:
                current_price = self._get_current_price(symbol)
                total_value += position * current_price
        
        return total_value
    
    def _update_statistics(self, current_value: float):
        """update统计Info"""
        # update峰值
        if current_value > self.state.peak_value:
            self.state.peak_value = current_value
        
        # updatemax drawdown
        drawdown = (self.state.peak_value - current_value) / self.state.peak_value
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
    
    def _get_info(self) -> Dict[str, Any]:
        """GetEnvironmentInfo"""
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
        """calculateSharpe Ratio"""
        if len(self.episode_rewards) < 2:
            return 0.0
        
        returns = np.array(self.episode_rewards)
        excess_returns = returns - 0.02/252  # 假设无risk利率2%
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def render(self, mode='human'):
        """渲染EnvironmentState"""
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

# 便捷function
def create_trading_env(symbol='SPY', **kwargs):
    """CreateTrading Environment的便捷function"""
    return AdvancedTradingEnv(symbols=symbol, **kwargs)

def create_multi_asset_env(symbols=['SPY', 'QQQ'], **kwargs):
    """Create多AssetTrading Environment"""
    return AdvancedTradingEnv(symbols=symbols, **kwargs)