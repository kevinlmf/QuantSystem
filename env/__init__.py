"""
Trading Environmentmodule

提供Industrial-grade的Trading Environment，支持：
- 单Asset和多AssetTrade
- Real的Trade费用和SlippageSimulation
- 丰富的Technical indicators
- 性能监控和基准Test
- 集成data基础设施
"""

from .trading_env import TradingEnv  # OriginalSimpleEnvironment
from .advanced_trading_env import (
    AdvancedTradingEnv,
    ActionType,
    OrderType,
    TradingState,
    create_trading_env,
    create_multi_asset_env
)
from .data_loader import (
    load_csv_data,
    load_market_data,
    load_multiple_symbols,
    get_data_info,
    format_for_trading_env,
    load_data_for_env
)
from .env_monitor import (
    TradingMetrics,
    PerformanceMonitor,
    EnvironmentBenchmark,
    create_env_monitor
)

# 主要接口
__all__ = [
    # Environmentclass
    'TradingEnv',           # OriginalSimpleEnvironment
    'AdvancedTradingEnv',   # Advanced工业Environment
    'ActionType',
    'OrderType', 
    'TradingState',
    
    # EnvironmentCreatefunction
    'create_trading_env',
    'create_multi_asset_env',
    
    # dataLoad
    'load_csv_data',
    'load_market_data',
    'load_multiple_symbols',
    'get_data_info',
    'format_for_trading_env',
    'load_data_for_env',
    
    # 监控工具
    'TradingMetrics',
    'PerformanceMonitor',
    'EnvironmentBenchmark',
    'create_env_monitor'
]

# 版本Info
__version__ = "2.0.0"

# 便捷别名
TradingEnvironment = AdvancedTradingEnv
create_env = create_trading_env