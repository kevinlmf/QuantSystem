"""
交易环境模块

提供工业级的交易环境，支持：
- 单资产和多资产交易
- 真实的交易费用和滑点模拟
- 丰富的技术指标
- 性能监控和基准测试
- 集成数据基础设施
"""

from .trading_env import TradingEnv  # 原始简单环境
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
    # 环境类
    'TradingEnv',           # 原始简单环境
    'AdvancedTradingEnv',   # 高级工业环境
    'ActionType',
    'OrderType', 
    'TradingState',
    
    # 环境创建函数
    'create_trading_env',
    'create_multi_asset_env',
    
    # 数据加载
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

# 版本信息
__version__ = "2.0.0"

# 便捷别名
TradingEnvironment = AdvancedTradingEnv
create_env = create_trading_env