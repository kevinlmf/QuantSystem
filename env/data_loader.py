"""
数据加载器 - 集成新数据基础设施的统一数据接口
"""
import pandas as pd
import os
import sys
from typing import Union, Optional, Dict, Any
import logging

# 导入数据基础设施
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from data.pipeline import data_pipeline
    from data.database import data_storage
    DATA_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    DATA_INFRASTRUCTURE_AVAILABLE = False
    logging.warning("数据基础设施未找到，将使用传统CSV加载方法")

logger = logging.getLogger(__name__)

def load_csv_data(path):
    """
    Load market data from a CSV file and preprocess it for TradingEnv.
    (保持向后兼容性)

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing OHLCV data
    """
    df = pd.read_csv(path, skiprows=3, header=None)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Drop the Date column and keep only OHLCV numerical values
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df

def load_market_data(symbol: str, 
                    start_date: str = None, 
                    end_date: str = None,
                    source: str = 'auto',
                    for_trading_env: bool = True) -> pd.DataFrame:
    """
    统一的市场数据加载接口
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD) 
        source: 数据源 ('auto', 'database', 'online', 'csv')
        for_trading_env: 是否为交易环境格式化数据
        
    Returns:
        pd.DataFrame: 市场数据
    """
    df = pd.DataFrame()
    
    if not DATA_INFRASTRUCTURE_AVAILABLE and source != 'csv':
        logger.warning("数据基础设施不可用，切换到CSV模式")
        source = 'csv'
    
    try:
        if source == 'auto' and DATA_INFRASTRUCTURE_AVAILABLE:
            # 自动选择最佳数据源
            df = data_pipeline.get_data_with_fallback(symbol, start_date, end_date)
            
        elif source == 'database' and DATA_INFRASTRUCTURE_AVAILABLE:
            # 从数据库加载
            df = data_storage.get_market_data(symbol, start_date, end_date)
            
        elif source == 'online' and DATA_INFRASTRUCTURE_AVAILABLE:
            # 从在线源获取
            if not start_date or not end_date:
                from datetime import datetime, timedelta
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            result = data_pipeline.fetch_yahoo_data(symbol, start_date, end_date)
            if result['success']:
                df = data_storage.get_market_data(symbol, start_date, end_date)
            else:
                logger.error(f"在线获取数据失败: {result.get('error_message')}")
                
        elif source == 'csv':
            # CSV文件加载（向后兼容）
            csv_path = f"data/{symbol}_1d.csv" if not start_date else f"data/{symbol}_{start_date}_{end_date}.csv"
            if os.path.exists(csv_path):
                df = load_csv_data(csv_path)
            else:
                logger.error(f"CSV文件不存在: {csv_path}")
                
        else:
            logger.error(f"不支持的数据源: {source}")
            
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        
    # 为交易环境格式化数据
    if not df.empty and for_trading_env:
        df = format_for_trading_env(df)
        
    return df

def format_for_trading_env(df: pd.DataFrame) -> pd.DataFrame:
    """
    为交易环境格式化数据
    
    Args:
        df: 原始数据
        
    Returns:
        pd.DataFrame: 格式化后的数据
    """
    # 确保列名标准化
    df = df.copy()
    
    # 列名映射
    column_mapping = {
        'Open': 'open', 'HIGH': 'high', 'High': 'high',
        'Low': 'low', 'LOW': 'low',
        'Close': 'close', 'CLOSE': 'close', 
        'Volume': 'volume', 'VOLUME': 'volume',
        'Adj Close': 'adj_close', 'Adj_Close': 'adj_close'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # 确保必要的列存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"数据缺少必要列: {missing_columns}")
        return pd.DataFrame()
    
    # 只保留需要的数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    if 'adj_close' in df.columns:
        numeric_columns.append('adj_close')
    
    df = df[numeric_columns].astype(float)
    
    # 数据清洗
    df = df.dropna()
    df = df[df > 0]  # 移除负值和零值
    
    return df

def load_multiple_symbols(symbols: list,
                         start_date: str = None,
                         end_date: str = None,
                         source: str = 'auto') -> Dict[str, pd.DataFrame]:
    """
    批量加载多个股票数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        source: 数据源
        
    Returns:
        Dict[str, pd.DataFrame]: {symbol: dataframe}
    """
    results = {}
    
    for symbol in symbols:
        logger.info(f"加载 {symbol} 数据...")
        df = load_market_data(symbol, start_date, end_date, source)
        if not df.empty:
            results[symbol] = df
            logger.info(f"✅ {symbol}: {len(df)}行数据")
        else:
            logger.warning(f"❌ {symbol}: 数据加载失败")
    
    return results

def get_data_info(symbol: str) -> Dict[str, Any]:
    """
    获取数据基本信息
    
    Args:
        symbol: 股票代码
        
    Returns:
        Dict: 数据信息
    """
    info = {
        'symbol': symbol,
        'available': False,
        'source': None,
        'date_range': None,
        'record_count': 0
    }
    
    if DATA_INFRASTRUCTURE_AVAILABLE:
        try:
            # 从数据库获取信息
            df = data_storage.get_market_data(symbol)
            if not df.empty:
                info.update({
                    'available': True,
                    'source': 'database',
                    'date_range': {
                        'start': df.index.min().strftime('%Y-%m-%d'),
                        'end': df.index.max().strftime('%Y-%m-%d')
                    },
                    'record_count': len(df)
                })
        except Exception as e:
            logger.error(f"获取数据信息失败: {e}")
    else:
        # 检查CSV文件
        csv_path = f"data/{symbol}_1d.csv"
        if os.path.exists(csv_path):
            info.update({
                'available': True,
                'source': 'csv',
                'file_path': csv_path
            })
    
    return info

# 向后兼容的别名
def load_data_for_env(symbol_or_path: Union[str, os.PathLike], **kwargs) -> pd.DataFrame:
    """向后兼容的数据加载函数"""
    if os.path.exists(str(symbol_or_path)):
        # 如果是文件路径，使用CSV加载
        return load_csv_data(str(symbol_or_path))
    else:
        # 如果是股票代码，使用新的数据加载接口
        return load_market_data(str(symbol_or_path), **kwargs)






