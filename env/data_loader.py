"""
dataLoad器 - 集成新data基础设施的统一data接口
"""
import pandas as pd
import os
import sys
from typing import Union, Optional, Dict, Any
import logging

# Import data infrastructure
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from data.pipeline import data_pipeline
    from data.database import data_storage
    DATA_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    DATA_INFRASTRUCTURE_AVAILABLE = False
    logging.warning("data基础设施Not found，将UsingTraditionalCSVLoadmethod")

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
    统一的市场dataLoad接口
    
    Args:
        symbol: Stock代码
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) 
        source: data源 ('auto', 'database', 'online', 'csv')
        for_trading_env: Is否为Trading EnvironmentFormatdata
        
    Returns:
        pd.DataFrame: 市场data
    """
    df = pd.DataFrame()
    
    if not DATA_INFRASTRUCTURE_AVAILABLE and source != 'csv':
        logger.warning("data基础设施Unavailable，Switching toCSV模式")
        source = 'csv'
    
    try:
        if source == 'auto' and DATA_INFRASTRUCTURE_AVAILABLE:
            # Auto选择最佳data源
            df = data_pipeline.get_data_with_fallback(symbol, start_date, end_date)
            
        elif source == 'database' and DATA_INFRASTRUCTURE_AVAILABLE:
            # 从data库Load
            df = data_storage.get_market_data(symbol, start_date, end_date)
            
        elif source == 'online' and DATA_INFRASTRUCTURE_AVAILABLE:
            # 从在线源Get
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
                logger.error(f"在线Getdatafailure: {result.get('error_message')}")
                
        elif source == 'csv':
            # CSV文件Load（向后兼容）
            csv_path = f"data/{symbol}_1d.csv" if not start_date else f"data/{symbol}_{start_date}_{end_date}.csv"
            if os.path.exists(csv_path):
                df = load_csv_data(csv_path)
            else:
                logger.error(f"CSV文件Does not exist: {csv_path}")
                
        else:
            logger.error(f"不支持的data源: {source}")
            
    except Exception as e:
        logger.error(f"dataLoadfailure: {e}")
        
    # 为Trading EnvironmentFormatdata
    if not df.empty and for_trading_env:
        df = format_for_trading_env(df)
        
    return df

def format_for_trading_env(df: pd.DataFrame) -> pd.DataFrame:
    """
    为Trading EnvironmentFormatdata
    
    Args:
        df: Originaldata
        
    Returns:
        pd.DataFrame: Format后的data
    """
    # 确保列名Normalization
    df = df.copy()
    
    # 列名Map
    column_mapping = {
        'Open': 'open', 'HIGH': 'high', 'High': 'high',
        'Low': 'low', 'LOW': 'low',
        'Close': 'close', 'CLOSE': 'close', 
        'Volume': 'volume', 'VOLUME': 'volume',
        'Adj Close': 'adj_close', 'Adj_Close': 'adj_close'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # 确保Required的列存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"dataMissingRequired列: {missing_columns}")
        return pd.DataFrame()
    
    # 只保留需要的数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    if 'adj_close' in df.columns:
        numeric_columns.append('adj_close')
    
    df = df[numeric_columns].astype(float)
    
    # data清洗
    df = df.dropna()
    df = df[df > 0]  # Remove负值和零值
    
    return df

def load_multiple_symbols(symbols: list,
                         start_date: str = None,
                         end_date: str = None,
                         source: str = 'auto') -> Dict[str, pd.DataFrame]:
    """
    BatchLoad多个Stockdata
    
    Args:
        symbols: Stock代码列表
        start_date: Start date
        end_date: End date
        source: data源
        
    Returns:
        Dict[str, pd.DataFrame]: {symbol: dataframe}
    """
    results = {}
    
    for symbol in symbols:
        logger.info(f"Load {symbol} data...")
        df = load_market_data(symbol, start_date, end_date, source)
        if not df.empty:
            results[symbol] = df
            logger.info(f"✅ {symbol}: {len(df)}行data")
        else:
            logger.warning(f"❌ {symbol}: dataLoadfailure")
    
    return results

def get_data_info(symbol: str) -> Dict[str, Any]:
    """
    Getdata基本Info
    
    Args:
        symbol: Stock代码
        
    Returns:
        Dict: dataInfo
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
            # 从data库GetInfo
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
            logger.error(f"GetdataInfofailure: {e}")
    else:
        # checkCSV文件
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
    """向后兼容的dataLoadfunction"""
    if os.path.exists(str(symbol_or_path)):
        # IfIs文件路径，UsingCSVLoad
        return load_csv_data(str(symbol_or_path))
    else:
        # IfIsStock代码，UsingNewdataLoad接口
        return load_market_data(str(symbol_or_path), **kwargs)






