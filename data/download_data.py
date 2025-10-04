"""
Enhanced data download script - integrated data validation and storage
"""
import yfinance as yf
import os
import argparse
from datetime import datetime, timedelta
import logging
from typing import List

# 导入Newdata基础设施module
try:
    from .pipeline import data_pipeline
    from .database import data_storage
    from .validators import DataQualityMonitor
except ImportError:
    # If作为脚本直接Run，Using相对导入
    import sys
    sys.path.append(os.path.dirname(__file__))
    from pipeline import data_pipeline
    from database import data_storage
    from validators import DataQualityMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_yahoo_data(symbol="SPY", start="2023-01-01", end="2024-01-01", 
                       use_pipeline=True, validate=True, save_csv=True):
    """
    DownloadYahoo Financedata（Enhanced版）
    
    Args:
        symbol: Stock代码
        start: Start date
        end: End date
        use_pipeline: Is否UsingNewdata管道
        validate: Is否进行datavalidate
        save_csv: Is否同时SaveCSV文件（兼容性）
    """
    if use_pipeline:
        # UsingNewdata管道
        logger.info(f"Usingdata管道Get {symbol} data")
        result = data_pipeline.fetch_yahoo_data(symbol, start, end, validate)
        
        if result['success']:
            logger.info(f"✅ dataGetsuccess: {result['data_rows']}行")
            logger.info(f"✅ datavalidate: {'Passed' if result['validation_passed'] else 'Failed'}")
            logger.info(f"✅ data存储: {'success' if result['storage_saved'] else 'failure'}")
            
            # Optional：同时SaveCSV文件以保持兼容性
            if save_csv:
                df = data_storage.get_market_data(symbol, start, end)
                if not df.empty:
                    os.makedirs("data", exist_ok=True)
                    filepath = f"data/{symbol}_1d.csv"
                    df.to_csv(filepath)
                    logger.info(f"✅ CSV文件Saved: {filepath}")
        else:
            logger.error(f"❌ dataGetfailure: {result.get('error_message', '未知error')}")
            
    else:
        # UsingOriginalmethod（向后兼容）
        logger.info(f"UsingTraditionalmethodGet {symbol} data")
        df = yf.download(symbol, start=start, end=end)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{symbol}_1d.csv"
        df.to_csv(filepath)
        logger.info(f"✅ Saved data to: {filepath}")

def batch_download(symbols: List[str], start="2023-01-01", end=None):
    """BatchDownload多个Stockdata"""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"StartBatchDownload {len(symbols)} 个Stock")
    
    # Usingdata管道的Batch功能
    result = data_pipeline.batch_fetch_symbols(symbols, start, end)
    
    # PrintResultsSummary
    logger.info("=== BatchDownloadResultsSummary ===")
    logger.info(f"总Stock数: {result['total_symbols']}")
    logger.info(f"success: {result['successful']}")
    logger.info(f"failure: {result['failed']}")
    logger.info(f"validatePassed: {result['validation_passed']}")
    logger.info(f"存储success: {result['storage_saved']}")
    logger.info(f"总data行: {result['total_rows']}")
    logger.info(f"耗时: {result['duration_seconds']:.2f}秒")
    
    # PrintfailureDetails
    failed_details = [d for d in result['details'] if not d['success']]
    if failed_details:
        logger.warning("failureDetails:")
        for detail in failed_details:
            logger.warning(f"  {detail['symbol']}: {detail.get('error_message', '未知error')}")

def update_daily_data():
    """Dailydataupdate"""
    logger.info("StartDailydataupdate")
    result = data_pipeline.run_daily_update()
    
    logger.info("=== DailyupdateResults ===")
    for key, value in result.items():
        logger.info(f"{key}: {value}")

def show_data_summary():
    """Showdata库dataSummary"""
    summary = data_storage.get_data_summary()
    
    print("\n=== data库Summary ===")
    if summary:
        print(f"StockCount: {summary['symbols_count']}")
        print(f"总记录数: {summary['total_records']}")
        print(f"Date范围: {summary['date_range']['earliest']} 到 {summary['date_range']['latest']}")
        
        print("\n各StockDetails:")
        for detail in summary['symbols_detail'][:10]:  # 只Show前10个
            print(f"  {detail['symbol']}: {detail['records']}行 "
                  f"({detail['start_date']} 到 {detail['end_date']})")
    else:
        print("暂无data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced版dataDownload工具')
    parser.add_argument('--symbol', '-s', default='SPY', help='Stock代码')
    parser.add_argument('--symbols', nargs='+', help='BatchDownloadStock代码列表')
    parser.add_argument('--start', default='2023-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD (Default今天)')
    parser.add_argument('--legacy', action='store_true', help='UsingOriginalDownloadmethod')
    parser.add_argument('--no-validate', action='store_true', help='跳过datavalidate')
    parser.add_argument('--batch', action='store_true', help='BatchDownload模式')
    parser.add_argument('--update', action='store_true', help='Dailydataupdate')
    parser.add_argument('--summary', action='store_true', help='ShowdataSummary')
    
    args = parser.parse_args()
    
    if args.summary:
        show_data_summary()
    elif args.update:
        update_daily_data()
    elif args.batch or args.symbols:
        symbols = args.symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        batch_download(symbols, args.start, args.end)
    else:
        # 单个StockDownload
        end_date = args.end or datetime.now().strftime('%Y-%m-%d')
        download_yahoo_data(
            symbol=args.symbol,
            start=args.start,
            end=end_date,
            use_pipeline=not args.legacy,
            validate=not args.no_validate
        )
