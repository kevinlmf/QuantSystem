"""
Enhanced data download script - integrated data validation and storage
"""
import yfinance as yf
import os
import argparse
from datetime import datetime, timedelta
import logging
from typing import List

# 导入新的数据基础设施模块
try:
    from .pipeline import data_pipeline
    from .database import data_storage
    from .validators import DataQualityMonitor
except ImportError:
    # 如果作为脚本直接运行，使用相对导入
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
    下载Yahoo Finance数据（增强版）
    
    Args:
        symbol: 股票代码
        start: 开始日期
        end: 结束日期
        use_pipeline: 是否使用新的数据管道
        validate: 是否进行数据验证
        save_csv: 是否同时保存CSV文件（兼容性）
    """
    if use_pipeline:
        # 使用新的数据管道
        logger.info(f"使用数据管道获取 {symbol} 数据")
        result = data_pipeline.fetch_yahoo_data(symbol, start, end, validate)
        
        if result['success']:
            logger.info(f"✅ 数据获取成功: {result['data_rows']}行")
            logger.info(f"✅ 数据验证: {'通过' if result['validation_passed'] else '未通过'}")
            logger.info(f"✅ 数据存储: {'成功' if result['storage_saved'] else '失败'}")
            
            # 可选：同时保存CSV文件以保持兼容性
            if save_csv:
                df = data_storage.get_market_data(symbol, start, end)
                if not df.empty:
                    os.makedirs("data", exist_ok=True)
                    filepath = f"data/{symbol}_1d.csv"
                    df.to_csv(filepath)
                    logger.info(f"✅ CSV文件已保存: {filepath}")
        else:
            logger.error(f"❌ 数据获取失败: {result.get('error_message', '未知错误')}")
            
    else:
        # 使用原始方法（向后兼容）
        logger.info(f"使用传统方法获取 {symbol} 数据")
        df = yf.download(symbol, start=start, end=end)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{symbol}_1d.csv"
        df.to_csv(filepath)
        logger.info(f"✅ Saved data to: {filepath}")

def batch_download(symbols: List[str], start="2023-01-01", end=None):
    """批量下载多个股票数据"""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"开始批量下载 {len(symbols)} 个股票")
    
    # 使用数据管道的批量功能
    result = data_pipeline.batch_fetch_symbols(symbols, start, end)
    
    # 打印结果摘要
    logger.info("=== 批量下载结果摘要 ===")
    logger.info(f"总股票数: {result['total_symbols']}")
    logger.info(f"成功: {result['successful']}")
    logger.info(f"失败: {result['failed']}")
    logger.info(f"验证通过: {result['validation_passed']}")
    logger.info(f"存储成功: {result['storage_saved']}")
    logger.info(f"总数据行: {result['total_rows']}")
    logger.info(f"耗时: {result['duration_seconds']:.2f}秒")
    
    # 打印失败详情
    failed_details = [d for d in result['details'] if not d['success']]
    if failed_details:
        logger.warning("失败详情:")
        for detail in failed_details:
            logger.warning(f"  {detail['symbol']}: {detail.get('error_message', '未知错误')}")

def update_daily_data():
    """每日数据更新"""
    logger.info("开始每日数据更新")
    result = data_pipeline.run_daily_update()
    
    logger.info("=== 每日更新结果 ===")
    for key, value in result.items():
        logger.info(f"{key}: {value}")

def show_data_summary():
    """显示数据库数据摘要"""
    summary = data_storage.get_data_summary()
    
    print("\n=== 数据库摘要 ===")
    if summary:
        print(f"股票数量: {summary['symbols_count']}")
        print(f"总记录数: {summary['total_records']}")
        print(f"日期范围: {summary['date_range']['earliest']} 到 {summary['date_range']['latest']}")
        
        print("\n各股票详情:")
        for detail in summary['symbols_detail'][:10]:  # 只显示前10个
            print(f"  {detail['symbol']}: {detail['records']}行 "
                  f"({detail['start_date']} 到 {detail['end_date']})")
    else:
        print("暂无数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强版数据下载工具')
    parser.add_argument('--symbol', '-s', default='SPY', help='股票代码')
    parser.add_argument('--symbols', nargs='+', help='批量下载股票代码列表')
    parser.add_argument('--start', default='2023-01-01', help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', help='结束日期 YYYY-MM-DD (默认今天)')
    parser.add_argument('--legacy', action='store_true', help='使用原始下载方法')
    parser.add_argument('--no-validate', action='store_true', help='跳过数据验证')
    parser.add_argument('--batch', action='store_true', help='批量下载模式')
    parser.add_argument('--update', action='store_true', help='每日数据更新')
    parser.add_argument('--summary', action='store_true', help='显示数据摘要')
    
    args = parser.parse_args()
    
    if args.summary:
        show_data_summary()
    elif args.update:
        update_daily_data()
    elif args.batch or args.symbols:
        symbols = args.symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
        batch_download(symbols, args.start, args.end)
    else:
        # 单个股票下载
        end_date = args.end or datetime.now().strftime('%Y-%m-%d')
        download_yahoo_data(
            symbol=args.symbol,
            start=args.start,
            end=end_date,
            use_pipeline=not args.legacy,
            validate=not args.no_validate
        )
