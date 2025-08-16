"""
测试新的数据基础设施
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.pipeline import data_pipeline
from data.database import data_storage
from data.validators import DataQualityMonitor
import pandas as pd
from datetime import datetime, timedelta


def test_data_validation():
    """测试数据验证功能"""
    print("=== 测试数据验证功能 ===")

    # ✅ 正常数据（确保 Close/Open 均在 [Low, High] 区间内）
    test_data = pd.DataFrame({
        'Open':  [100.0, 101.0,  99.0, 102.0],
        'High':  [102.0, 103.0, 100.5, 104.0],
        'Low':   [ 99.0, 100.0,  98.0, 101.0],
        'Close': [101.0, 101.0, 100.0, 103.0],  # 第二行修正为 101.0
        'Volume':[1_000_000, 1_200_000, 800_000, 1_500_000]
    })
    monitor = DataQualityMonitor()
    passed = monitor.monitor_data_quality(test_data, 'TEST')
    print(f"✅ 数据验证测试: {'通过' if passed else '失败'}")
    assert passed is True, "期望正常数据应通过质量检查"

    # 异常数据（用于验证能检测出问题）
    bad_data = pd.DataFrame({
        'Open':  [100.0, 101.0,  -99.0, 102.0],   # 负价格
        'High':  [102.0, 103.0, 100.5, 104.0],
        'Low':   [ 99.0, 100.0,  98.0, 105.0],    # Low > High
        'Close': [101.0,  99.0, 100.0, 103.0],
        'Volume':[1_000_000, 1_200_000, -800_000, 1_500_000]  # 负成交量
    })
    passed_bad = monitor.monitor_data_quality(bad_data, 'BAD_TEST')
    print(f"✅ 异常数据检测: {'检测到异常' if not passed_bad else '未检测到异常'}")
    assert passed_bad is False, "期望异常数据应未通过质量检查"


def test_database_operations():
    """测试数据库操作"""
    print("\n=== 测试数据库操作 ===")

    # 构造并落库
    test_data = pd.DataFrame({
        'Date':   pd.date_range('2024-01-01', periods=5),
        'Open':   [100.0, 101.0, 99.0, 102.0, 98.0],
        'High':   [102.0, 103.0, 100.5, 104.0, 99.5],
        'Low':    [99.0, 100.0, 98.0, 101.0, 97.0],
        'Close':  [101.0, 99.0, 100.0, 103.0, 98.5],
        'Volume': [1_000_000, 1_200_000, 800_000, 1_500_000, 900_000]
    }).set_index('Date')

    result = data_storage.save_market_data(test_data, 'TEST_SYMBOL', 'test')
    print(f"✅ 数据保存: {result['status']} - {result['rows_added']}行")
    assert result['status'] == 'success', "保存市场数据应成功"
    assert result['rows_added'] >= 1, "应至少写入一行数据"

    # 读回
    retrieved = data_storage.get_market_data('TEST_SYMBOL')
    print(f"✅ 数据读取: {len(retrieved)}行")
    assert not retrieved.empty, "读取数据不应为空"
    assert {'open','high','low','close','volume'}.issubset(retrieved.columns), "读取列缺失"

    # 摘要
    summary = data_storage.get_data_summary()
    print(f"✅ 数据摘要: {summary.get('total_records', 0)}总记录")
    assert summary.get('total_records', 0) >= len(retrieved), "摘要记录数应合理"


def test_data_pipeline():
    """测试数据管道"""
    print("\n=== 测试数据管道 ===")
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"获取SPY数据: {start_date} 到 {end_date}")
    result = data_pipeline.fetch_yahoo_data('SPY', start_date, end_date, validate=True)

    print("✅ 管道测试结果:")
    print(f"   成功: {result['success']}")
    print(f"   数据行数: {result['data_rows']}")
    print(f"   验证通过: {result['validation_passed']}")
    print(f"   存储成功: {result['storage_saved']}")
    if not result['success']:
        print(f"   错误: {result.get('error_message', '未知错误')}")

    assert result['success'] is True, "数据管道应运行成功"
    assert result['data_rows'] >= 1, "应至少拉取到一行数据"
    assert result['validation_passed'] is True, "数据质量验证应通过"
    assert result['storage_saved'] is True, "数据应成功写入存储"


def test_data_retrieval():
    """测试数据获取 with 容错"""
    print("\n=== 测试数据获取容错 ===")
    df = data_pipeline.get_data_with_fallback('SPY', max_staleness_days=30)

    if not df.empty:
        print(f"✅ 获取SPY数据: {len(df)}行")
        print(f"   日期范围: {df.index.min()} 到 {df.index.max()}")
        print(f"   最新收盘价: {df['close'].iloc[-1]:.2f}")
    else:
        print("❌ 未能获取SPY数据")
    assert not df.empty, "应能通过缓存/网络至少得到一段 SPY 数据"


def test_batch_operations():
    """测试批量操作"""
    print("\n=== 测试批量操作 ===")
    symbols = ['AAPL', 'MSFT']
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"批量获取 {symbols} 数据")
    result = data_pipeline.batch_fetch_symbols(symbols, start_date, end_date, max_workers=2)

    print("✅ 批量操作结果:")
    print(f"   成功: {result['successful']}/{result['total_symbols']}")
    print(f"   总数据行: {result['total_rows']}")
    print(f"   耗时: {result['duration_seconds']:.2f}秒")

    assert result['total_symbols'] == len(symbols), "统计的总股票数不正确"
    assert result['successful'] >= 1, "至少应有一只股票抓取成功"
    assert result['total_rows'] >= result['successful'], "行数统计应合理"


# 保留 CLI 入口（pytest 不会调用）
def run_all_tests():
    print("开始测试数据基础设施...")
    tests = [
        ("数据验证", test_data_validation),
        ("数据库操作", test_database_operations),
        ("数据管道", test_data_pipeline),
        ("数据获取", test_data_retrieval),
        ("批量操作", test_batch_operations),
    ]
    results = {}
    for name, fn in tests:
        try:
            fn()
            results[name] = True
        except Exception as e:
            print(f"❌ {name} 测试失败: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    passed = sum(1 for ok in results.values() if ok)
    for name, ok in results.items():
        print(f"{name:<15} {'✅ 通过' if ok else '❌ 失败'}")
    print(f"\n总体结果: {passed}/{len(tests)} 测试通过")
    if passed == len(tests):
        print("🎉 所有测试通过！数据基础设施就绪。")
    else:
        print("⚠️  部分测试失败，请检查配置和网络连接。")


if __name__ == "__main__":
    run_all_tests()

