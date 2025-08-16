try:
    # 优先导入顶层模块名
    from cpp_trading2 import *
except Exception:
    # 回退：部分构建/运行路径下要带包前缀
    from cpp_core.cpp_trading2 import *
