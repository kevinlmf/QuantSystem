from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cpp_trading2",  # 和setup中的name保持一致
        [
            "bindings/all_bindings.cpp",   # ✅ 合并绑定
            "src/data_feed.cpp",
            "src/order.cpp",
            "src/order_executor.cpp"
        ],
        include_dirs=["include"],
    ),
]

setup(
    name="cpp_trading2",
    version="0.2",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)


