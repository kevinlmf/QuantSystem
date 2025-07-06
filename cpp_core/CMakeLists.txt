cmake_minimum_required(VERSION 3.14)
project(cpp_trading)

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Add source files
pybind11_add_module(cpp_trading
    src/order.cpp
    src/order_executor.cpp
)

# Include headers
target_include_directories(cpp_trading PRIVATE ${CMAKE_SOURCE_DIR}/include)
