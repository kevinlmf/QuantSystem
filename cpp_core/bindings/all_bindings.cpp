#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "data_feed.h"
#include "order.hpp"
#include "order_executor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_trading2, m) {
    // --- Bind DataFeed ---
    py::class_<Row>(m, "Row")
        .def_readwrite("date", &Row::date)
        .def_readwrite("open", &Row::open)
        .def_readwrite("high", &Row::high)
        .def_readwrite("low", &Row::low)
        .def_readwrite("close", &Row::close)
        .def_readwrite("volume", &Row::volume);

    py::class_<DataFeed>(m, "DataFeed")
        .def(py::init<>())
        .def("load", &DataFeed::load)
        .def("next", &DataFeed::next)
        .def("current", &DataFeed::current)
        .def("moving_average", &DataFeed::moving_average);

    // --- Bind OrderType enum ---
    py::enum_<OrderType>(m, "OrderType")
        .value("BUY", OrderType::BUY)
        .value("SELL", OrderType::SELL);

    // --- Bind Order struct ---
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("type", &Order::type)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("timestamp", &Order::timestamp);

    // --- Bind OrderExecutor class ---
    py::class_<OrderExecutor>(m, "OrderExecutor")
        .def(py::init<>())
        .def("submit_order", &OrderExecutor::submit_order)
        .def("simulate_execution", &OrderExecutor::simulate_execution)
        .def("get_filled_orders", &OrderExecutor::get_filled_orders);
}
