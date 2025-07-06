#include "order_executor.hpp"

void OrderExecutor::submit_order(const Order& order) {
    orders_.push_back(order);
}

void OrderExecutor::simulate_execution() {
    // For simplicity: all orders are filled instantly
    filled_orders_ = orders_;
    orders_.clear();
}

std::vector<Order> OrderExecutor::get_filled_orders() const {
    return filled_orders_;
}
