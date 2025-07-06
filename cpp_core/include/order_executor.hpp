#pragma once
#include "order.hpp"
#include <vector>

class OrderExecutor {
public:
    void submit_order(const Order& order);
    void simulate_execution();
    std::vector<Order> get_filled_orders() const;

private:
    std::vector<Order> orders_;
    std::vector<Order> filled_orders_;
};
