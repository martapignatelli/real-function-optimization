#ifndef VECTOR_OP_HPP
#define VECTOR_OP_HPP

#include <vector>
#include <cmath>
#include <numeric>   // For std::iota
#include <execution> // For std::execution::par
#include <iostream>

// Norm of an std::vector<double>
double norm(const std::vector<double> &x);

// Norm squared of an std::vector<double>
double norm_squared(const std::vector<double> &x);

// Multiply an std::vector<double> by a scalar
std::vector<double> operator*(const double scalar, const std::vector<double> &x);

// Sum of two std::vector<double>
std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y);

// Subtraction of two std::vector<double>
std::vector<double> operator-(const std::vector<double> &x, const std::vector<double> &y);

std::vector<double> elemwise_product(const std::vector<double> &a, const std::vector<double> &b);

std::vector<double> elemwise_division(const std::vector<double> &a, const std::vector<double> &b);

std::vector<double> elemwise_sqrt(const std::vector<double> &a);

#endif // VECTOR_OP_HPP