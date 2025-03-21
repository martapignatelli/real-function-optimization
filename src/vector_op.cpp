#include <vector>
#include <iostream>
#include <cmath>

#include "vector_op.hpp"

// Norm of an std::vector<double>
double norm(const std::vector<double> &x)
{
    double norm = 0.0;
    for (double val : x)
    {
        norm += val * val;
    }
    return std::sqrt(norm);
}

// Norm squared of an std::vector<double>
double norm_squared(const std::vector<double> &x)
{
    double norm = 0.0;
    for (double val : x)
    {
        norm += val * val;
    }
    return norm;
}

// Multiply an std::vector<double> by a scalar
std::vector<double> operator*(const double scalar, const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = x[i] * scalar;
    }
    return result;
}

// Sum of two std::vector<double>
std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y)
{
    std::vector<double> result(x.size());
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        return result;
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = x[i] + y[i];
    }
    return result;
}

// Subtraction of two std::vector<double>
std::vector<double> operator-(const std::vector<double> &x, const std::vector<double> &y)
{
    std::vector<double> result(x.size());
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        return result;
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = x[i] - y[i];
    }
    return result;
}

// Auxiliary functions for operations in Adam algorithm

std::vector<double> elemwise_product(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    if (!a.size() || a.size() != b.size())
    {
        std::cerr << "Error: vectors must have the same positive size." << std::endl;
        return result;
    }
    for (unsigned int i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] * b[i];
    }
    return result;
};


std::vector<double> elemwise_division(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    if (!a.size() || a.size() != b.size())
    {
        std::cerr << "Error: vectors must have the same positive size." << std::endl;
        return result;
    }
    for (unsigned int i = 0; i < a.size(); ++i)
    {
        if (b[i] == 0)
        {
            std::cerr << "Error: division by zero." << std::endl;
            return result;
        }
        result[i] = a[i] / b[i];
    }
    return result;
};


std::vector<double> elemwise_sqrt(const std::vector<double> &a) {
    std::vector<double> result(a.size());
    if (!a.size())
    {
        std::cerr << "Error: vector must have a positive size." << std::endl;
        return result;
    }
    for (unsigned int i = 0; i < a.size(); ++i)
    {
        if (a[i] < 0)
        {
            std::cerr << "Error: square root of a negative number." << std::endl;
            return result;
        }
        result[i] = std::sqrt(a[i]);
    }
    return result;
};