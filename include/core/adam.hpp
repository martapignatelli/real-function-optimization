#ifndef ADAM_HPP
#define ADAM_HPP

#include <functional>
#include <iostream>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

#include "vector_op.hpp"

// Parameters for the gradient descent algorithm
struct AdamParams
{
    std::function<double(const std::vector<double> &)> f;                   // Function f
    std::function<std::vector<double>(const std::vector<double> &)> grad_f; // Gradient of f
    std::vector<double> initial_condition;                                  // Initial condition
    double tolerance_r;                                                     // Tolerance for convergence (residual)
    double tolerance_s;                                                     // Tolerance for convergence (step length)
    double initial_step;                                                    // Initial step size αlpha0
    int max_iterations;                                                     // Maximal number of iterations
    double minimum_step;                                                    // Minimum step size
    double mu;                                                              // Parameter for the exponential decay and the inverse decay
    double beta1;                                                           // Exponential decay rate for 1st moment estimate
    double beta2;                                                           // Exponential decay rate for 2nd moment estimate  
};

// Descent types
enum class AdamType
{
    dynamic,
    constant
};

// Adam algorithm
template <AdamType T>
class Adam
{

public:
    // Contructor with parameters
    Adam(const AdamParams &params) : params(params) {}
    // Constructor with default parameters
    Adam(
        std::function<double(const std::vector<double> &)> f,
        std::function<std::vector<double>(const std::vector<double> &)> grad_f,
        const std::vector<double> &initial_condition) : params({
                                                            f,
                                                            grad_f,
                                                            initial_condition,
                                                            1e-6,
                                                            1e-6,
                                                            1e-3,
                                                            100,
                                                            1e-6,
                                                            0.2,
                                                            0.9,
                                                            0.999,
                                                        })
    {
    }

    // Run the Adam algorithm
    // It takes two functions as optional arguments: f and grad_f
    std::vector<double> operator()() const;
    std::vector<double> operator()(
        std::function<double(const std::vector<double> &)> f,
        std::function<std::vector<double>(const std::vector<double> &)> grad_f,
        const std::vector<double> &initial_condition)
    {
        params.f = f;
        params.grad_f = grad_f;
        params.initial_condition = initial_condition;
        return operator()();
    };

    // Getters
    std::function<double(const std::vector<double> &)> get_f() const { return params.f; }
    std::function<std::vector<double>(const std::vector<double> &)> get_grad_f() const { return params.grad_f; }
    std::vector<double> get_initial_condition() const { return params.initial_condition; }
    double get_tolerance_r() const { return params.tolerance_r;}
    double get_tolerance_s() const { return params.tolerance_s; }
    double get_initial_step() const { return params.initial_step; }
    int get_max_iterations() const { return params.max_iterations; }
    double get_minimum_step() const { return params.minimum_step; }
    double get_mu() const { return params.mu; }
    double get_beta1() const { return params.beta1; }
    double get_beta2() const { return params.beta2; }
    // Initial condition setter
    void set_initial_condition(const std::vector<double> &initial_condition) { params.initial_condition = initial_condition; }
    // Print the parameters
    void print() const;

private:
    AdamParams params;
};

#endif // ADAM_HPP