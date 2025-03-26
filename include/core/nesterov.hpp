#ifndef NESTEROV_HPP
#define NESTEROV_HPP

#include <Math> 

// Parameters for the gradient descent algorithm
struct NesterovParams
{
    std::function<double(const Eigen::VectorXd &)> f;                     // Function f
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;       // Gradient of f
    Eigen::VectorXd initial_condition;                                    // Initial condition
    double tolerance_r;                                                   // Tolerance for convergence (residual)
    double tolerance_s;                                                   // Tolerance for convergence (step length)
    double initial_step;                                                  // Initial step size αlpha0
    int max_iterations;                                                   // Maximal number of iterations
    double minimum_step;                                                  // Minimum step size
    double mu;                                                            // Parameter for the exponential decay and the inverse decay
    double eta;                                                           // Memory parameter
};

// Descent types
enum class NesterovType
{
    exponential,
    inverse,
    constant
};

enum class NesterovStrategy
{
    dynamic,
    constant
};

// Nesterov algorithm
// T is the type of the descent strategy
template <NesterovType T, NesterovStrategy S>
class Nesterov
{

public:
    // Constructor with parameters
    Nesterov(const NesterovParams &params) : params(params) {}
    // Constructor with default parameters
    Nesterov(
        std::function<double(const Eigen::VectorXd &)> f,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f,
        const Eigen::VectorXd &initial_condition) : params({
                                                            f,
                                                            grad_f,
                                                            initial_condition,
                                                            1e-6,
                                                            1e-6,
                                                            1,
                                                            100,
                                                            1e-2,
                                                            0.2,
                                                            0.9,
                                                        })
    {
    }

    // Run the Nesterov algorithm
    // It takes two functions as optional arguments: f and grad_f
    Eigen::VectorXd operator()() const;
    Eigen::VectorXd operator()(
        std::function<double(const Eigen::VectorXd &)> f,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f,
        const Eigen::VectorXd &initial_condition)
    {
        params.f = f;
        params.grad_f = grad_f;
        params.initial_condition = initial_condition;
        return operator()();
    };

    // Getters
    std::function<double(const Eigen::VectorXd &)> get_f() const { return params.f; }
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> get_grad_f() const { return params.grad_f; }
    Eigen::VectorXd get_initial_condition() const { return params.initial_condition; }
    double get_tolerance_r() const { return params.tolerance_r;}
    double get_tolerance_s() const { return params.tolerance_s; }
    double get_initial_step() const { return params.initial_step; }
    int get_max_iterations() const { return params.max_iterations; }
    double get_minimum_step() const { return params.minimum_step; }
    double get_eta() const { return params.eta; }
    // Initial condition setter
    void set_initial_condition(const Eigen::VectorXd &initial_condition) { params.initial_condition = initial_condition; }
    // Print the parameters
    void print() const;

private:
    NesterovParams params;
};

#endif // NESTEROV_HPP
