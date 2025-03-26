#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include <Math>

// Parameters for the gradient descent algorithm
struct GradientDescentParams
{
    std::function<double(const Eigen::VectorXd &)> f;                   // Function f
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;     // Gradient of f
    Eigen::VectorXd initial_condition;                                  // Initial condition
    double tolerance_r;                                                 // Tolerance for convergence (residual)
    double tolerance_s;                                                 // Tolerance for convergence (step length)
    double initial_step;                                                // Initial step size αlpha0
    int max_iterations;                                                 // Maximal number of iterations
    double minimum_step;                                                // Minimum step size for the Armijo rule
    double sigma;                                                       // Parameter for the Armijo rule
    double mu;                                                          // Parameter for the exponential decay and the inverse decay
};

// Descent types
enum class GradientDescentType
{
    exponential,
    inverse,
    armijo
};

// Gradient descent algorithm
// T is the type of the descent strategy
template <GradientDescentType T>
class GradientDescent
{

public:
    // Constructor with parameters
    GradientDescent(const GradientDescentParams &params) : params(params) {}
    // Constructor with default parameters
    GradientDescent(
        std::function<double(const Eigen::VectorXd &)> f,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f,
        const Eigen::VectorXd &initial_condition) : params({
                                                            f,
                                                            grad_f,
                                                            initial_condition,
                                                            1e-6,
                                                            1e-6,
                                                            1,
                                                            1000,
                                                            1e-2,
                                                            0.1,
                                                            0.2,
                                                        })
    {
    }
    // Run the gradient descent algorithm
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
    double get_tolerance_r() const { return params.tolerance_r; }
    double get_tolerance_s() const { return params.tolerance_s; }
    double get_initial_step() const { return params.initial_step; }
    int get_max_iterations() const { return params.max_iterations; }
    // Initial condition setter
    void set_initial_condition(const Eigen::VectorXd &initial_condition) { params.initial_condition = initial_condition; }
    // Print the parameters
    void print() const;

private:
    GradientDescentParams params;
};

#endif // GRADIENT_DESCENT_HPP
