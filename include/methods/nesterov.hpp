#ifndef NESTEROV_HPP
#define NESTEROV_HPP

#include "method.hpp"

// Parameters for the gradient descent algorithm
struct NesterovParams : public Params
{
    NesterovParams() = default;
    // Constructor for NesterovParams
    NesterovParams(scalar_function f, vector_function df_grad_f, vector_type initial_condition,
                   scalar_type tolerance_r, scalar_type tolerance_s, scalar_type initial_step,
                   int_type max_iterations, scalar_type minimum_step,
                   scalar_type mu, scalar_type eta)
        : Params(f, df_grad_f, initial_condition, tolerance_r, tolerance_s,
                 initial_step, max_iterations, minimum_step),
          mu(mu), eta(eta) {}
    scalar_type mu;  // Parameter for the exponential decay and the inverse decay
    scalar_type eta; // Memory parameter
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
class Nesterov : public Method
{

public:
    // Constructor with parameters
    Nesterov(const NesterovParams &params) : Method(params), params(params) {}

    /**
     * Run the Nesterov algorithm.
     *
     * @return The converged solution
     *
     * @note The algorithm stops when the norm of the gradient is less than
     * `tolerance_r` or when the step size is less than `tolerance_s`.
     *
     * @note The algorithm uses a dynamic step size based on the Armijo rule if
     * `T == NesterovType::exponential`, an adaptive inverse decay of the step
     * size (improvement) if `T == NesterovType::inverse`, or a constant step
     * size if `T == NesterovType::constant`.
     *
     * @note The algorithm also uses a small number \f$ \epsilon \f$ to avoid
     * division by zero.
     */
    vector_type operator()() const override
    {
        vector_type x = Eigen::Map<const vector_type>(params.initial_condition.data(), params.initial_condition.size());
        scalar_type alpha = params.initial_step;
        index_type iteration = 0;
        vector_type y = Eigen::Map<const vector_type>(params.initial_condition.data(), params.initial_condition.size());

        for (iteration = 0; iteration < params.max_iterations; ++iteration)
        {
            // Compute the gradient at the current point and in auxiliary vector y
            vector_type grad = Eigen::Map<const vector_type>(params.grad_f(x).data(), params.grad_f(x).size());
            vector_type grad_y = Eigen::Map<const vector_type>(params.grad_f(y).data(), params.grad_f(y).size());

            // Check for convergence (norm of the gradient)
            scalar_type residual = grad.norm();
            if (residual < params.tolerance_r)
            {
                std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
                break;
            }

            // Normalization of the gradient
            grad_y.normalize();

            // Use constexpr if to select the descent strategy at compile time
            if (alpha > params.minimum_step)
            {
                if constexpr (T == NesterovType::exponential)
                {
                    // Exponential decay of the step size
                    alpha *= std::exp(-params.mu);
                }
                else if constexpr (T == NesterovType::inverse)
                {
                    // Adaptive inverse decay of the step size (improvement)
                    alpha = params.initial_step / (1 + params.mu * iteration * (1 / residual));
                }
                // if constexpr (T == NesterovType::constant) // not needed
            }

            vector_type x_prev = x;

            // Update the current point
            x = y - alpha * grad_y;
            if constexpr (S == NesterovStrategy::dynamic)
            {
                if (alpha < 1)
                {
                    y = x + (1. - alpha) * (x - x_prev);
                }
                else
                {
                    y = x + params.eta * (x - x_prev);
                }
            }
            else if constexpr (S == NesterovStrategy::constant)
            {
                y = x + params.eta * (x - x_prev);
            }

            // Check for convergence (step size)
            scalar_type step_size = (x - x_prev).norm();
            if (step_size < params.tolerance_s)
            {
                std::cout << "Converged in " << iteration << " iterations thanks to step size criterion." << std::endl;
                break;
            }
        }

        if (iteration == params.max_iterations)
            std::cout << "Not converged (max_iteration = " << iteration << ")" << std::endl;

        return x;
    };

    // Getters
    const Params &get_params() const override { return params; }
    scalar_type get_mu() const { return params.mu; }
    scalar_type get_eta() const { return params.eta; }

    /**
     * Prints the parameters of this method.
     *
     * The parameters are: initial condition, tolerance_r, tolerance_s,
     * initial step, maximum iterations, minimum step, mu, and eta.
     */
    void print() const override
    {
        // Use constexpr if to select the descent strategy at compile time
        if constexpr (T == NesterovType::exponential)
        {
            std::cout << "Descend type: exponential decay of the step size" << std::endl;
        }
        else if constexpr (T == NesterovType::inverse)
        {
            std::cout << "Descend type: inverse decay of the step size" << std::endl;
        }
        else if constexpr (T == NesterovType::constant)
        {
            std::cout << "Descend type: constant step size" << std::endl;
        }
        if constexpr (S == NesterovStrategy::constant)
        {
            std::cout << "Strategy to compute the momentum: constant (eta)" << std::endl;
        }
        else if constexpr (S == NesterovStrategy::dynamic)
        {
            std::cout << "Strategy to compute the momentum: dynamic (1-aplha)" << std::endl;
        }
        Method::print();
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "eta: " << params.eta << std::endl;
    };

private:
    NesterovParams params;
};

#endif // NESTEROV_HPP
