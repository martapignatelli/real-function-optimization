#ifndef HEAVY_BALL_HPP
#define HEAVY_BALL_HPP

#include "method.hpp"

// Parameters for the gradient descent algorithm
struct HeavyBallParams : public Params
{
    HeavyBallParams() = default;
    // Constructor for HeavyBallParams
    HeavyBallParams(scalar_function f, vector_function df_grad_f, vector_type initial_condition,
                    scalar_type tolerance_r, scalar_type tolerance_s, scalar_type initial_step,
                    int_type max_iterations, scalar_type minimum_step,
                    scalar_type mu, scalar_type eta)
        : Params(f, df_grad_f, initial_condition, tolerance_r, tolerance_s,
                 initial_step, max_iterations, minimum_step),
          mu(mu), eta(eta) {}
    scalar_type mu;  // Paramter for the exponential decay and the inverse decay
    scalar_type eta; // Memory parameter
};

// Descent types
enum class HeavyBallType
{
    exponential,
    inverse,
    constant
};

enum class HeavyBallStrategy
{
    dynamic,
    constant
};

// Heavy ball algorithm
// T is the type of the descent strategy
// S is the type of the memory strategy
template <HeavyBallType T, HeavyBallStrategy S>
class HeavyBall : public Method
{

public:
    // Constructor with parameters
    HeavyBall(const HeavyBallParams &params) : Method(params), params(params) {}

    /**
     * Run the Heavy Ball algorithm.
     *
     * This function takes no arguments and returns the converged solution.
     *
     * @return The converged solution
     *
     * @note The algorithm stops when the norm of the gradient is less than
     * `tolerance_r` or when the step size is less than `tolerance_s`.
     *
     * @note The algorithm also uses a small number \f$ \epsilon \f$ to avoid
     * division by zero.
     */
    vector_type operator()() const override
    {
        vector_type x = Eigen::Map<const vector_type>(params.initial_condition.data(), params.initial_condition.size());
        scalar_type alpha = params.initial_step;
        index_type iteration = 0;
        vector_type d = vector_type::Zero(params.initial_condition.size());

        for (iteration = 0; iteration < params.max_iterations; ++iteration)
        {
            // Compute the gradient at the current point
            vector_type grad = Eigen::Map<const vector_type>(params.grad_f(x).data(), params.grad_f(x).size());

            // Check for convergence (norm of the gradient)
            scalar_type residual = grad.norm();
            if (residual < params.tolerance_r)
            {
                std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
                break;
            }

            // Normalization of the gradient
            grad.normalize();

            // Use constexpr if to select the descent strategy at compile time
            if (alpha > params.minimum_step)
            {
                if constexpr (T == HeavyBallType::exponential)
                {
                    // Exponential decay of the step size
                    alpha *= std::exp(-params.mu);
                }
                else if constexpr (T == HeavyBallType::inverse)
                {
                    // Adaptive inverse decay of the step size (improvement)
                    alpha = params.initial_step / (1 + params.mu * iteration * (1 / residual));
                }
                // if constexpr (T == HeavyBallType:constant) // not needed
            }

            // Update the current point
            if constexpr (S == HeavyBallStrategy::dynamic)
            {
                if (alpha < 1)
                {
                    d = (1.0 - alpha) * d - alpha * grad;
                    x = x + d;
                }
                else
                {
                    d = params.eta * d - alpha * grad;
                    x = x + d;
                }
            }
            else if constexpr (S == HeavyBallStrategy::constant)
            {
                d = params.eta * d - alpha * grad;
                x = x + d;
            }

            // Check for convergence (step size)
            scalar_type step_size = d.norm();
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
     * Print the parameters of the heavy ball method.
     *
     * This function prints the type of descent strategy used based on the template parameter `T`.
     * It displays whether the strategy is exponential decay, inverse decay, or Armijo rule.
     * Additionally, it prints the method parameters including initial condition, tolerance for
     * convergence (residual and step length), initial step size, maximum iterations,
     * minimum step size, mu, and sigma.
     */
    void print() const override
    {
        // Use constexpr if to select the descent strategy at compile time
        if constexpr (T == HeavyBallType::exponential)
        {
            std::cout << "Descend type: exponential decay of the step size" << std::endl;
        }
        else if constexpr (T == HeavyBallType::inverse)
        {
            std::cout << "Descend type: inverse decay of the step size" << std::endl;
        }
        else if constexpr (T == HeavyBallType::constant)
        {
            std::cout << "Descend type: constant step size" << std::endl;
        }
        if constexpr (S == HeavyBallStrategy::constant)
        {
            std::cout << "Strategy to compute the momentum: constant (eta)" << std::endl;
        }
        else if constexpr (S == HeavyBallStrategy::dynamic)
        {
            std::cout << "Strategy to compute the momentum: dynamic (1-aplha)" << std::endl;
        }
        Method::print();
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "eta: " << params.eta << std::endl;
    };

private:
    HeavyBallParams params;
};

#endif // HEAVY_BALL_HPP
