#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "method.hpp"

// Parameters for the gradient descent algorithm
struct GradientDescentParams : public Params
{
    scalar_type sigma; // Parameter for the Armijo rule
    scalar_type mu;    // Parameter for the exponential decay and the inverse decay
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
class GradientDescent : public Method
{

public:
    // Constructor with parameters
    GradientDescent(const GradientDescentParams &params) : params(params) {}

    // Constructor with default parameters
    GradientDescent(
        scalar_function f,
        vector_function grad_f,
        const vector_type &initial_condition) : params({
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

    /**
     * Run the gradient descent algorithm.
     *
     * @return The converged solution
     *
     * @note The algorithm stops when the norm of the gradient is less than
     * `tolerance_r` or when the step size is less than `tolerance_s`.
     *
     * @note The algorithm uses a dynamic step size based on the Armijo rule if
     * `T == GradientDescentType::armijo`, an exponential decay of the step size
     * if `T == GradientDescentType::exponential`, or an adaptive inverse decay
     * of the step size (improvement) if `T == GradientDescentType::inverse`.
     *
     * @note The algorithm also uses a small number \f$ \epsilon \f$ to avoid
     * division by zero.
     */
    vector_type operator()() const override
    {
        vector_type x = Eigen::Map<const vector_type>(params.initial_condition.data(), params.initial_condition.size());
        scalar_type alpha = params.initial_step;
        index_type iteration = 0;

        for (iteration = 0; iteration < params.max_iterations; ++iteration)
        {
            // Compute the gradient at the current point
            // source of error
            vector_type grad = params.grad_f(x);

            // Check for convergence (norm of the gradient)
            scalar_type residual = grad.norm();
            if (residual < params.tolerance_r)
            {
                std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
                break;
            }
            if constexpr (T == GradientDescentType::exponential || T == GradientDescentType::inverse)
            {
                grad = (1 / residual) * grad;
            }

            // Use constexpr if to select the descent strategy at compile time
            if constexpr (T == GradientDescentType::exponential)
            {
                // Exponential decay of the step size
                alpha *= std::exp(-params.mu);
            }
            else if constexpr (T == GradientDescentType::inverse)
            {
                // Adaptive inverse decay of the step size (improvement)
                alpha = params.initial_step / (1 + params.mu * iteration * (1 / residual));
            }
            else if constexpr (T == GradientDescentType::armijo)
            {
                alpha = params.initial_step;
                // Armijo rule for the step size
                while (alpha > params.minimum_step && params.f(x) - params.f(x - alpha * grad) < params.sigma * alpha * grad.squaredNorm())
                    alpha *= 0.5;
            }

            vector_type x_prev = x;

            // Update the current point
            x = x - alpha * grad;

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
    scalar_type get_sigma() const { return params.sigma; }

    /**
     * Prints the parameters of the gradient descent algorithm.
     *
     * The parameters are: the type of the descent strategy, the initial
     * condition, the tolerance for the norm of the gradient, the tolerance for
     * the step size, the initial step size, the maximum number of iterations,
     * the minimum step size, \f$ \mu \f$ and \f$ \sigma \f$.
     */
    void print() const
    {
        // Use constexpr if to select the descent strategy at compile time
        if constexpr (T == GradientDescentType::exponential)
        {
            std::cout << "Descend type: exponential decay of the step size" << std::endl;
        }
        else if constexpr (T == GradientDescentType::inverse)
        {
            std::cout << "Descend type: inverse decay of the step size" << std::endl;
        }
        else if constexpr (T == GradientDescentType::armijo)
        {
            std::cout << "Descend type: Armijo for the step size" << std::endl;
        }
        Method::print();
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "sigma: " << params.sigma << std::endl;
    };

private:
    GradientDescentParams params;
};

#endif // GRADIENT_DESCENT_HPP
