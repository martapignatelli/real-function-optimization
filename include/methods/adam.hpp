#ifndef ADAM_HPP
#define ADAM_HPP

#include "method.hpp"

// Parameters for the gradient descent algorithm
struct AdamParams : public Params
{
    scalar_type mu;                // Parameter for the exponential decay and the inverse decay
    scalar_type beta1;             // Exponential decay rate for 1st moment estimate
    scalar_type beta2;             // Exponential decay rate for 2nd moment estimate
};

// Descent types
enum class AdamType
{
    dynamic,
    constant
};

// Adam algorithm
template <AdamType T>
class Adam : public Method
{

public:
    // Constructor with parameters
    Adam(const AdamParams &params) : params(params) {}

    // Constructor with default parameters
    Adam(
        scalar_function f,
        vector_function grad_f,
        const vector_type &initial_condition) : params({
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

    /**
     * Run the Adam algorithm.
     *
     * @return The converged solution
     *
     * @note The algorithm stops when the norm of the gradient is less than
     * `tolerance_r` or when the step size is less than `tolerance_s`.
     *
     * @note The algorithm uses a dynamic step size based on the moment estimates
     * of the gradient, which are computed using the following formulas:
     * \f[
     *     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
     *     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
     *     \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
     *     \hat{v}_t &= \frac{v_t}{1 - \beta_2^t}
     * \f]
     * The step size is then computed as:
     * \f[
     *     \alpha_t &= \frac{\alpha_0 \sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
     * \f]
     *
     * @note The algorithm also uses a small number \f$ \epsilon \f$ to avoid division by zero.
     */
    vector_type operator()() const override
    {
        vector_type x = vector_type::Map(params.initial_condition.data(), params.initial_condition.size());
        scalar_type alpha = params.initial_step;
        vector_type epsilon = vector_type::Ones(x.size()) * 1e-8; // small number to avoid division by zero
        vector_type m = vector_type::Zero(x.size());              // first moment estimate
        vector_type mhat = vector_type::Zero(x.size());           // 1st moment estimate normalised
        vector_type v = vector_type::Zero(x.size());              // second moment estimate
        vector_type vhat = vector_type::Zero(x.size());           // 2nd moment estimate normalised
        scalar_type beta1_iter = params.beta1;                    // beta1 elevated to the number of iterations
        scalar_type beta2_iter = params.beta2;                    // beta2 elevated to the number of iterations
        index_type iteration = 0;

        for (iteration = 0; iteration < params.max_iterations; ++iteration)
        {
            // Compute the gradient at the current point
            vector_type grad = vector_type::Map(params.grad_f(x).data(), params.grad_f(x).size());

            // Check for convergence (norm of the gradient)
            scalar_type residual = grad.norm();
            if (residual < params.tolerance_r)
            {
                std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
                break;
            }

            vector_type x_prev = x;

            // Update the current point and auxiliary elements
            m = params.beta1 * m + (1 - params.beta1) * grad;
            v = params.beta2 * v + (1 - params.beta2) * grad.array().square().matrix();
            // Correct bias in moment estimates and update
            mhat = 1 / (1 - beta1_iter) * m;
            vhat = 1 / (1 - beta2_iter) * v;

            // Compute adaptive learning rate
            // Use constexpr if to select the descent strategy at compile time
            if (alpha > params.minimum_step)
            {
                if constexpr (T == AdamType::dynamic)
                {
                    // Adaptive dynamic decay of the step size
                    alpha = params.initial_step * std::sqrt(1 - beta2_iter) / (1 - beta1_iter);
                }
                // if constexpr (T == AdamType::constant) // not needed
            }

            x = x - alpha * (mhat.array() / (vhat.array().sqrt() + epsilon.array())).matrix();

            beta1_iter *= params.beta1;
            beta2_iter *= params.beta2;

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
    scalar_type get_beta1() const { return params.beta1; }
    scalar_type get_beta2() const { return params.beta2; }

    // Initial condition setter
    void set_initial_condition(const vector_type &initial_condition) { params.initial_condition = initial_condition; }

    /**
     * Print the parameters of this method.
     *
     * The parameters are: initial condition, tolerance_r, tolerance_s,
     * initial step, maximum iterations, minimum step, mu, beta1, and beta2.
     */
    void print() const override
    {
        // Use constexpr if to select the descent strategy at compile time
        if constexpr (T == AdamType::dynamic)
        {
            std::cout << "Descend type: dynamic decay of the step size" << std::endl;
        }
        else if constexpr (T == AdamType::constant)
        {
            std::cout << "Descend type: constant step size" << std::endl;
        }
        Method::print();
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "beta1: " << params.beta1 << std::endl;
        std::cout << "beta2: " << params.beta2 << std::endl;
    };

private:
    AdamParams params;
};

#endif // ADAM_HPP
