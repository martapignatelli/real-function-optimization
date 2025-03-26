#ifndef ADAM_HPP
#define ADAM_HPP

#include <Math>

// Parameters for the gradient descent algorithm
struct AdamParams
{
    scalar_function f;             // Function f
    vector_function grad_f;        // Gradient of f
    vector_type initial_condition; // Initial condition
    scalar_type tolerance_r;       // Tolerance for convergence (residual)
    scalar_type tolerance_s;       // Tolerance for convergence (step length)
    scalar_type initial_step;      // Initial step size αlpha0
    int_type max_iterations;       // Maximal number of iterations
    scalar_type minimum_step;      // Minimum step size
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
class Adam
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
    vector_type operator()() const
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
        int_type iteration = 0;

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

    /**
     * Run the Adam algorithm.
     *
     * This function takes two functions as optional arguments: f and grad_f
     * and an initial condition.
     *
     * @param f The function to optimize.
     * @param grad_f The gradient of the function to optimize.
     * @param initial_condition The initial condition of the algorithm.
     * @return The minimum of the function.
     */
    vector_type operator()(
        scalar_function f,
        vector_function grad_f,
        const vector_type &initial_condition)
    {
        params.f = f;
        params.grad_f = grad_f;
        params.initial_condition = initial_condition;
        return operator()();
    };

    // Getters
    scalar_function get_f() const { return params.f; }
    vector_function get_grad_f() const { return params.grad_f; }
    vector_type get_initial_condition() const { return params.initial_condition; }
    scalar_type get_tolerance_r() const { return params.tolerance_r; }
    scalar_type get_tolerance_s() const { return params.tolerance_s; }
    scalar_type get_initial_step() const { return params.initial_step; }
    int_type get_max_iterations() const { return params.max_iterations; }
    scalar_type get_minimum_step() const { return params.minimum_step; }
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
    void print() const
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
        std::cout << "The parameters of this method are:" << std::endl;
        std::cout << "initial_condition: (";
        bool first = true;
        for (int i = 0; i < params.initial_condition.size(); ++i)
        {
            if (!first)
            {
                std::cout << ", ";
            }
            std::cout << params.initial_condition[i];
            first = false;
        }
        std::cout << ")";
        std::cout << std::endl;
        std::cout << "tolerance_r: " << params.tolerance_r << std::endl;
        std::cout << "tolerance_s: " << params.tolerance_s << std::endl;
        std::cout << "initial_step: " << params.initial_step << std::endl;
        std::cout << "max_iterations: " << params.max_iterations << std::endl;
        std::cout << "minimum_step: " << params.minimum_step << std::endl;
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "beta1: " << params.beta1 << std::endl;
        std::cout << "beta2: " << params.beta2 << std::endl;
    };

private:
    AdamParams params;
};

#endif // ADAM_HPP
