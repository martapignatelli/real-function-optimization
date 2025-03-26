#ifndef HEAVY_BALL_HPP
#define HEAVY_BALL_HPP

#include <Math>

// Parameters for the gradient descent algorithm
struct HeavyBallParams
{
    scalar_function f;             // Function f
    vector_function grad_f;        // Gradient of f
    vector_type initial_condition; // Initial condition
    scalar_type tolerance_r;       // Tolerance for convergence (residual)
    scalar_type tolerance_s;       // Tolerance for convergence (step length)
    scalar_type initial_step;      // Initial step size αlpha0
    int_type max_iterations;       // Maximal number of iterations
    scalar_type minimum_step;      // Minimum step size
    scalar_type mu;                // Paramter for the exponential decay and the inverse decay
    scalar_type eta;               // Memory parameter
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
template <HeavyBallType T, HeavyBallStrategy S>
class HeavyBall
{

public:
    // Constructor with parameters
    HeavyBall(const HeavyBallParams &params) : params(params) {}

    // Constructor with default parameters
    HeavyBall(
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
                                                    0.2,
                                                    0.9,
                                                })
    {
    }

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
    vector_type operator()() const
    {
        vector_type x = Eigen::Map<const vector_type>(params.initial_condition.data(), params.initial_condition.size());
        scalar_type alpha = params.initial_step;
        int_type iteration = 0;
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
            grad = (1 / residual) * grad;

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

    /*!
     * Run the Heavy Ball algorithm.
     *
     * This function takes three arguments: a scalar function f, its gradient
     * grad_f, and an initial condition. It modifies the internal parameters
     * of the HeavyBall object and then calls the other operator()() to run
     * the algorithm.
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
    scalar_type get_eta() const { return params.eta; }

    // Initial condition setter
    void set_initial_condition(const vector_type &initial_condition) { params.initial_condition = initial_condition; }

    /**
     * Print the parameters of the heavy ball method.
     *
     * This function prints the type of descent strategy used based on the template parameter `T`.
     * It displays whether the strategy is exponential decay, inverse decay, or Armijo rule.
     * Additionally, it prints the method parameters including initial condition, tolerance for
     * convergence (residual and step length), initial step size, maximum iterations,
     * minimum step size, mu, and sigma.
     */
    void print() const
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
        std::cout << "\b)" << std::endl;
        std::cout << "tolerance_r: " << params.tolerance_r << std::endl;
        std::cout << "tolerance_s: " << params.tolerance_s << std::endl;
        std::cout << "initial_step: " << params.initial_step << std::endl;
        std::cout << "max_iterations: " << params.max_iterations << std::endl;
        std::cout << "minimum_step: " << params.minimum_step << std::endl;
        std::cout << "mu: " << params.mu << std::endl;
        std::cout << "eta: " << params.eta << std::endl;
    };

private:
    HeavyBallParams params;
};

#endif // HEAVY_BALL_HPP
