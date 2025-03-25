#include "adam.hpp"

// Adam algorithm
template <AdamType T>
std::vector<double> Adam<T>::operator()() const
{
    std::vector<double> x = params.initial_condition;
    double alpha = params.initial_step;
    std::vector<double> espilon(x.size(), 1e-8); // small number to avoid division by zero
    std::vector<double> m(x.size(), 0);          // first moment estimate
    std::vector<double> mhat(x.size(), 0);       // 1st moment estimate normalised
    std::vector<double> v(x.size(), 0);          // second moment estimate
    std::vector<double> vhat(x.size(), 0);       // 2nd moment estimate normalised
    double beta1_iter = params.beta1;            // beta1 elevated to the number of iterations
    double beta2_iter = params.beta2;            // beta2 elevated to the number of iterations
    int iteration = 0;

    for (iteration = 0; iteration < params.max_iterations; ++iteration)
    {
        // Compute the gradient at the current point
        std::vector<double> grad = params.grad_f(x);

        // Check for convergence (norm of the gradient)
        double residual = norm(grad);
        if (residual < params.tolerance_r)
        {
            std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
            break;
        }

        std::vector<double> x_prev = x;

        // Update the current point and auxiliary elements
        m = params.beta1 * m + (1 - params.beta1) * grad;
        v = params.beta2 * v + (1 - params.beta2) * elemwise_product(grad, grad);
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
            // if constexpr (T == AdamType:constant) // not needed
        }

        x = x - alpha * elemwise_division(mhat, (elemwise_sqrt(vhat) + espilon));

        beta1_iter *= params.beta1;
        beta2_iter *= params.beta2;

        // Check for convergence (step size)
        double step_size = norm(x - x_prev);
        if (step_size < params.tolerance_s)
        {
            std::cout << "Converged in " << iteration << " iterations thanks to step size criterion." << std::endl;
            break;
        }
    }

    if (iteration == params.max_iterations)
        std::cout << "Not converged (max_iteration = " << iteration << ")" << std::endl;

    return x;
}

template <AdamType T>
void Adam<T>::print() const
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
    for (const auto &var : params.initial_condition)
    {
        if (!first)
        {
            std::cout << ", ";
        }
        std::cout << var;
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
    std::cout << "beta1: " << params.beta1 << std::endl;
    std::cout << "beta2: " << params.beta2 << std::endl;
}

// Explicit instantiation of the Adam class for the three descent strategies
template class Adam<AdamType::dynamic>;
template class Adam<AdamType::constant>;