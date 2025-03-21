#include <functional>
#include <iostream>
#include <cmath>
#include <limits>

#include "nesterov.hpp"

// Nesterov algorithm
template <NesterovType T, NesterovStrategy S>
std::vector<double> Nesterov<T, S>::operator()() const
{

    std::vector<double> x = params.initial_condition;
    double alpha = params.initial_step;
    int iteration = 0;
    std::vector<double> y{params.initial_condition};

    for (iteration = 0; iteration < params.max_iterations; ++iteration)
    {
        // Compute the gradient at the current point and in auxiliary vector y
        std::vector<double> grad = params.grad_f(x);
        std::vector<double> grad_y = params.grad_f(y);

        // Check for convergence (norm of the gradient)
        double residual = norm(grad);
        if (residual < params.tolerance_r)
        {
            std::cout << "Converged in " << iteration << " iterations thanks to residual criterion." << std::endl;
            break;
        }

        // Normalization of the gradient
        grad_y = (1 / norm(grad_y)) * grad_y;

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
            // if constexpr (T == NesterovType:constant) // not needed
        }

        std::vector<double> x_prev = x;

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

template <NesterovType T, NesterovStrategy S>
void Nesterov<T, S>::print() const
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
    std::cout << "eta: " << params.eta << std::endl;
}
// Explicit instantiation of the Nesterov class for the three descent strategies
template class Nesterov<NesterovType::exponential, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::inverse, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::constant, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::exponential, NesterovStrategy::constant>;
template class Nesterov<NesterovType::inverse, NesterovStrategy::constant>;
template class Nesterov<NesterovType::constant, NesterovStrategy::constant>;