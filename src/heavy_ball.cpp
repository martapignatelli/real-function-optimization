#include <functional>
#include <iostream>
#include <cmath>
#include <limits>

#include "heavy_ball.hpp"

// Heavy Ball algorithm
template <HeavyBallType T, HeavyBallStrategy S>
std::vector<double> HeavyBall<T, S>::operator()() const
{

    std::vector<double> x = params.initial_condition;
    double alpha = params.initial_step;
    int iteration = 0;
    std::vector<double> d(params.initial_condition.size(), 0.0);

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
        double step_size = norm(d);
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

template <HeavyBallType T, HeavyBallStrategy S>
void HeavyBall<T, S>::print() const
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

// Explicit instantiation of the HeavyBall class for the three descent strategies
template class HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::dynamic>;
template class HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::dynamic>;
template class HeavyBall<HeavyBallType::constant, HeavyBallStrategy::dynamic>;
template class HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::constant>;
template class HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::constant>;
template class HeavyBall<HeavyBallType::constant, HeavyBallStrategy::constant>;