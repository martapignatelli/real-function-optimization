#include "nesterov.hpp"

// Nesterov algorithm
template <NesterovType T, NesterovStrategy S>
Eigen::VectorXd Nesterov<T, S>::operator()() const
{
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(params.initial_condition.data(), params.initial_condition.size());
    double alpha = params.initial_step;
    int iteration = 0;
    Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(params.initial_condition.data(), params.initial_condition.size());

    for (iteration = 0; iteration < params.max_iterations; ++iteration)
    {
        // Compute the gradient at the current point and in auxiliary vector y
        Eigen::VectorXd grad = Eigen::Map<const Eigen::VectorXd>(params.grad_f(x).data(), params.grad_f(x).size());
        Eigen::VectorXd grad_y = Eigen::Map<const Eigen::VectorXd>(params.grad_f(y).data(), params.grad_f(y).size());

        // Check for convergence (norm of the gradient)
        double residual = grad.norm();
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

        Eigen::VectorXd x_prev = x;

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
        double step_size = (x - x_prev).norm();
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
}
// Explicit instantiation of the Nesterov class for the three descent strategies
template class Nesterov<NesterovType::exponential, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::inverse, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::constant, NesterovStrategy::dynamic>;
template class Nesterov<NesterovType::exponential, NesterovStrategy::constant>;
template class Nesterov<NesterovType::inverse, NesterovStrategy::constant>;
template class Nesterov<NesterovType::constant, NesterovStrategy::constant>;
