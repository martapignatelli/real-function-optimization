#include "gradient_descent.hpp"

// Gradient descent algorithm
template <GradientDescentType T>
std::vector<double> GradientDescent<T>::operator()() const
{

    std::vector<double> x = params.initial_condition;
    double alpha = params.initial_step;
    int iteration = 0;

    for (iteration = 0; iteration < params.max_iterations; ++iteration)
    {
        // Compute the gradient at the current point
        // source of error
        std::vector<double> grad = params.grad_f(x);

        // Check for convergence (norm of the gradient)
        double residual = norm(grad);
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
            while (alpha > params.minimum_step && params.f(x) - params.f(x - alpha * grad) < params.sigma * alpha * norm_squared(grad))
                alpha *= 0.5;
        }

        std::vector<double> x_prev = x;

        // Update the current point
        x = x - alpha * grad;

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

template <GradientDescentType T>
void GradientDescent<T>::print() const
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
    std::cout << "sigma: " << params.sigma << std::endl;
}

// Explicit instantiation of the GradientDescent class for the three descent strategies
template class GradientDescent<GradientDescentType::exponential>;
template class GradientDescent<GradientDescentType::inverse>;
template class GradientDescent<GradientDescentType::armijo>;