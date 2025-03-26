#ifndef FD_HESSIAN_HPP
#define FD_HESSIAN_HPP

#include "fd_gradient.hpp"

/**
 * Compute the Hessian of a function by finite differences
 *
 * @brief Compute the Hessian of a function by finite differences
 * @param f the function to compute the Hessian of
 * @param h the step for computing the Hessian
 * @return a callable object of signature std::vector<std::vector<T>>(const std::vector<T> &)
 * @note it uses the same signature as the input function
 * @warning it does not check if the input function is valid
 * @warning it does not check if the step is valid
 * @example Computing the Hessian of sin(x) + sin(y) at x0 = (0, 0)
 * auto f = [](std::vector<double> x) { return std::sin(x[0]) + std::sin(x[1]); };
 * auto d4 = derive<decltype(f), double, DifferenceType::Centered>(f, 1.e-4);
 * auto d  = d4(x0); // Computes the Hessian of f at x0 (x0 is a vector of size 2)
 */

template <typename F, typename T, typename DT = DifferenceType::Centered>
std::function<matrix_type(const vector_type &)> hessian(const F &f, const T &h)
{
    return [=](const vector_type &x) -> matrix_type
    {
        // Initialize the Hessian matrix
        matrix_type hessian = matrix_type::Zero(x.size(), x.size());

        // Compute the gradient with finite differences (DT type)
        auto grad = gradient<decltype(f), double, DT>(f, h);

        // Compute the Hessian
        for (int i = 0; i < x.size(); ++i)
        {
            // The i-th component of the gradient
            auto grad_i = [=](const vector_type &x)
            { return grad(x)(i); };

            // Gradient of the i-th component of the gradient of f (other type of finite differences)
            auto grad_grad_i = gradient<decltype(f), double, typename DT::otherType>(f, h);

            // Compute and insert the i-th row
            hessian.row(i) = grad_grad_i(x);
        }

        return hessian;
    };
}

#endif // FD_HESSIAN_HPP