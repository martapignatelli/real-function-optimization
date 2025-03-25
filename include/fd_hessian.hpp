#ifndef FD_HESSIAN_HPP
#define FD_HESSIAN_HPP

#include <functional>
#include "fd_gradient.hpp"

/// @brief Computes the Hessian matrix of second-order partial derivatives of a real-valued function.
/// @tparam F is a callable object with signature T (std::vector<T> const &)
/// @tparam T is the type of the input and output values (e.g., double)
/// @param f is the callable object representing the function for which to compute the Hessian.
/// @param h is the step size used for computing finite differences.
/// @return A callable object that computes the Hessian matrix, represented as a vector of vectors, 
///         for a given point in the input space.

template <typename F, typename T>
std::function<std::vector<std::vector<T>>(const std::vector<T> &)> hessian(const F &f, const T &h)
{

    return [=](const std::vector<T> &x) -> std::vector<std::vector<T>>
    {
        // initialize the hessian
        std::vector<std::vector<T>> hessian(x.size(), std::vector<T>(x.size(), 0));

        // compute the gradient with finite differences
        auto grad = gradient<decltype(f), double, DifferenceType::Centered>(f, h);

        // compute the hessian
        for (std::size_t i = 0; i < grad.size(); ++i)
        {
            // the i-th component of the gradient
            auto grad_i = [=](const std::vector<T> &x)
            { return grad(x)[i]; };

            // the i-th row of the hessian
            auto grad_grad_i = gradient<decltype(f), double, DifferenceType::Centered>(f, h);

            // compute and insert the i-th row
            hessian.push_back(grad_grad_i(x));
        }

        return hessian;
    };
}

#endif // FD_HESSIAN_HPP