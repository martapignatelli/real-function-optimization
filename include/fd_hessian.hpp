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
         const int_type n = x.size();
         matrix_type hessian = matrix_type::Zero(n, n);
 
         // Create an index vector [0, 1, 2, ..., n-1]
         std::vector<index_type> indices(n);
         std::iota(indices.begin(), indices.end(), 0);
 
         // Compute each Hessian row in parallel
         std::for_each(std::execution::par, indices.begin(), indices.end(), [&](index_type i)
         {
             // Define grad_i: the i-th component of the gradient of f.
             auto grad_i = [=](const vector_type &y) -> scalar_type {
                 // Compute the full gradient of f at y, then select component i.
                 auto grad = gradient<decltype(f), scalar_type, DT>(f, h);
                 return grad(y)(i);
             };
 
             // Now compute the gradient of grad_i, using the alternate finite difference type.
             auto grad_grad_i = gradient<decltype(grad_i), double, typename DT::otherType>(grad_i, h);
             
             // Compute the i-th row of the Hessian as the gradient of grad_i at x.
             hessian.row(i) = grad_grad_i(x);
         });
 
         return hessian;
     };
 }
 

#endif // FD_HESSIAN_HPP