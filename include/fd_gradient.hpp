
#ifndef FD_GRADIENT_HPP
#define FD_GRADIENT_HPP

#include <functional>
#include <type_traits> // for std::is_same_v
#include <Eigen/Dense>

// Finite difference types
namespace DifferenceType
{
  struct Backward; // forward declaration
  /*!
   * It expresses the concept of a forward difference
   */
  struct Forward
  {
    //! I alternate among different types when performing higher derivatives
    using otherType = Backward;
  };
  /*!
   * It expresses the concept of a backward difference
   */
  struct Backward
  {
    using otherType = Forward;
  };

  struct Centered
  {
    using otherType = Centered;
  };
} // namespace DifferenceType

/// @brief This function computes the gradient of a real valued function by finite differences
/// @tparam F is the callable object of signature T (std::vector<T> const & )
/// @tparam T is is the difference type: forward, backward or centered
/// @tparam DT is the difference type: forward, backward or centered
/// @param f is the callable object of signature T (std::vector<T> const & )
/// @param h is the step for computing the gradient
/// @return a callable object of signature std::vector<T> (const std::vector<T> &)
/// @note it uses the same signature as the input function
/// @warning it does not check if the input function is valid
/// @warning it does not check if the step is valid
/*
 * Example usage: Computing the gradient of sin(x) + sin(y) at x0 = (0, 0)
 * auto f = [](std::vector<double> x) { return std::sin(x[0]) + std::sin(x[1]); };
 * auto d4 = derive<decltype(f), double, DifferenceType::Centered>(f, 1.e-4);
 * auto d  = d4(x0); // Computes the gradient of f at x0 (x0 is a vector of size 2)
 */
template <typename F, typename T, typename DT = DifferenceType::Centered>
std::function<std::vector<T>(const std::vector<T> &)> gradient(const F &f, const T &h)
{
  return [=](const std::vector<T> &x) -> std::vector<T>
  {
    std::vector<T> grad(x.size(), 0);

    for (std::size_t i = 0; i < x.size(); ++i)
    {
      std::vector<T> x_forward = x;
      std::vector<T> x_backward = x;

      x_forward[i] += h;
      x_backward[i] -= h;

      if constexpr (std::is_same_v<DT, DifferenceType::Forward>)
      {
        grad[i] = (f(x_forward) - f(x)) / h;
      }
      else if constexpr (std::is_same_v<DT, DifferenceType::Backward>)
      {
        grad[i] = (f(x) - f(x_backward)) / h;
      }
      else // Centered
      {
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h);
      }
    }

    return grad;
  };
}

#endif // FD_GRADIENT_HPP
