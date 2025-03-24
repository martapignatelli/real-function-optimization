#include <vector>
#include <cmath>
#include <numeric>   // For std::iota
#include <execution> // For std::execution::par
#include <iostream>

/**
 * @brief Calculates the Euclidean norm of a vector using a parallel algorithm.
 *
 * @param x Input vector.
 * @return Euclidean norm of the input vector.
 */
double norm(const std::vector<double> &x)
{
    double norm_sq = std::transform_reduce(
        std::execution::par,
        x.begin(), x.end(),
        x.begin(), 0.0,
        std::plus<>(),
        std::multiplies<>());

    return std::sqrt(norm_sq);
}

/**
 * @brief Calculates the squared norm of a vector using a parallel algorithm.
 *
 * @param x Input vector.
 * @return Squared norm of the input vector.
 */
double norm_squared(const std::vector<double> &x)
{
    return std::transform_reduce(
        std::execution::par, // Parallel execution
        x.begin(), x.end(),  // Input range
        x.begin(),           // Second input range (same as first)
        0.0,                 // Initial value
        std::plus<>(),       // Reduction operation (sum)
        std::multiplies<>()  // Element-wise operation (square)
    );
}

/**
 * @brief Multiplies a vector by a scalar value using a parallel algorithm.
 *
 * @param scalar Input scalar value.
 * @param x Input vector.
 * @return Vector with elements that are the product of the input scalar and
 *         the input vector.
 */
std::vector<double> operator*(const double scalar, const std::vector<double> &x)
{
    std::vector<double> result(x.size());
    std::vector<std::size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      result[i] = x[i] * scalar;
                  });

    return result;
}

/**
 * @brief Element-wise addition of two std::vector<double> using a parallel algorithm.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @return Vector with elements that are the sum of the input vectors.
 */
std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y)
{
    std::vector<double> result(x.size());
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        return result;
    }

    std::vector<std::size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      result[i] = x[i] + y[i];
                  });

    return result;
}

/**
 * @brief Element-wise subtraction of two std::vector<double> using a parallel algorithm.
 *
 * @param x Input vector.
 * @param y Input vector.
 * @return Vector with elements that are the difference of the input vectors.
 */
std::vector<double> operator-(const std::vector<double> &x, const std::vector<double> &y)
{
    std::vector<double> result(x.size());
    if (x.size() != y.size())
    {
        std::cerr << "Error: vectors have different sizes" << std::endl;
        return result;
    }

    std::vector<std::size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      result[i] = x[i] - y[i];
                  });

    return result;
}

/**
 * @brief Element-wise product of two std::vector<double> using a parallel algorithm.
 *
 * @param a Input vector.
 * @param b Input vector.
 * @return Vector with elements that are the product of the input vectors.
 */
std::vector<double> elemwise_product(const std::vector<double> &a, const std::vector<double> &b)
{
    std::vector<double> result(a.size());
    if (!a.size() || a.size() != b.size())
    {
        std::cerr << "Error: vectors must have the same positive size." << std::endl;
        return result;
    }

    std::vector<std::size_t> indices(a.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      result[i] = a[i] * b[i];
                  });

    return result;
}

/**
 * @brief Element-wise division of two std::vector<double> using a parallel algorithm.
 *
 * @param a Input vector.
 * @param b Input vector.
 * @return Vector with elements that are the quotient of the input vectors.
 *
 * If an element of the denominator is zero, an error message is printed and the
 * corresponding element of the result is set to zero.
 */
std::vector<double> elemwise_division(const std::vector<double> &a, const std::vector<double> &b)
{
    std::vector<double> result(a.size());
    if (!a.size() || a.size() != b.size())
    {
        std::cerr << "Error: vectors must have the same positive size." << std::endl;
        return result;
    }

    std::vector<std::size_t> indices(a.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      if (b[i] == 0)
                      {
                          std::cerr << "Error: division by zero at index " << i << std::endl;
                          result[i] = 0;
                      }
                      else
                      {
                          result[i] = a[i] / b[i];
                      }
                  });

    return result;
}

/**
 * @brief Element-wise square root of a std::vector<double> using a parallel algorithm.
 *
 * @param a Input vector.
 * @return Vector with elements that are the square root of the input vector.
 *
 * If an element of the input vector is negative, an error message is printed and
 * the corresponding element of the result is set to zero.
 */
std::vector<double> elemwise_sqrt(const std::vector<double> &a)
{
    std::vector<double> result(a.size());
    if (!a.size())
    {
        std::cerr << "Error: vector must have a positive size." << std::endl;
        return result;
    }

    std::vector<std::size_t> indices(a.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](std::size_t i)
                  {
                      if (a[i] < 0)
                      {
                          std::cerr << "Error: square root of a negative number at index " << i << std::endl;
                          result[i] = 0;
                      }
                      else
                      {
                          result[i] = std::sqrt(a[i]);
                      }
                  });

    return result;
}
