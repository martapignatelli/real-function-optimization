#ifndef RUN_HPP
#define RUN_HPP

#include <Math>
#include <Methods>

/// @brief Prints the computed minimum, function value at the minimum, and gradient norm at the minimum
/// @param minimum computed minimum
/// @param f function
/// @param grad_f gradient of the function f
void print_result(const vector_type &minimum,
                  scalar_function f,
                  vector_function grad_f);

/// @brief Runs the gradient descent algorithm
/// @param params specified parameters
/// @param gradient_method_t step size method
void run_gradient_descent(const GradientDescentParams &params, const string_type &gradient_method_t);

/// @brief Runs the heavy ball algorithm
/// @param params specified parameters
/// @param heavy_ball_t step size method
/// @param heavy_ball_s memory strategy
void run_heavy_ball(const HeavyBallParams &params, const string_type &heavy_ball_t, const string_type &heavy_ball_s);

/// @brief Runs the nesterov algorithm
/// @param params specified parameters
/// @param nesterov_t step size method
/// @param nesterov_s memory strategy
void run_nesterov(const NesterovParams &params, const string_type &nesterov_t, const string_type &nesterov_s);

/// @brief Runs the adam algorithm
/// @param params specified parameters
/// @param nesterov_t step size method
void run_adam(const AdamParams &params, const string_type &adam_t);

#endif // READ_HPP