#ifndef RUN_HPP
#define RUN_HPP

#include "gradient_descent.hpp"
#include "heavy_ball.hpp"
#include "nesterov.hpp"
#include "adam.hpp"

/// @brief Prints the computed minimum, function value at the minimum, and gradient norm at the minimum
/// @param minimum 
/// @param f 
/// @param grad_f 
void print_result(const std::vector<double> &minimum, std::function<double(const std::vector<double> &)> f,
                  std::function<std::vector<double>(const std::vector<double> &)> grad_f);
                  
/// @brief Runs the gradient descent algorithm
/// @param params specified parameters
/// @param gradient_method_t step size method
void run_gradient_descent(const GradientDescentParams &params, const std::string &gradient_method_t);

/// @brief Runs the heavy ball algorithm
/// @param params specified parameters
/// @param heavy_ball_t step size method
/// @param heavy_ball_s memory strategy
void run_heavy_ball(const HeavyBallParams &params, const std::string &heavy_ball_t, const std::string &heavy_ball_s);

/// @brief Runs the nesterov algorithm
/// @param params specified parameters
/// @param nesterov_t step size method
/// @param nesterov_s memory strategy
void run_nesterov(const NesterovParams &params, const std::string &nesterov_t, const std::string &nesterov_s);

/// @brief Runs the adam algorithm
/// @param params specified parameters
/// @param nesterov_t step size method
void run_adam(const AdamParams &params, const std::string &adam_t);

#endif //READ_HPP