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

void run_solver(const Method &solver);

void run(const Params &params, const string_type &method_t, const string_type &method_s = "");

#endif // READ_HPP