#ifndef NESTEROV_HPP
#define NESTEROV_HPP

#include "method.hpp"

// Parameters for the gradient descent algorithm
struct NewtonParams : public Params
{
};

// Descent types
enum class NewtonType
{
    exponential,
    inverse,
    constant
};

// Nesterov algorithm
// T is the type of the descent strategy
template <NewtonType T>
class Newton : public Method
{

public:
    // Constructor with parameters
    Newton(const NewtonParams &params) : params(params) {}
    // Constructor with default parameters
    Newton(
        scalar_function f,
        vector_function grad_f,
        const vector_type &initial_condition) : params({
                                                    f,
                                                    grad_f,
                                                    initial_condition,
                                                    1e-6,
                                                    1e-6,
                                                    1,
                                                    100,
                                                    1e-2,
                                                })
    {
    }

    vector_type operator()() const override {};

    // Getters
    const Params &get_params() const override { return params; }
    scalar_type get_mu() const { return params.mu; }
    scalar_type get_eta() const { return params.eta; }

    /**
     * Prints the parameters of this method.
     *
     * The parameters are: initial condition, tolerance_r, tolerance_s,
     * initial step, maximum iterations, minimum step, mu, and eta.
     */
    void print() const override
    {
        Method::print();
    };

private:
    NesterovParams params;
};

#endif // NESTEROV_HPP
