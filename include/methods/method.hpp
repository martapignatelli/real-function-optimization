#ifndef METHOD_HPP
#define METHOD_HPP

#include <Math>

// Parameters for the gradient descent algorithm
struct Params
{
    Params() = default;
     // Constructor to initialize Params using an initializer list
     Params(scalar_function f, vector_function grad_f, vector_type initial_condition,
        scalar_type tolerance_r, scalar_type tolerance_s, scalar_type initial_step,
        int_type max_iterations, scalar_type minimum_step)
     : f(f), grad_f(grad_f), initial_condition(initial_condition),
       tolerance_r(tolerance_r), tolerance_s(tolerance_s), 
       initial_step(initial_step), max_iterations(max_iterations),
       minimum_step(minimum_step) {}

    // contructor
    scalar_function f;             // Function f
    vector_function grad_f;        // Gradient of f
    vector_type initial_condition; // Initial condition
    scalar_type tolerance_r;       // Tolerance for convergence (residual)
    scalar_type tolerance_s;       // Tolerance for convergence (step length)
    scalar_type initial_step;      // Initial step size αlpha0
    int_type max_iterations;       // Maximal number of iterations
    scalar_type minimum_step;      // Minimum step size
    // virtual destructor
    virtual ~Params() {}
};

class Method
{

public:
    // Constructor with paramters
    Method(const Params &params) : params(params) {}

    // Run the algorithm
    virtual vector_type operator()() const = 0;

    /**
     * Run the algorithm.
     *
     * This function takes two functions as optional arguments: f and grad_f
     * and an initial condition.
     *
     * @param f The function to optimize.
     * @param grad_f The gradient of the function to optimize.
     * @param initial_condition The initial condition of the algorithm.
     * @return The minimum of the function.
     */
    vector_type operator()(
        scalar_function f,
        vector_function grad_f,
        const vector_type &initial_condition)
    {
        params.f = f;
        params.grad_f = grad_f;
        params.initial_condition = initial_condition;
        return operator()();
    };

    // Getters
    virtual const Params &get_params() const { return params; }
    scalar_function get_f() const { return params.f; }
    vector_function get_grad_f() const { return params.grad_f; }
    vector_type get_initial_condition() const { return params.initial_condition; }
    scalar_type get_tolerance_r() const { return params.tolerance_r; }
    scalar_type get_tolerance_s() const { return params.tolerance_s; }
    scalar_type get_initial_step() const { return params.initial_step; }
    int_type get_max_iterations() const { return params.max_iterations; }
    scalar_type get_minimum_step() const { return params.minimum_step; }

    // Initial condition setter
    void set_initial_condition(const vector_type &initial_condition) { params.initial_condition = initial_condition; }

    /**
     * Prints the parameters of the method.
     *
     * This function outputs the initial condition, tolerance for the residual
     * and step length, initial step size, maximum number of iterations, and
     * minimum step size to the standard output.
     */
    virtual void print() const
    {
        Eigen::IOFormat commaFormat(Eigen::StreamPrecision, 0, ",", ",", "", "", "(", ")");
        const Params &p = get_params(); // Get correct params
        std::cout << "initial_condition: " << p.initial_condition.format(commaFormat) << std::endl;
        std::cout << "tolerance_r: " << p.tolerance_r << std::endl;
        std::cout << "tolerance_s: " << p.tolerance_s << std::endl;
        std::cout << "initial_step: " << p.initial_step << std::endl;
        std::cout << "max_iterations: " << p.max_iterations << std::endl;
        std::cout << "minimum_step: " << p.minimum_step << std::endl;
    };

    // virtual destructor
    virtual ~Method() {}

private:
    Params params;
};

#endif // METHOD_HPP
