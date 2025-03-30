#include "readnew.hpp"

/// @brief 
/// @param datafile 
/// @param params 
void read(const GetPot &datafile, Params &params)
{
    const int_type N = datafile.vector_variable_size("initial_condition");                                 // Dimension of the problem
    const string_type f_str = datafile("f", "4*x[0]*x[0]*x[0]*x[0] + 2*x[1]*x[1] + 2*x[0]*x[1] + 2*x[0]"); // Function f
    std::cout << "Function to be optimized: " << f_str << std::endl;
    muParserXScalarInterface f(f_str, N);                           // Initialize the function with muparserx
    const bool fd = datafile("fd", true);                           // Use finite differences to compute the gradient
    const scalar_type tolerance_r = datafile("tolerance_r", 1e-6);  // Tolerance for convergence (residual)
    const scalar_type tolerance_s = datafile("tolerance_s", 1e-6);  // Tolerance for convergence (step length)
    const scalar_type initial_step = datafile("initial_step", 1.0); // Initial step size αlpha0
    const int max_iterations = datafile("max_iterations", 1000);    // Maximal number of iterations
    const scalar_type mu = datafile("mu", 0.2);                     // Parameter for the exponential and inverse decay
    const scalar_type minimum_step = datafile("minimum_step", 1e-2); // Minimum step size

    vector_type initial_condition = vector_type::Zero(N);           // Initial condition
    if (N > 0)
    {
        // If entries exist, parse them
        for (int i = 0; i < N; ++i)
        {
            initial_condition[i] = datafile("initial_condition", 0.0, i); // default 0.0 if not found
        }
    }
    else
    {
        // Default values
        initial_condition = vector_type::Zero(2);
    }

    // Gradient of f
    std::function<vector_type(const vector_type &)> grad_f;
    
    // if user prefers to use the approximate gradient
    if (fd)
    {
        const string_type fd_t = datafile("fd_t", "Centered");
        const scalar_type h = datafile("h", 1e-2);
        std::cout << "Finite differences type: " << fd_t << " (h = " << h << ")" << std::endl;
        if (fd_t == "Forward")
        {
            grad_f = gradient<decltype(f), scalar_type, DifferenceType::Forward>(f, h);
        }
        else if (fd_t == "Backward")
        {
            grad_f = gradient<decltype(f), scalar_type, DifferenceType::Backward>(f, h);
        }
        else
        {
            grad_f = gradient<decltype(f), scalar_type, DifferenceType::Centered>(f, h);
        }
    }
    else
    {
        const string_type grad_f_str = datafile("grad_f", "{16*x[0]*x[0]*x[0] + 2*x[1] +2, 4*x[1]+2*x[0]}"); // Gradient of f
        muParserXVectorInterface grad_f(grad_f_str, N);                                                      // Initialize the gradient with muparserx
    }

    
    if (dynamic_cast<GradientDescentParams *>(&params) != nullptr)
    {
        auto *p = dynamic_cast<GradientDescentParams *>(&params);
        // Gradient descent specific paramters
        const scalar_type sigma = datafile("sigma", 0.1);       // Parameter for the Armijo rule  
        *p = {
            f,
            grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            sigma,
            mu,
        };
        return;
    }

    else if (dynamic_cast<HeavyBallParams *>(&params) != nullptr)
    {
        auto *p = dynamic_cast<HeavyBallParams *>(&params);
        // Heavy ball specific paramters
        const scalar_type eta = datafile("eta", 0.9);       // Memory parameter for Heavy ball
        (*p) = {
            f,
            grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            mu,
            eta,
        };
        return;
    }

    else if (dynamic_cast<NesterovParams *>(&params) != nullptr)
    {
        auto *p = dynamic_cast<NesterovParams *>(&params);
        // Nesterov specific paramters
        const scalar_type eta = datafile("eta_nest", 0.9);   // Memory parameter for Nesterov
        (*p) = {
            f,
            grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            mu,
            eta,
        };
        return;
    }

    else if (dynamic_cast<AdamParams *>(&params) != nullptr)
    {
        auto *p = dynamic_cast<AdamParams *>(&params);
        // Adam specific paramters
        const scalar_type beta1 = datafile("beta1", 0.9);     // Exponential decay rate for 1st moment estimate
        const scalar_type beta2 = datafile("beta2", 0.999);   // Exponential decay rate for 2nd moment estimate
        (*p) = {
            f,
            grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            mu,
            beta1,
            beta2,
        };
        return;
    }
    
}