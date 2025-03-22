#include "read.hpp"

GradientDescentParams read_gradient_descent(const GetPot &datafile)
{
    const int N = datafile.vector_variable_size("initial_condition");                                      // Dimension of the problem
    const std::string f_str = datafile("f", "4*x[0]*x[0]*x[0]*x[0] + 2*x[1]*x[1] + 2*x[0]*x[1] + 2*x[0]"); // Function f
    std::cout << "Function to be optimized: " << f_str << std::endl;
    muParserXScalarInterface f(f_str, N);                        // Initialize the function with muparserx
    const bool fd = datafile("fd", true);                        // Use finite differences to compute the gradient
    const double tolerance_r = datafile("tolerance_r", 1e-6);    // Tolerance for convergence (residual)
    const double tolerance_s = datafile("tolerance_s", 1e-6);    // Tolerance for convergence (step length)
    const double initial_step = datafile("initial_step", 1.0);   // Initial step size αlpha0
    const int max_iterations = datafile("max_iterations", 1000); // Maximal number of iterations
    const double mu = datafile("mu", 0.2);                       // Parameter for the exponential and inverse decay
    std::vector<double> initial_condition(N, 0);                 // Initial condition
    if (N > 0)
    {
        // If entries exist, parse them
        initial_condition.resize(N);
        for (int i = 0; i < N; ++i)
        {
            initial_condition[i] = datafile("initial_condition", 0.0, i); // default 0.0 if not found
        }
    }
    else
    {
        // Default values
        initial_condition = {0., 0.};
    }
    // Gradient descent specific paramters
    const double minimum_step = datafile("minimum_step", 1e-2); // Minimum step size for the Armijo rule
    const double sigma = datafile("sigma", 0.1);                // Parameter for the Armijo rule

    // if user prefers to use the approximate gradient
    GradientDescentParams params;
    if (fd)
    {
        const std::string fd_t = datafile("fd_t", "Centered");
        const double h = datafile("h", 1e-2);
        std::cout << "Finite differences type: " << fd_t << " (h = " << h << ")" << std::endl;
        std::function<std::vector<double>(const std::vector<double> &)> df_grad_f;
        if (fd_t == "Forward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Forward>(f, h);
        }
        else if (fd_t == "Backward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Backward>(f, h);
        }
        else
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Centered>(f, h);
        }
        params = {
            f,
            df_grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            sigma,
            mu,
        };
    }
    else
    {
        const std::string grad_f_str = datafile("grad_f", "{16*x[0]*x[0]*x[0] + 2*x[1] +2, 4*x[1]+2*x[0]}"); // Gradient of f
        muParserXInterface grad_f(grad_f_str, N);                                                            // Initialize the gradient with muparserx
        // Parameters for the gradient descent algorithm
        params = {
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
    }
    return params;
}

HeavyBallParams read_heavy_ball(const GetPot &datafile)
{
    const int N = datafile.vector_variable_size("initial_condition");                                      // Dimension of the problem
    const std::string f_str = datafile("f", "4*x[0]*x[0]*x[0]*x[0] + 2*x[1]*x[1] + 2*x[0]*x[1] + 2*x[0]"); // Function f
    std::cout << "Function to be optimized: " << f_str << std::endl;
    muParserXScalarInterface f(f_str, N);                        // Initialize the function with muparserx
    const bool fd = datafile("fd", true);                        // Use finite differences to compute the gradient
    const double tolerance_r = datafile("tolerance_r", 1e-6);    // Tolerance for convergence (residual)
    const double tolerance_s = datafile("tolerance_s", 1e-6);    // Tolerance for convergence (step length)
    const double initial_step = datafile("initial_step", 1.0);   // Initial step size αlpha0
    const int max_iterations = datafile("max_iterations", 1000); // Maximal number of iterations
    const double mu = datafile("mu", 0.2);                       // Parameter for the exponential and inverse decay
    std::vector<double> initial_condition(N, 0);                 // Initial condition
    if (N > 0)
    {
        // If entries exist, parse them
        initial_condition.resize(N);
        for (int i = 0; i < N; ++i)
        {
            initial_condition[i] = datafile("initial_condition", 0.0, i); // default 0.0 if not found
        }
    }
    else
    {
        // Default values
        initial_condition = {0., 0.};
    }

    // Heavy ball specific paramters
    const double minimum_step = datafile("minimum_step", 1e-2); // Minimum step size
    const double eta = datafile("eta", 0.9);                    // Memory parameter

    // if user prefers to use the approximate gradient
    HeavyBallParams params;
    if (fd)
    {
        const std::string fd_t = datafile("fd_t", "Centered");
        const double h = datafile("h", 1e-2);
        std::cout << "Finite differences type: " << fd_t << " (h = " << h << ")" << std::endl;
        std::function<std::vector<double>(const std::vector<double> &)> df_grad_f;
        if (fd_t == "Forward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Forward>(f, h);
        }
        else if (fd_t == "Backward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Backward>(f, h);
        }
        else
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Centered>(f, h);
        }
        params = {
            f,
            df_grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            mu,
            eta,
        };
    }
    else
    {
        const std::string grad_f_str = datafile("grad_f", "{16*x[0]*x[0]*x[0] + 2*x[1] +2, 4*x[1]+2*x[0]}"); // Gradient of f
        muParserXInterface grad_f(grad_f_str, N);                                                            // Initialize the gradient with muparserx
        // Parameters for the algorithm
        params = {
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
    }
    return params;
}

NesterovParams read_nesterov(const GetPot &datafile)
{
    const int N = datafile.vector_variable_size("initial_condition");                                      // Dimension of the problem
    const std::string f_str = datafile("f", "4*x[0]*x[0]*x[0]*x[0] + 2*x[1]*x[1] + 2*x[0]*x[1] + 2*x[0]"); // Function f
    std::cout << "Function to be optimized: " << f_str << std::endl;
    muParserXScalarInterface f(f_str, N);                        // Initialize the function with muparserx
    const bool fd = datafile("fd", true);                        // Use finite differences to compute the gradient
    const double tolerance_r = datafile("tolerance_r", 1e-6);    // Tolerance for convergence (residual)
    const double tolerance_s = datafile("tolerance_s", 1e-6);    // Tolerance for convergence (step length)
    const double initial_step = datafile("initial_step", 1.0);   // Initial step size αlpha0
    const int max_iterations = datafile("max_iterations", 1000); // Maximal number of iterations
    const double mu = datafile("mu", 0.2);                       // Parameter for the exponential and inverse decay
    std::vector<double> initial_condition(N, 0);                 // Initial condition
    if (N > 0)
    {
        // If entries exist, parse them
        initial_condition.resize(N);
        for (int i = 0; i < N; ++i)
        {
            initial_condition[i] = datafile("initial_condition", 0.0, i); // default 0.0 if not found
        }
    }
    else
    {
        // Default values
        initial_condition = {0., 0.};
    }

    // Nesterov specific paramters
    const double minimum_step = datafile("minimum_step", 1e-2); // Minimum step size
    const double eta = datafile("eta_nest", 0.9);               // Memory parameter

    // if user prefers to use the approximate gradient
    NesterovParams params;
    if (fd)
    {
        const std::string fd_t = datafile("fd_t", "Centered");
        const double h = datafile("h", 1e-2);
        std::cout << "Finite differences type: " << fd_t << " (h = " << h << ")" << std::endl;
        std::function<std::vector<double>(const std::vector<double> &)> df_grad_f;
        if (fd_t == "Forward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Forward>(f, h);
        }
        else if (fd_t == "Backward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Backward>(f, h);
        }
        else
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Centered>(f, h);
        }
        params = {
            f,
            df_grad_f,
            initial_condition,
            tolerance_r,
            tolerance_s,
            initial_step,
            max_iterations,
            minimum_step,
            mu,
            eta,
        };
    }
    else
    {
        const std::string grad_f_str = datafile("grad_f", "{16*x[0]*x[0]*x[0] + 2*x[1] +2, 4*x[1]+2*x[0]}"); // Gradient of f
        muParserXInterface grad_f(grad_f_str, N);                                                            // Initialize the gradient with muparserx
        // Parameters for the algorithm
        params = {
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
    }
    return params;
}

AdamParams read_adam(const GetPot &datafile)
{
    const int N = datafile.vector_variable_size("initial_condition");                                      // Dimension of the problem
    const std::string f_str = datafile("f", "4*x[0]*x[0]*x[0]*x[0] + 2*x[1]*x[1] + 2*x[0]*x[1] + 2*x[0]"); // Function f
    std::cout << "Function to be optimized: " << f_str << std::endl;
    muParserXScalarInterface f(f_str, N);                        // Initialize the function with muparserx
    const bool fd = datafile("fd", true);                        // Use finite differences to compute the gradient
    const double tolerance_r = datafile("tolerance_r", 1e-6);    // Tolerance for convergence (residual)
    const double tolerance_s = datafile("tolerance_s", 1e-6);    // Tolerance for convergence (step length)
    const double initial_step = datafile("initial_step", 1.0);   // Initial step size αlpha0
    const int max_iterations = datafile("max_iterations", 1000); // Maximal number of iterations
    const double minimum_step = datafile("minimum_step", 1e-6);  // Minimum step size
    const double mu = datafile("mu", 0.2);                       // Parameter for the exponential decay and the inverse decay

    std::vector<double> initial_condition(N, 0); // Initial condition
    if (N > 0)
    {
        // If entries exist, parse them
        initial_condition.resize(N);
        for (int i = 0; i < N; ++i)
        {
            initial_condition[i] = datafile("initial_condition", 0.0, i); // default 0.0 if not found
        }
    }
    else
    {
        // Default values
        initial_condition = {0., 0.};
    }

    // Adam specific paramters
    const double beta1 = datafile("beta1", 0.9);   // Exponential decay rate for 1st moment estimate
    const double beta2 = datafile("beta2", 0.999); // Exponential decay rate for 2nd moment estimate

    // if user prefers to use the approximate gradient
    AdamParams params;
    if (fd)
    {
        const std::string fd_t = datafile("fd_t", "Centered");
        const double h = datafile("h", 1e-2);
        std::cout << "Finite differences type: " << fd_t << " (h = " << h << ")" << std::endl;
        std::function<std::vector<double>(const std::vector<double> &)> df_grad_f;
        if (fd_t == "Forward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Forward>(f, h);
        }
        else if (fd_t == "Backward")
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Backward>(f, h);
        }
        else
        {
            df_grad_f = gradient<decltype(f), double, DifferenceType::Centered>(f, h);
        }
        params = {
            f,
            df_grad_f,
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
    }
    else
    {
        const std::string grad_f_str = datafile("grad_f", "{16*x[0]*x[0]*x[0] + 2*x[1] +2, 4*x[1]+2*x[0]}"); // Gradient of f
        muParserXInterface grad_f(grad_f_str, N);                                                            // Initialize the gradient with muparserx
        // Parameters for the gradient descent algorithm
        params = {
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
    }
    return params;
}