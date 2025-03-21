#include <iostream>
#include <vector>
#include <GetPot>

#include "gradient_descent.hpp"
#include "muparserx_interface.hpp"
#include "bivar_gradient.hpp"
#include "read.hpp"
#include "run.hpp"
#include "heavy_ball.hpp"
#include "nesterov.hpp"
#include "adam.hpp"

// This is a simple example of how to use the gradient_descent function.
int main()
{
    const GetPot datafile("data.txt");   

    const bool gradient_descent = datafile("gradient_descent", "true");
    if (gradient_descent)
    {
        // Read gradient descent parameters
        std::cout << "GRADIENT DESCENT" << std::endl;
        GradientDescentParams params_gd = read_gradient_descent(datafile);
        
        // Run the gradient descent algorithm with the chosen strategy
        const std::string gradient_method_t = datafile("gradient_method_t", "Armijo rule"); // Rule to compute the step size for the gradient descent algorithm
        run_gradient_descent(params_gd, gradient_method_t);
    }

    const bool heavy_ball = datafile("heavy_ball", "true");
    if (heavy_ball)
    {
        // Read heavy ball parameters
        std::cout << "HEAVY BALL" << std::endl;
        HeavyBallParams params_hb = read_heavy_ball(datafile);
        
        // Run the heavy ball algorithm with the chosen strategy
        const std::string heavy_ball_t = datafile("heavy_ball_t", "Exponential decay"); // Rule for the step size
        const std::string heavy_ball_s = datafile("heavy_ball_s", "Constant");          // Rule for eta
        run_heavy_ball(params_hb, heavy_ball_t, heavy_ball_s);
    }

    const bool nesterov = datafile("nesterov", "true");
    if (nesterov)
    {
        // Read Nesterov parameters
        std::cout << "NESTEROV" << std::endl;
        NesterovParams params_n = read_nesterov(datafile);
        
        // Run the nesterov algorithm with the chosen strategy
        const std::string nesterov_t = datafile("nesterov_t", "Exponential decay"); // Rule for the step size
        const std::string nesterov_s = datafile("nesterov_s", "Constant");          // Rule for eta
        run_nesterov(params_n, nesterov_t, nesterov_s);
    }

    const bool adam = datafile("adam", "true");
    if (adam)
    {
        // Read Adam parameters
        std::cout << "ADAM" << std::endl;
        AdamParams params_n = read_adam(datafile);
        
        // Run the adam algorithm with the chosen strategy
        const std::string adam_t = datafile("adam_t", "Exponential decay"); // Rule for the step size
        run_adam(params_n, adam_t);
    }

    return 0;
}