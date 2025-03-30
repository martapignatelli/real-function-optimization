#include "read.hpp"
#include "readnew.hpp"
#include "run.hpp"


// This is a simple example of how to use the gradient_descent function.
int main()
{
    const GetPot datafile("data.txt"); 

    const bool gradient_descent = datafile("gradient_descent", "true");
    if (gradient_descent)
    {
        // Read gradient descent parameters
        std::cout << "GRADIENT DESCENT" << std::endl;        
        GradientDescentParams params_gd;
        read(datafile, params_gd);

        // Run the gradient descent algorithm with the chosen strategy
        const string_type gradient_method_t = datafile("gradient_method_t", "Armijo rule"); // Rule to compute the step size for the gradient descent algorithm
        run(params_gd, gradient_method_t, "");
    }

    const bool heavy_ball = datafile("heavy_ball", "true");
    if (heavy_ball)
    {
        // Read heavy ball parameters
        std::cout << "HEAVY BALL" << std::endl;
        HeavyBallParams params_hb;
        read(datafile, params_hb);
        
        // Run the heavy ball algorithm with the chosen strategy
        const string_type heavy_ball_t = datafile("heavy_ball_t", "Exponential decay"); // Rule for the step size
        const string_type heavy_ball_s = datafile("heavy_ball_s", "Constant");          // Rule for eta
        run(params_hb, heavy_ball_t, heavy_ball_s);
    }

    const bool nesterov = datafile("nesterov", "true");
    if (nesterov)
    {
        // Read Nesterov parameters
        std::cout << "NESTEROV" << std::endl;

        NesterovParams params_n;
        read(datafile, params_n);
        
        // Run the nesterov algorithm with the chosen strategy
        const string_type nesterov_t = datafile("nesterov_t", "Exponential decay"); // Rule for the step size
        const string_type nesterov_s = datafile("nesterov_s", "Constant");          // Rule for eta
        run(params_n, nesterov_t, nesterov_s);
    }

    const bool adam = datafile("adam", "true");
    if (adam)
    {
        // Read Adam parameters
        std::cout << "ADAM" << std::endl;

        AdamParams params_a;
        read(datafile, params_a);
        
        // Run the adam algorithm with the chosen strategy
        const string_type adam_t = datafile("adam_t", "Exponential decay"); // Rule for the step size
        run(params_a, adam_t, "");
    }

    return 0;
}