#include "run.hpp"

// Prints the computed minimum, function value at the minimum, and gradient norm at the minimum.
void print_result(const Eigen::VectorXd &minimum, std::function<double(const Eigen::VectorXd &)> f, // Function f
                  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f)
{
    std::cout << "Computed minimum: (";
    for (size_t i = 0; i < minimum.size(); ++i)
    {
        std::cout << minimum[i];
        if (i < minimum.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::cout << "f(";
    for (size_t i = 0; i < minimum.size(); ++i)
    {
        std::cout << minimum[i];
        if (i < minimum.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << ") = " << f(minimum) << std::endl;
    std::cout << "||grad_f(";
    for (size_t i = 0; i < minimum.size(); ++i)
    {
        std::cout << minimum[i];
        if (i < minimum.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << ")|| = " << grad_f(minimum).norm() << std::endl
              << std::endl;
}

// Runs the gradient descent algorithm with specified parameters and step size method.
void run_gradient_descent(const GradientDescentParams &params, const std::string &gradient_method_t)
{
    if (gradient_method_t == "Exponential decay")
    {
        // Runs gradient descent with exponential step decay.
        GradientDescent<GradientDescentType::exponential> exponential_solver(params);
        exponential_solver.print();
        Eigen::VectorXd minimum_exponential = exponential_solver();
        print_result(minimum_exponential, params.f, params.grad_f);
    }
    else if (gradient_method_t == "Inverse decay")
    {
        // Runs gradient descent with inverse step decay.
        GradientDescent<GradientDescentType::inverse> inverse_solver(params);
        inverse_solver.print();
        Eigen::VectorXd minimum_inverse = inverse_solver();
        print_result(minimum_inverse, params.f, params.grad_f);
    }
    else if (gradient_method_t == "Armijo rule")
    {
        // Runs gradient descent using the Armijo rule for step size selection.
        GradientDescent<GradientDescentType::armijo> armijo_solver(params);
        armijo_solver.print();
        Eigen::VectorXd minimum_armijo = armijo_solver();
        print_result(minimum_armijo, params.f, params.grad_f);
    }
    else
    {
        std::cerr << "Invalid gradient method type" << std::endl;
    }
}

// Runs the heavy ball algorithm with specified parameters, step size method, and memory strategy.
void run_heavy_ball(const HeavyBallParams &params, const std::string &heavy_ball_t, const std::string &heavy_ball_s)
{
    if (heavy_ball_t == "Exponential decay")
    {
        if (heavy_ball_s == "Dynamic")
        {
            // Runs heavy ball with exponential step decay and dynamic memory.
            HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (heavy_ball_s == "Constant")
        {
            // Runs heavy ball with exponential step decay and constant memory.
            HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
    else if (heavy_ball_t == "Inverse decay")
    {
        if (heavy_ball_s == "Dynamic")
        {
            // Runs heavy ball with inverse step decay and dynamic memory.
            HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (heavy_ball_s == "Constant")
        {
            // Runs heavy ball with inverse step decay and constant memory.
            HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
    else if (heavy_ball_t == "Constant")
    {
        if (heavy_ball_s == "Dynamic")
        {
            // Runs heavy ball with constant step size and dynamic memory.
            HeavyBall<HeavyBallType::constant, HeavyBallStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (heavy_ball_s == "Constant")
        {
            // Runs heavy ball with constant step size and constant memory.
            HeavyBall<HeavyBallType::constant, HeavyBallStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
}

// Runs the nesterov algorithm with specified parameters, step size method, and memory strategy.
void run_nesterov(const NesterovParams &params, const std::string &nesterov_t, const std::string &nesterov_s)
{
    if (nesterov_t == "Exponential decay")
    {
        if (nesterov_s == "Dynamic")
        {
            // Runs nesterov with exponential step decay and dynamic memory.
            Nesterov<NesterovType::exponential, NesterovStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (nesterov_s == "Constant")
        {
            // Runs nesterov with exponential step decay and constant memory.
            Nesterov<NesterovType::exponential, NesterovStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
    else if (nesterov_t == "Inverse decay")
    {
        if (nesterov_s == "Dynamic")
        {
            // Runs nesterov with inverse step decay and dynamic memory.
            Nesterov<NesterovType::inverse, NesterovStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (nesterov_s == "Constant")
        {
            // Runs nesterov with inverse step decay and constant memory.
            Nesterov<NesterovType::inverse, NesterovStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
    else if (nesterov_t == "Constant")
    {
        if (nesterov_s == "Dynamic")
        {
            // Runs nesterov with constant step size and dynamic memory.
            Nesterov<NesterovType::constant, NesterovStrategy::dynamic> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else if (nesterov_s == "Constant")
        {
            // Runs nesterov with constant step size and constant memory.
            Nesterov<NesterovType::constant, NesterovStrategy::constant> solver(params);
            solver.print();
            Eigen::VectorXd minimum = solver();
            print_result(minimum, params.f, params.grad_f);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
            return;
        }
    }
}

void run_adam(const AdamParams &params, const std::string &adam_t)
{
    if (adam_t == "Dynamic")
    {
        // Runs adam with dynamic step size.
        Adam<AdamType::dynamic> solver(params);
        solver.print();
        Eigen::VectorXd minimum = solver();
        print_result(minimum, params.f, params.grad_f);
    }
    else if (adam_t == "Constant")
    {
        // Runs adam with constant step size.
        Adam<AdamType::constant> solver(params);
        solver.print();
        Eigen::VectorXd minimum = solver();
        print_result(minimum, params.f, params.grad_f);
    }
    else
    {
        std::cerr << "Invalid method type" << std::endl;
    }
}
