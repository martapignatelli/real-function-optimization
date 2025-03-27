#include "run.hpp"

/**
 * @brief Prints the optimization results.
 *
 * @param minimum The computed minimum vector.
 * @param f The scalar function being minimized.
 * @param grad_f The gradient of the scalar function.
 */
void print_result(const vector_type &minimum, scalar_function f, vector_function grad_f)
{
    Eigen::IOFormat commaFormat(Eigen::StreamPrecision, 0, ",", ",", "", "", "(", ")");
    std::cout << "Computed minimum: " << minimum.format(commaFormat) << std::endl;
    std::cout << "f " << minimum.format(commaFormat) << " = " << f(minimum) << std::endl;
    std::cout << "|| grad_f " << minimum.format(commaFormat) << " || = " << grad_f(minimum).norm() << std::endl
              << std::endl;
}

/**
 * @brief Runs the solver and prints the results.
 *
 * @param solver The optimization method to use.
 */
void run_solver(const Method &solver)
{
    solver.print();
    vector_type minimum = solver();
    print_result(minimum, solver.get_f(), solver.get_grad_f());
}

/**
 * @brief Runs the specified optimization method based on given parameters.
 *
 * @param params The parameters for the optimization method.
 * @param method_t The primary optimization method type (e.g., "Exponential decay", "Inverse decay").
 * @param method_s (Optional) The secondary strategy for some methods (e.g., "Dynamic", "Constant").
 */
void run(const Params &params, const string_type &method_t, const string_type &method_s)
{
    if (dynamic_cast<const GradientDescentParams *>(&params) != nullptr)
    {
        const auto *p = dynamic_cast<const GradientDescentParams *>(&params);
        if (method_t == "Exponential decay")
        {
            // Runs gradient descent with exponential step decay.
            GradientDescent<GradientDescentType::exponential> solver(*p);
            run_solver(solver);
        }
        else if (method_t == "Inverse decay")
        {
            // Runs gradient descent with inverse step decay.
            GradientDescent<GradientDescentType::inverse> solver(*p);
            run_solver(solver);
        }
        else if (method_t == "Armijo rule")
        {
            // Runs gradient descent using the Armijo rule for step size selection.
            GradientDescent<GradientDescentType::armijo> solver(*p);
            run_solver(solver);
        }
        else
        {
            std::cerr << "Invalid gradient method type" << std::endl;
        }
    }
    else if (dynamic_cast<const HeavyBallParams *>(&params) != nullptr)
    {
        const auto *p = dynamic_cast<const HeavyBallParams *>(&params);
        if (method_t == "Exponential decay")
        {
            if (method_s == "Dynamic")
            {
                // Runs heavy ball with exponential step decay and dynamic memory.
                HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs heavy ball with exponential step decay and constant memory.
                HeavyBall<HeavyBallType::exponential, HeavyBallStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
        else if (method_t == "Inverse decay")
        {
            if (method_s == "Dynamic")
            {
                // Runs heavy ball with inverse step decay and dynamic memory.
                HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs heavy ball with inverse step decay and constant memory.
                HeavyBall<HeavyBallType::inverse, HeavyBallStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
        else if (method_t == "Constant")
        {
            if (method_s == "Dynamic")
            {
                // Runs heavy ball with constant step size and dynamic memory.
                HeavyBall<HeavyBallType::constant, HeavyBallStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs heavy ball with constant step size and constant memory.
                HeavyBall<HeavyBallType::constant, HeavyBallStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
    }
    else if (dynamic_cast<const NesterovParams *>(&params) != nullptr)
    {
        const auto *p = dynamic_cast<const NesterovParams *>(&params);
        if (method_t == "Exponential decay")
        {
            if (method_s == "Dynamic")
            {
                // Runs Nesterov's method with exponential step decay and dynamic memory.
                Nesterov<NesterovType::exponential, NesterovStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs Nesterov's method with exponential step decay and constant memory.
                Nesterov<NesterovType::exponential, NesterovStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
        else if (method_t == "Inverse decay")
        {
            if (method_s == "Dynamic")
            {
                // Runs Nesterov's method with inverse step decay and dynamic memory.
                Nesterov<NesterovType::inverse, NesterovStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs Nesterov's method with inverse step decay and constant memory.
                Nesterov<NesterovType::inverse, NesterovStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
        else if (method_t == "Constant")
        {
            if (method_s == "Dynamic")
            {
                // Runs Nesterov's method with constant step size and dynamic memory.
                Nesterov<NesterovType::constant, NesterovStrategy::dynamic> solver(*p);
                run_solver(solver);
            }
            else if (method_s == "Constant")
            {
                // Runs Nesterov's method with constant step size and constant memory.
                Nesterov<NesterovType::constant, NesterovStrategy::constant> solver(*p);
                run_solver(solver);
            }
            else
            {
                std::cerr << "Invalid method type" << std::endl;
                return;
            }
        }
    }
    else if (dynamic_cast<const AdamParams *>(&params) != nullptr)
    {
        const auto *p = dynamic_cast<const AdamParams *>(&params);
        if (method_t == "Dynamic")
        {
            // Runs Adam with dynamic step size.
            Adam<AdamType::dynamic> solver(*p);
            run_solver(solver);
        }
        else if (method_t == "Constant")
        {
            // Runs Adam with constant step size.
            Adam<AdamType::constant> solver(*p);
            run_solver(solver);
        }
        else
        {
            std::cerr << "Invalid method type" << std::endl;
        }
    }
}
