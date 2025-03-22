#ifndef READ_HPP
#define READ_HPP

#include "gradient_descent.hpp"
#include "GetPot"
#include "muparserx_interface.hpp"
#include "fd_gradient.hpp"
#include "heavy_ball.hpp"
#include "nesterov.hpp"
#include "adam.hpp"

//Read the parameters for the gradient descent method from a GetPot object
GradientDescentParams read_gradient_descent(const GetPot &datafile);

//Read the parameters for the heavy ball method from a GetPot object
HeavyBallParams read_heavy_ball(const GetPot &datafile);

//Read the parameters for the nesterov method from a GetPot object
NesterovParams read_nesterov(const GetPot &datafile);

//Read the parameters for the adam method from a GetPot object
AdamParams read_adam(const GetPot &datafile);

#endif //READ_HPP