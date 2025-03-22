# Minimization of a multivariate function

## Setup
After cloning the repository with:

```bash
git clone --recurse-submodules git@github.com:alessandropedone/real-function-optimization.git
```

Once you are in the right directoryu, you can compile the `muparserx` library by running the provided little script with the command:

```bash
./setup.sh
```

If you encounter an error such as `cmake: command not found`, you can run `sudo apt install cmake` if you are on Debian, Ubuntu, Linux Mint and related distributions (similar commands can be found online for other distributions).


## Implemented Methods
We implemented the following optimization algorithms.
Basic method:
1. Gradient Descent (GD)
Momentum-based methods:
2. Heavy Ball (or Momentum) Method (HB)
3. Nesterov Accelerated Gradient (NAG)
Adaptive methods:
4. ADAM Method (it combines momentum the benefits of momentum and the adaptiveness of _RMSProp method_)
Second order methods:
5. Newton's Method
6. Quasi-Newton Methods
Derivative-Free Optimization (DFO) methods:
7. Powell's Method
8. Nelder-Mead (Downhill Simplex)


## Enhancements and Optimizations
The following optimizations, that we decided to implement, improve the convergence speed and stability of all methods.
- **Gradient Normalization**: prevents large updates and ensures stable convergence.
- **Minimum Step Size Constraint**: avoids excessively small steps, guaranteeing progress toward convergence.

## Test Case
The implemented methods were validated on the following test function (which has a very steep gradient, also near its minimum):

$$
f(x_1, x_2) = 4x_1^4 + 2x_2^2 + 2x_1x_2 + 2x_1
$$

For each execution of a method, the following information is printed:
- number of iterations to convergence (if it converged before the maximum number of iterations specified);
- final values of $x$ (the estimated minimum) and $f(x)$ (the value that the function takes at the estemated minimum);
- norm of the gradient at the minimum point;
- method-specific parameters used during the run.

## Gradient Evaluation Options
Two options are available:
1. **Exact Gradient (Analytical)**: it must be specified in data.txt.
2. **Finite Differences (FD)**: it is available in the following three variations.
   - Forward Differences
   - Backward Differences
   - Centered Differences

## Design Decisions
- **Solver Design**: the solver is implemented as a **functor** and its state corresponds to the method's parameters, allowing for modular and reusable design.
- **Template Usage**: template programming is used to manage method choices and gradient computation strategies efficiently, as the set of choices is finite and limited in size.

## `data.txt`
`data.txt` is used with the aim to take as input all information from the user, exploiting GetPot and muparsex combined.
The explanation of all input parameters is provided directly inside data.txt as comments.
This structured file facilitates testing of different optimization scenarios (without recompiling anything).

**NOTE**: the default value of `max_iterations` is set to $1000$ to observe the **convergence of ADAM Method** too, but it's important to note that GD converges in **less than $100$** iterations with each one of the three possible strategies to compute the step.

## `muparser_interface.hpp`
It's an interface developed to parse functions (also vector functions) of an arbitrary number of variables. It's adapted from `muParserInterface` inside [`pacs-examples`](https://github.com/pacs-course/pacs-examples.git) repository.

## Authors
- Marta Pignatelli ([@martapignatelli](https://github.com/martapignatelli))
- Alessandro Pedone ([@alessandropedone](https://github.com/alessandropedone))



