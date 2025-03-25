


# Minimization of a multivariate function

## Setup
After cloning the repository with:

```bash
git clone git@github.com:alessandropedone/real-function-optimization.git
```

Once you are in the right directory, you can install [*muparserx*](https://github.com/beltoforion/muparserx) (used to parse information from the user) and [*TBB*](https://github.com/uxlfoundation/oneTBB) (used for parallel execution) libraries by running the provided little script with the command.

```bash
./install_pkgs.sh
```

You can find more information about TBB on [Wikipedia](https://en.wikipedia.org/wiki/Threading_Building_Blocks) and about muparserx on its [website](https://beltoforion.de/en/muparserx/).

You can uninstall all the required packages running the other bash script running the following command.

```bash
./uninstall_pkgs.sh
```

## Implemented Methods

We implemented the following optimization algorithms.

Basic method:
1. [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) (GD)

Momentum-based methods:

2. Heavy Ball (or Momentum) Method (HB)
3. Nesterov Accelerated Gradient (NAG)

Adaptive methods:

4. ADAM Method (it combines momentum the benefits of momentum and the adaptiveness of _RMSProp method_)

Second order methods:

5. [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
6. [Quasi-Newton Methods](https://en.wikipedia.org/wiki/Quasi-Newton_method)

Derivative-Free Optimization (DFO) methods:

7. [Powell's Method](http://en.wikipedia.org/wiki/Powell%27s_method)
8. [Nelder-Mead Method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) (Downhill Simplex)

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
- **Solvers Design**: each solver is implemented as a **functor** and its state corresponds to the method's parameters, allowing for modular and reusable design.
- **Template Usage**: template programming is used to manage method choices and gradient computation strategies efficiently, as the set of choices is finite and limited in size.

## Relevant files

### `data.txt`
`data.txt` is used with the aim to take as input all information from the user, exploiting GetPot and muparsex combined.
The explanation of all input parameters is provided directly inside data.txt as comments.
This structured file facilitates testing of different optimization scenarios (without recompiling anything).

### `muparser_interface.hpp`
It's an interface developed to parse functions (also vector functions) of an arbitrary number of variables. It's adapted from `muParserInterface` inside [`pacs-examples`](https://github.com/pacs-course/pacs-examples.git) repository.

### `fd_gradient.hpp`
WIP

## Future development: parallelization
Parallelizable implementations: `muparser_interface`, `fd_gradient`.

## Authors
- Marta Pignatelli ([@martapignatelli](https://github.com/martapignatelli))
- Alessandro Pedone ([@alessandropedone](https://github.com/alessandropedone))



