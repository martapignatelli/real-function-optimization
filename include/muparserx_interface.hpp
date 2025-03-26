#ifndef MUPARSERX_INTERFACE_HPP
#define MUPARSERX_INTERFACE_HPP

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mpParser.h>

// Adapted from muParserInterface inside pacs-examples repository (https://github.com/pacs-course/pacs-examples.git)
// The class that will be used to parse functions
class muParserXInterface
{
public:
    //! Default constructor
    //!
    //! mup::pckALL_NON_COMPLEX|mup::pckMATRIX means that I do not want the module
    //! for complex numbers but I want to treat arrays and matrices in muparserX
    //! expressions
    //! N is not a template parameter and is set at run time in order to achive more flexibility
    muParserXInterface(const unsigned N = 1)
        : My_e(),
          M_parser(mup::pckALL_NON_COMPLEX | mup::pckMATRIX), M_value{N, 0.0}, N(N)
    {
        M_parser.DefineVar("x", mup::Variable(&M_value));
    }
    //! Constructor that takes a string containing muParserX expression
    muParserXInterface(const std::string expression, const unsigned N = 1) : muParserXInterface(N)
    {
        try
        {
            My_e = expression;
            M_parser.SetExpr(My_e.c_str());
        }
        catch (mup::ParserError &error)
        {
            std::cerr << "Muparsex error with code:" << error.GetCode()
                      << std::endl;
            std::cerr << "While processing expression: " << error.GetExpr()
                      << std::endl;
            std::cerr << "Error Message: " << error.GetMsg() << std::endl;
            throw error;
        }
    }

    /*!
     * The copy constructor
     *
     * MuparserX has a particular design, which obliges to define a special copy
     * constructor The reson is that a muparser engine stores the address of the
     * variables. So a normal copy would do a shallow copy, which is NOT what you
     * want. Moreover, because of a poor design, you may loose the expression.
     * That's why I keep a copy in the class as a string and a redefine in in the
     * muparser engine.
     *
     * @param mpi the muParserXIterface to be copied
     */
    muParserXInterface(muParserXInterface const &mpi)
        : My_e(mpi.My_e),
          M_parser(mup::pckALL_NON_COMPLEX | mup::pckMATRIX), M_value{mpi.N, 0.0}, N(mpi.N)
    {
        M_parser.DefineVar("x", mup::Variable(&M_value));
        M_parser.SetExpr(My_e.c_str());
    }

    /*!
     * The copy assignment operator
     *
     * MuparserX has a particular design, which obliges to define a special copy
     * assignement
     * @param mpi the muParserXInterface to be copied
     * The copy constructor
     */
    muParserXInterface
    operator=(muParserXInterface const &mpi)
    {
        if (this != &mpi)
        {
            this->My_e = mpi.My_e;
            this->M_parser.ClearVar(); // clear the variables!
            this->M_value = mpi.M_value;
            this->N = mpi.N;
            M_parser.DefineVar("x", mup::Variable(&M_value));
            M_parser.SetExpr(My_e.c_str());
        }
        return *this;
    }

    //! Sets the muparserX expression.
    /*!
     * Beware, the input variables are indicated by x[].
     * example of a valid expression: sin(x[0])+x[1]*x[2]
     * @par e The expression
     */
    void
    set_expression(const std::string &e)
    {
        My_e = e;
        M_parser.SetExpr(e.c_str());
    }

    //! Function call operator to evaluate the expression
    /*!
     * Takes a vector of input variables, evaluates the expression, and returns the result.
     * Supports both scalar and matrix outputs.
     *
     * @param x Vector of input variable values.
     * @return Vector of evaluated results.
     */
    Eigen::VectorXd operator()(const Eigen::VectorXd &x) const
    {
        // Assign input values to the parser's variable storage
        for (unsigned i = 0; i < N; ++i)
        {
            M_value.At(i) = x(i);
        }

        mup::Value val;
        Eigen::VectorXd vec;
        try
        {
            // Evaluate the parsed expression
            val = M_parser.Eval();

            if (val.IsScalar())
            {
                // If the result is a scalar, return a single-element vector
                vec = Eigen::VectorXd(1);
                vec(0) = val.GetFloat();
            }
            else
            {
                // If the result is a matrix, extract its values
                unsigned rows = val.GetRows();
                unsigned cols = val.GetCols();
                vec = Eigen::VectorXd(rows * cols);

                for (unsigned i = 0; i < rows; ++i)
                {
                    for (unsigned j = 0; j < cols; ++j)
                    {
                        vec(i * cols + j) = val.At(i, j).GetFloat();
                    }
                }
            }
        }
        catch (mup::ParserError &error)
        {
            // Handle parsing errors
            std::cerr << "Muparsex error with code:" << error.GetCode() << std::endl;
            std::cerr << "While processing expression: " << error.GetExpr() << std::endl;
            std::cerr << "Error Message: " << error.GetMsg() << std::endl;
            throw error;
        }
        return vec;
    }

protected:
    // a copy of the muparserX expression, used for the copy operations
    std::string My_e;
    // The muparseX engine
    mup::ParserX M_parser;
    // The muparserX value used to set the variables in the engine
    mutable mup::Value M_value;
    mutable unsigned N;
};

/*!
 * Specialized class for scalar-valued functions
 *
 * Inherits from muParserXInterface, simplifies evaluation by returning a single scalar result.
 */
class muParserXScalarInterface : public muParserXInterface
{
public:
    //! Constructor
    /*!
     * Initializes scalar interface with an expression and number of variables.
     *
     * @param expression The muParserX expression string.
     * @param N Number of input variables.
     */
    muParserXScalarInterface(const std::string expression, const unsigned N = 1) : muParserXInterface(expression, N) {}

    //! Function call operator
    /*!
     * Evaluates the scalar expression with given input variables.
     *
     * @param x Vector of input variable values.
     * @return Single scalar result.
     */
    double operator()(const Eigen::VectorXd &x) const
    {
        Eigen::VectorXd vec = muParserXInterface::operator()({x});
        return vec.size() == 0 ? 0.0 : vec(0);
    }
};

#endif // MUPARSERX_INTERFACE_HPP
