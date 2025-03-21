//BIVARIATE VERSION OF FINITE DIFFERENCES GRADIENT

#ifndef BIVAR_GRADIENT_HPP
#define BIVAR_GRADIENT_HPP

#include <functional>
#include <type_traits> // for std::is_same_v

namespace bivar_gradient{

  /*
  This namespace defines the directions of differences
  I alternate among different types when performing higher
  derivatives, that explains the use of otherType
  */

  namespace DifferenceType
    {
      struct BACKWARD; // forward declaration (to enable the usage of BACKWARD in FORWARD)
      struct FORWARD
      {
        using otherType = BACKWARD;
      };
      struct BACKWARD
      {
        using otherType = FORWARD;
      };

      struct CENTERED
      {
        using otherType = CENTERED;
      };
    } // namespace DifferenceType


  /*
  This function computes the N-th derivative of a BIVARIATE function
  by finite differences using lambda expressions 

  N is the order of the derivative
  F the callable object of signature T (std::vector<T>)
  T The argument and return type of the callable object
  DT is the difference type: FORWARD, BACKWARD or CENTERED
  h is the step for computing the derivative

  * Example of usage:
    *
    * f = auto [](std::vector<double> x){return std::sin(x[0])+std::sin(x[1]);}
    * auto d4 = derive<1, std::function<double(std::vector<double>)>, double, DifferenceType::CENTERED>(f, 1.e-4);
    * auto d  = d1(x0);// gradient of f in x0 (x0 is a vector of size 2)
    *
    * Adapted from pacs-examples (from Discovering Modern C++, II Edition, by Peter Gottschling)
    *
  */

  template <unsigned int N, typename F, typename T, typename DT>
  constexpr auto derive_2D =[](F f, double h)
  {
    if constexpr (N==0u)
    {
      return [f](std::vector<T> x){return std::vector<T> {f(x), f(x)};};
    }
    else
    {
      auto prevFuncs = derive_2D<N-1u, F, T, typename DT::otherType>(f,h);
      std::vector<T> h1{h, 0}, h2{0, h}; //increment vectors in both directions

      if constexpr (std::is_same_v<DT, DifferenceType::FORWARD>)
        {
          return [=](std::vector<T> x)
            {return std::vector<T> {(prevFuncs(x + h1)[0] - prevFuncs(x)[0]) / h, (prevFuncs(x + h2)[1] - prevFuncs(x)[1]) / h};};
        }
      else if constexpr (std::is_same_v<DT, DifferenceType::BACKWARD>)
        {
          return [=](std::vector<T> x)
            {return std::vector<T> {(prevFuncs(x)[0] - prevFuncs(x - h1)[0]) / h, (prevFuncs(x)[1] - prevFuncs(x - h2)[1]) / h};};
        }
      else
        {
          return [=](std::vector<T> x)
            {return std::vector<T> {(prevFuncs(x + h1)[0] - prevFuncs(x - h1)[0]) / (2.*h), (prevFuncs(x + h2)[1] - prevFuncs(x - h2)[1]) / (2.*h)};};    
        };

    };

  };

} // namespace bivar_gradient

#endif // BIVAR_GRADIENT_HPP

