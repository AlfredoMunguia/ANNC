
#include<utils/ActivationFunctions.hpp>
#include <math.h>

namespace utils{
  
  Scalar activationFunction(Scalar x)
  {
    return tanhf(x);
  }
  
  Scalar activationFunctionDerivative(Scalar x)
  {
    return 1 - tanhf(x) * tanhf(x);
  }

}
