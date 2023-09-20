#include <Eigen/Eigen> 
#include <vector> 

namespace domain{
  using namespace Eigen;
  
  template <typename T>
  class Domain{
    
  protected:    

    typedef  Matrix<T,Dynamic,Dynamic> MatrixXXT;
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    
    int _nIntP;
    int _nBdyP;
    int _nEvalP;
    
  public:
    Domain(){};
    Domain(int nIntP, int nBdyP, int nEvalP):_nIntP(nIntP),
                                             _nBdyP(nBdyP),
                                             _nEvalP(nEvalP){};
    
    void setNIntP(int nIntP){_nIntP = nIntP;}
    void setNBdyP(int nBdyP){_nBdyP = nBdyP;}
    void setNEvalP(int nEvalP){_nEvalP = nEvalP;}
  };

}
