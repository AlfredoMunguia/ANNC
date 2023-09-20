#include <Eigen/Eigen> 
#include <vector> 
#include <domain/Domain.h>

namespace domain{

  using namespace Eigen;
  
  template <typename T>
  class TwoDimDomain : public Domain<T>{    

  private:
    // typedef  Matrix<T,1,Dynamic> RowVectorXT;
    // typedef  Matrix<T,2,Dynamic> TwoRowVectorXT;

    typedef  Matrix<T,Dynamic,Dynamic> MatrixXXT;
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    

  public:
    
    TwoDimDomain(){};
    TwoDimDomain(int nIntP, int nBdyP, int nEvalP):Domain<T>(nIntP,nBdyP,nEvalP){};
    
    virtual void sampleFromInner(MatrixXXT &) = 0;
    virtual void sampleFromBoundary(MatrixXXT&,  MatrixXXT&, MatrixXXT&,  MatrixXXT&) = 0;
    virtual void sampleFromBoundary(MatrixXXT&,  MatrixXXT&) = 0;
    virtual void sampleFromBoundary(MatrixXXT&) = 0;
   
    
    virtual T area() = 0;
    virtual T perimeter() = 0;  
    
  };

}
