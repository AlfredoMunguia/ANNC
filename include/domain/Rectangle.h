#include <Eigen/Eigen> 
#include <vector> 
#include <domain/TwoDimDomain.h>

namespace domain{

  using namespace Eigen;
  using namespace std;
  
  template <typename T>
  class Rectangle : public TwoDimDomain<T>{    

  private:

    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;
    typedef Matrix<T,1,Dynamic> RowVectorXT;

    // typedef  Matrix<T,2,Dynamic> MatrixXXT;

    T _a;
    T _b;
    T _c;
    T _d;
    
  public:
    Rectangle();    

    Rectangle(T a, T b, T c, T d, int nEvalP, int nIntP=10, int nBdyP=10):
      TwoDimDomain<T>(nIntP,nBdyP,nEvalP),_a(a),_b(b),_c(c),_d(d){};

    Rectangle(T a, T b, T c, T d):_a(a),_b(b),_c(c),_d(d){};

    T area(){return (_b-_a)*(_d-_c);}
    T perimeter(){return _a+_b+_c+_d;}

    void sampleFromInner(MatrixXXT& X_train, MatrixXXT& X_test);
    void sampleFromInner(MatrixXXT& X_test);
    void sampleFromInner(MatrixXXT& X_test, int nIntP);
    
    void uniformSampleFromInner(MatrixXXT& X_test);
    void uniformSampleFromInner(MatrixXXT& X_test, int nP);
    void sampleFromBoundary(MatrixXXT& bdyL,  MatrixXXT& bdyB,MatrixXXT& bdyR,  MatrixXXT& bdyT);
    void sampleFromBoundary(MatrixXXT& bdyLB, MatrixXXT& bdyB);
    void sampleFromBoundary(MatrixXXT& bdy);
    void sampleFromBoundary(MatrixXXT& bdy, int nBdyP);
    void sampleFromBottomBoundary(MatrixXXT& bdy, int nBdyP);   

  };

  #include "../src/domain/Rectangle.cpp"

}
