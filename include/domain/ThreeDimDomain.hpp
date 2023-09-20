#include <Eigen/Eigen> 
#include <vector> 

namespace domain{
  
  class Domain{

  private:    
    int nInt;
    int nBdy;
    int nEval;

  public:
    virtual float area() = 0;
    virtual float volume() = 0;
    virtual float perimeter() = 0;

  };
}
