#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <nn/NeuralNetwork.h>

namespace fterm{
  
  using namespace std;
  using namespace Eigen;
  using namespace nn;
  
#ifndef FTERM_H
#define FTERM_H
  
  template <typename T>
  class FTerm{

    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;

  protected:
    
    NeuralNetwork<T>* _net;
    
  public:

    FTerm(NeuralNetwork<T>* net);
    // void setCTrainData(MatrixXXT*, MatrixXXT*,
    //                    MatrixXXT*, MatrixXXT*, int);


    
    virtual Matrix<T,1,Dynamic>* biasGradient(int) = 0;
    virtual Matrix<T,Dynamic,Dynamic>* weightGradient(int) = 0;
    
    virtual void forward()  = 0;
    virtual void backward() = 0;
    virtual void update()   = 0;
    virtual void train() = 0;
    virtual void setTrainData() = 0;
    virtual void setTrainData(MatrixXXT*,MatrixXXT* = 0) = 0;
    virtual    T cost() = 0;
    // virtual void gradient() = 0;

    // virtual void setAlpha(T) = 0;
    
  };
  
#include "../src/fterm/FTerm.cpp"
  
#endif
  
}
