#include <fterm/FTerm.hpp>
#include <Eigen/Eigen>
#include <nn/NeuralNetwork.hpp>

namespace optimizer{
  
  using namespace fterm;
  using namespace std;
  
  
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

  template <typename T>
  class Optimizer{

    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;

  protected:
    FTerm<T>* _loss;
    NeuralNetwork<T>* _nn;
    virtual void update(vector<RowVectorXT*>*,
                        vector<MatrixXXT*>*,int) = 0;

  };
  
  
// #include "../src/optimizer/Optimizer.cpp"
  
  
#endif
  
}
