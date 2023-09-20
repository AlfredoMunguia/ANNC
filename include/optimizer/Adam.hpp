#include <Eigen/Eigen>
#include <optimizer/Optimizer.hpp>
#include <math.h>

namespace optimizer{
  
#ifndef ADAM_H
#define ADAM_H


  using namespace fterm;
  using namespace std;
  
  template <typename T>
  class Adam: public Optimizer<T>{
    
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;


    vector<RowVectorXT*>  _mtb;
    vector<MatrixXXT*>    _mtW;

    vector<RowVectorXT*>  _vtb;
    vector<MatrixXXT*>    _vtW;

    vector<RowVectorXT*>  _hmtb;
    vector<MatrixXXT*>    _hmtW;

    vector<RowVectorXT*>  _hvtb;
    vector<MatrixXXT*>    _hvtW;

    vector<RowVectorXT*> *_gradb;
    vector<MatrixXXT*>*   _gradW;

    vector<int>*          _architecture;
    
    T _beta1;
    T _beta2;

    T _epsilon;
    
  public:
    Adam(NeuralNetwork<T>*);
    void update(vector<RowVectorXT*>*,
                vector<MatrixXXT*>*,int);
    
  };

  
#include "../src/optimizer/Adam.cpp"
  
  
#endif
  
}
