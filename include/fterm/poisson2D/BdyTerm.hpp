#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <fterm/FTerm.hpp>
#include <functional>

namespace fterm{
  
#ifndef BDYTERM_H
#define BDYTERM_H

  using namespace std;
  using namespace Eigen;

  template <typename T>
  class BdyTerm: public FTerm<T>{
    
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;
    
    MatrixXXT                  *_X_train;
    MatrixXXT                  *_Y_train;
    Matrix<T,Dynamic,Dynamic>  *_CX_train;
    Matrix<T,Dynamic,Dynamic>  *_CY_train;
    
    int _nCTrain;
    int _nTrain;
  
    MatrixXXT                 _Lambda;

    vector<vector<MatrixXXT*> >   _vD;
    vector<vector<MatrixXXT* > >  _vB;
    vector<vector<RowVectorXT*> > _va;
    
    vector<RowVectorXT*>  _gradb;
    vector<MatrixXXT*>    _gradW;
    
    int _nInput;
    int _nL;
    T _perimeter;
    T _penalty;
    
    vector<int> _architecture;
    RowVectorXT _errors;
    
    //void getCXTrain();
    void init();

    function<T(T,T)> _g_function;
  public:

    BdyTerm(MatrixXXT *X_train, MatrixXXT *Y_train, NeuralNetwork<T>* net,
            T (*g_function)(T,T), T perimeter, T penalty ):
      FTerm<T>(net)
    {
      _X_train    = X_train;
      _Y_train    = Y_train;
      // _nCTrain    = nCTrain;
      _nTrain     = X_train -> cols();
      
      _g_function = g_function;
      _perimeter  = perimeter;
      _penalty    = penalty;
      

      _errors.resize(_nTrain);
      _vD.resize(_nTrain);
      _vB.resize(_nTrain);
      _va.resize(_nTrain);

      init();
    };
    
    void forward();
    void backward();
    void update();
    void train();

    void setTrainData(MatrixXXT * X_train ,MatrixXXT * Y_train){
      _X_train = X_train;
      _Y_train = Y_train;
    }
    void setTrainData(){}
    T cost();
    
    Matrix<T,1,Dynamic>*       biasGradient(int);
    Matrix<T,Dynamic,Dynamic>* weightGradient(int);
    
  };
  
#include "../src/fterm/poisson2D/BdyTerm.cpp"
#endif
  
}
