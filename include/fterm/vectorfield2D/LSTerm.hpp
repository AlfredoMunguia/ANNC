#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <fterm/FTerm.hpp>

namespace fterm{
  
#ifndef LSTERM_H
#define LSTERM_H

  using namespace std;
  using namespace Eigen;

  template <typename T>
  class LSTerm: public FTerm<T>{
    
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;
    
    MatrixXXT                  *_X_train;
    MatrixXXT                  *_Y_train;

    MatrixXXT                  _Lambda;
    int _nTrain;
    
    vector<vector<MatrixXXT*> >   _vD;
    vector<vector<MatrixXXT* > >  _vB;
    vector<vector<RowVectorXT*> > _va;   
    
    vector<RowVectorXT*>  _gradb;
    vector<MatrixXXT*>    _gradW;
    
    int _nInput;
    int _nL;

    vector<int> _architecture;
    RowVectorXT _errors;

    void init();
    function<void  (T,T,T&,T&)> _u0;
    function< void (T,T,T&,T&)> _ue;
    
  public:

    LSTerm(MatrixXXT *X_train, MatrixXXT *Y_train, NeuralNetwork<T>* net, void (*u0)(T,T,T&,T&) ):
      FTerm<T>(net)
    {
      _X_train  = X_train;
      _Y_train  = Y_train;
      _nTrain   = X_train -> cols();
      _u0 = u0;
       init();
    };

    void forward();
    void backward();
    void update();
    void train();
    void setTrainData();
    void setTrainData(MatrixXXT* X_train, MatrixXXT* Y_train){
      _X_train = X_train;
      _Y_train = Y_train;
    }
    T cost();
    
    Matrix<T,1,Dynamic>*       biasGradient(int);
    Matrix<T,Dynamic,Dynamic>* weightGradient(int);
    
  };
  
#include "../src/fterm/vectorfield2D/LSTerm.cpp"
#endif
  
}
