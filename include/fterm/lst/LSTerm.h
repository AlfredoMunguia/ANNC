#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <fterm/FTerm.h>

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
    // Matrix<T,Dynamic,Dynamic>* _CX_train;
    // Matrix<T,Dynamic,Dynamic>* _CY_train;

    int _nTrain;
    
    MatrixXXT                 _Lambda;
    
    vector<vector<MatrixXXT*> >   _vD;
    vector<vector<MatrixXXT* > >  _vB;
    vector<vector<RowVectorXT*> > _va;   
    vector<RowVectorXT*>  _z;   
    
    vector<RowVectorXT*>  _gradb;
    vector<MatrixXXT*>    _gradW;
    
    int _nInput;
    int _nL;

    vector<int> _architecture;
    RowVectorXT _errors;

    void init();
    
  public:

    //LSTerm(MatrixXXT *X_train, MatrixXXT *Y_train, NeuralNetwork<T>* net, int nCTrain):
    LSTerm(NeuralNetwork<T>* net, int nTrain):    
      FTerm<T>(net)
    {
      _nTrain  = nTrain;
      init();
    };

    LSTerm(MatrixXXT *X_train, MatrixXXT *Y_train, NeuralNetwork<T>* net):
      FTerm<T>(net)
    {
      _nTrain  = X_train -> cols();
      _X_train = X_train;
      _Y_train = Y_train;
      init();
    }


    void forward();
    void backward();
    void update();
    void train();
    void setTrainData();

    void setTrainData(MatrixXXT* X_train, MatrixXXT* Y_train)
    {
      _X_train = X_train;
      _Y_train = Y_train;
    };

    T cost();
    
    Matrix<T,1,Dynamic>*       biasGradient(int);
    Matrix<T,Dynamic,Dynamic>* weightGradient(int);
    
  };
  
#include "../src/fterm/lst/LSTerm.cpp"
#endif
  
}
