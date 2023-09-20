#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <fterm/FTerm.hpp>
#include <functional>

namespace fterm{
  
#ifndef DIVTERM_H
#define DIVTERM_H

  using namespace std;
  using namespace Eigen;

  template <typename T>
  class DivTerm: public FTerm<T>{
    
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;
    
    MatrixXXT                  *_X_train;
    MatrixXXT                  *_Y_train;
    Matrix<T,Dynamic,Dynamic>  *_CX_train;
    Matrix<T,Dynamic,Dynamic>  *_CY_train;

    int _nCTrain;
    int _nTrain;
   
    MatrixXXT                   _Lambda;

    vector<vector<MatrixXXT*> >   _vD;
    vector<vector<MatrixXXT*> >   _vD_E;
    vector<vector<MatrixXXT*> >   _vD_W;
    vector<vector<MatrixXXT*> >   _vD_T;
    vector<vector<MatrixXXT*> >   _vD_B;

    vector<vector<MatrixXXT* > >  _vB;
    vector<vector<MatrixXXT* > >  _vB_E;
    vector<vector<MatrixXXT* > >  _vB_W;
    vector<vector<MatrixXXT* > >  _vB_T;
    vector<vector<MatrixXXT* > >  _vB_B;

    vector<vector<RowVectorXT*> > _va;
    vector<vector<RowVectorXT*> > _va_E;
    vector<vector<RowVectorXT*> > _va_W;   
    vector<vector<RowVectorXT*> > _va_T;
    vector<vector<RowVectorXT*> > _va_B;
    
    vector<RowVectorXT*>  _gradb;
    vector<MatrixXXT*>    _gradW;
    
    int _nInput;
    int _nL;
    T _delta_x;
    T _delta_y;
    T _area;
    T _penalty;
    
    vector<int> _architecture;
    RowVectorXT _errors;
    
    void init();

  public:

    DivTerm(MatrixXXT *X_train, NeuralNetwork<T>* net,
            T delta_x, T delta_y, T area, T penalty):FTerm<T>(net)
    {      
      _X_train = X_train;
      // _Y_train = Y_train;
      _nTrain  = X_train -> cols();
      _delta_x = delta_x;
      _delta_y = delta_y;      
      _area    = area;

      _penalty = penalty;
      
      _errors.resize(_nTrain);

      _vD.resize(_nTrain);
      _vD_E.resize(_nTrain);
      _vD_W.resize(_nTrain);
      _vD_T.resize(_nTrain);
      _vD_B.resize(_nTrain);
      
      _vB.resize(_nTrain);
      _vB_E.resize(_nTrain);
      _vB_W.resize(_nTrain);
      _vB_T.resize(_nTrain);
      _vB_B.resize(_nTrain);
      
      _va.resize(_nTrain);
      _va_E.resize(_nTrain);
      _va_W.resize(_nTrain);
      _va_T.resize(_nTrain);
      _va_B.resize(_nTrain);

      init();
    }


        
    void forward();
    void backward();
    void update();
    void train();
    // void setTrainData();

    void setTrainData(MatrixXXT * X_train, MatrixXXT * Y_train = 0){
      _X_train = X_train;
      _Y_train = Y_train;
    }

    void setTrainData(){}
    T cost();

    Matrix<T,1,Dynamic>*       biasGradient(int);
    Matrix<T,Dynamic,Dynamic>* weightGradient(int);
    
  };
  
#include "../src/fterm/vectorfield2D/DivTerm.cpp"
#endif
  
}

