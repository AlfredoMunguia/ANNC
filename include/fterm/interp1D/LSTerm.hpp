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

    int _nCTrain;
    int _nTrain;
    
    MatrixXXT                 _Lambda;
    
    vector<vector<MatrixXXT*> >   _vD;
    vector<vector<MatrixXXT* > >  _vB;
    vector<vector<RowVectorXT*> > _va;   
    vector<RowVectorXT*>  _z;   
    vector<MatrixXXT*>    _Da;
    vector<MatrixXXT* >   _Ba;
    vector<RowVectorXT*>  _aa;   
    vector<MatrixXXT*>    _Db;
    vector<MatrixXXT* >   _Bb;
    vector<RowVectorXT*>  _ab;   

    
    T _a;
    T _b;
    T _penalty1;
    T _penalty2;
    
    RowVectorXT* _arv;
    RowVectorXT* _brv;
    
    vector<RowVectorXT*>  _gradb;
    vector<MatrixXXT*>    _gradW;
    
    int _nInput;
    int _nL;

    vector<int> _architecture;
    RowVectorXT _errors;

    void init();
    
  public:


    LSTerm(MatrixXXT *X_train, MatrixXXT *Y_train, NeuralNetwork<T>* net):
      FTerm<T>(net)
    {
      _X_train  = X_train;
      _Y_train  = Y_train;      
      _nTrain   = X_train -> cols();

      
      init();
    };

    void forward();
    void backward();
    void update();
    void train();

    void gradient();
    

    void setTrainData();

    void setTrainData(MatrixXXT* X_train,MatrixXXT* Y_train)
    {
      _X_train  = X_train;
      _Y_train  = Y_train;      
    };

    T cost();
    
    Matrix<T,1,Dynamic>*        biasGradient(int);
    Matrix<T,Dynamic,Dynamic>*  weightGradient(int);

    vector<RowVectorXT*>*  getBias(){return & _gradb;}
    vector<MatrixXXT*>*    getWeights(){return & _gradW;}
    vector<int>*           getArchitecture(){return &_architecture;}           
  };
  
#include "../src/fterm/interp1D/LSTerm.cpp"
#endif
  
}
