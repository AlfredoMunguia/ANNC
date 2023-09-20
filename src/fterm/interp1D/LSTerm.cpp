template <typename T>
void LSTerm<T>::init(){  

  // _arv = new  RowVectorXT(1);
  // _brv = new  RowVectorXT(1);
  
  // (*_arv)(0) = _a;
  // (*_brv)(0) = _b;

  _errors.resize(_nTrain);
  _vD.resize(_nTrain);
  _vB.resize(_nTrain);
  _va.resize(_nTrain);
  
  _architecture = *FTerm<T>::_net -> getArchitecture();
  _nL           = _architecture[_architecture.size()-1];
  _nInput       = _architecture[0];

  
  // for (unsigned int l = 0; l < _architecture.size(); l++) {
  //   _z.push_back(new RowVectorXT(_architecture[l]));    
  // }    

  
  for(int k=0; k < _nTrain; k++){    

    for (unsigned int l = 0; l < _architecture.size(); l++) {
      _va[k].push_back(new RowVectorXT(_architecture[l]));
      
      if(l < _architecture.size() - 1){
        _vD[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));
        _vB[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
      }
    }        

  }
  
  for (unsigned int l = 0; l < _architecture.size(); l++) {

    _z.push_back(new RowVectorXT(_architecture[l]));
    
    if(l < _architecture.size() - 1){
      _gradb.push_back(new RowVectorXT(_architecture[l+1]));      
      _gradW.push_back(new MatrixXXT(_architecture[l],_architecture[l+1]));
      
      _gradb.back()->setZero();
      _gradW.back()->setZero();
    }
  }


}
  


template <typename T>
void LSTerm<T>::forward(){

  for(int k=0; k < _nTrain; k++){            

    *_va[k].front() = _X_train  -> block(0,k,_nInput,1).transpose();    
    // propagate forward (vector multiplication)
    for (int l = 1; l < _architecture.size(); l++) {
      // *_va[k][l] = *_va[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_z[l] = *_va[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      
      //apply activation function
      for (int col = 0; col < _architecture[l]; col++){      
        _va[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_z[l]->coeffRef(col));        
        _vD[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_z[l]->coeffRef(col));
      }
    }
  }


}


template <typename T>
void LSTerm<T>::backward(){

  int L = _architecture.size() - 2;

  for(int k=0; k < _nTrain; k++){        
    *_vB[k][L]  =  *_vD[k][L];    
    for (int l = L - 1; l >= 0; l--){
      *_vB[k][l]  =   (*_vD[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB[k][l+1]; 
    }    
  }

}


template <typename T>
Matrix<T,1,Dynamic>* LSTerm<T>::biasGradient(int l){  
  T output;   
  int L = _va[0].size() - 2;
  _gradb[l] -> setZero();  

  for(int k=0; k < _nTrain; k++){
    output      =  (*_Y_train)(0,k);
    *_gradb[l] +=  ((*_va[k][L+1])[0] - output ) * _vB[k][l]->transpose();
  }  

  *_gradb[l] =  (1.0/(_nTrain))* *_gradb[l];

  // cout <<"l "<< l <<" "<<*_gradb[l] << endl;
  
  return _gradb[l];
}

template <typename T>
Matrix<T,Dynamic,Dynamic>* LSTerm<T>::weightGradient(int l){  
  
  T output;
  int L = _va[0].size() - 2;
  _gradW[l] -> setZero();

  for(int k=0; k < _nTrain; k++){    
    output      =  (*_Y_train)(0,k);
    *_gradW[l] += _va[k][l]->transpose() * ((*_va[k][L+1])[0] - output) * _vB[k][l]->transpose() ;
  }

  *_gradW[l] = (1.0/(_nTrain))* *_gradW[l];

  return _gradW[l];
}

template <typename T>
void LSTerm<T>::gradient(){  

  forward();
  backward();
  
  int L = _architecture.size() - 2;
  for (int l = L; l >= 0 ; l--) {
    *_gradb[l] = *biasGradient(l);
    *_gradW[l] = *weightGradient(l);
  }


}





template <typename T>
void LSTerm<T>::train(){
  // setTrainData();
  forward();
  backward();
  update();
}





template <typename T>
void LSTerm<T>::update(){
  int L = _architecture.size() - 2;


  for (int l = L; l >= 0 ; l--) {
    *FTerm<T>::_net->_b[l]   -=   FTerm<T>::_net->mLearningRate  * *biasGradient(l);
    *FTerm<T>::_net->_W[l]   -=   FTerm<T>::_net->mLearningRate  * *weightGradient(l);
  }


}


template <typename T>
T LSTerm<T>::cost(){

  RowVectorXT _input(1);
  RowVectorXT _input_a(1);
  RowVectorXT _input_b(1);
  
  RowVectorXT _output;
  RowVectorXT _output_a;
  RowVectorXT _output_b;
  
  T _exact;
  T _error;
  T _error_a;
  T _error_b;
  
  int _nInput = FTerm<T>::_net->mArchitecture[0];

  // cout << _nInput << endl;
  
  for(int k=0; k < _nTrain; k++){
    _input(0) =   (*_X_train)(0,k);
    
    FTerm<T>::_net->forward(_input);
    _output  = FTerm<T>::_net->_a.back() -> transpose();    
    _exact =  (*_Y_train)(0,k);

    _error =  _exact - _output(0);
    _errors(k)  = _error * _error;

  }

  _error = ( _errors.sum()/_errors.size() );
  
  return _error; // + _error_a +  _error_b;

}


template <typename T>
void LSTerm<T>::setTrainData(){
  
  int nTrain  = _X_train -> cols();
  int min     = 0;
  int max     = _nTrain-1;

  int num;
  int j;
  bool find;
  vector<int> v(_nTrain);

  for(int i=0; i < _nTrain; i++){
    do{
      num   = rand() * (1.0 / RAND_MAX) * (max-min+1) + min;
      find = false;

      for(j=0; j < i; j++)
        if(num == v[j]){ 
          find = true;
          break;
        }      
    } while(find);
    
    v[i] = num;
    // v[i] = 3;
  }


  if(_X_train != 0 && _Y_train != 0 ){
    delete _X_train;
    delete _Y_train;
    
    _X_train = 0;
    _Y_train = 0;
  }
    
  _X_train = new Matrix<T,Dynamic,Dynamic> ((*_X_train)(Eigen::placeholders::all,v));
  _Y_train = new Matrix<T,Dynamic,Dynamic> ((*_Y_train)(Eigen::placeholders::all,v));

  // cout << *_X_train << endl;
  // cout << *_Y_train << endl;

  
}
