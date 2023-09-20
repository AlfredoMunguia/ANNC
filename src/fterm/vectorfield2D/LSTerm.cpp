template <typename T>
void LSTerm<T>::init(){  
  
  _errors.resize(_nTrain);
  _vD.resize(_nTrain);
  _vB.resize(_nTrain);
  _va.resize(_nTrain);
  
  _architecture = *FTerm<T>::_net -> getArchitecture();
  _nL           = _architecture[_architecture.size()-1];
  _nInput       = _architecture[0];
  
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
      *_va[k][l] = *_va[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];

      //apply activation function
      for (int col = 0; col < _architecture[l]; col++){      
        _va[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_va[k][l]->coeffRef(col));
        _vD[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_va[k][l]->coeffRef(col));
        
      }

      //cout << *_va[k][l] << endl << endl; 
      
    }
    
  }
}

template <typename T>
void LSTerm<T>::backward(){
  int L = _architecture.size() - 2;

  for(int k=0; k < _nTrain; k++){        
    *_vB[k][L]  =  *_vD[k][L];           
    
    //cout << *_vB[k][L] << endl << endl;
    
    for (int l = L - 1; l >= 0; l--){
      *_vB[k][l]     =   ( *_vD[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB[k][l+1]; 

    }
  }
}


template <typename T>
Matrix<T,1,Dynamic>* LSTerm<T>::biasGradient(int l){  
  
  RowVectorXT data;   
  RowVectorXT output(2);
  T u0_x;
  T u0_y;
  int n,m;
  
  
  int L = _va[0].size() - 2;

  _gradb[l] -> setZero();  

  for(int k=0; k < _nTrain; k++){
    // n = _vB[k][l]-> rows();    
    data    =  _X_train  -> block(0,k,_nInput,1).transpose();
    // cout << data << endl;
    output(0) = data(0);
    output(1) = -data(1);
    // _ue(data(0),data(1),output(0),output(1));
    // cout << output << endl;
    *_gradb[l] +=  ( *_va[k][L+1] - output ) * _vB[k][l]->transpose();
    // _u0(data(0),data(1),u0_x,u0_y);
    // *_gradb[l] +=  ( (*_va[k][L+1])[0] - u0_x ) * (_vB[k][l]->block(0,0,n,1)).transpose();
  }  

  *_gradb[l] =  (1.0/_nTrain)* *_gradb[l];
  
  return _gradb[l];

}

template <typename T>
Matrix<T,Dynamic,Dynamic>* LSTerm<T>::weightGradient(int l){  
  
  RowVectorXT data;
  RowVectorXT output(2);
  T u0_x;
  T u0_y;
  int n;

  int L = _va[0].size() - 2;

  _gradW[l] -> setZero();

  for(int k=0; k < _nTrain; k++){    
    n = _vB[k][l]-> rows();
    data    =  _X_train  -> block(0,k,_nInput,1).transpose();
    //    _ue(data(0),data(1),output(0),output(1));
    output(0) = data(0);
    output(1) = -data(1);

    *_gradW[l] += _va[k][l]->transpose() * ( (*_va[k][L+1]) - output ) * _vB[k][l]->transpose() ;
    
    // _u0(data(0),data(1),u0_x,u0_y);    
    // *_gradW[l] += _va[k][l]->transpose() * ((*_va[k][L+1])[0] - u0_x ) * (_vB[k][l]->block(0,0,n,1)).transpose() ;

  }

  *_gradW[l] = (1.0/_nTrain)* *_gradW[l];
  // cout <<* _gradW[l] << endl;
  return _gradW[l];
}

template <typename T>
void LSTerm<T>::train(){

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

  RowVectorXT _input;
  RowVectorXT _output;
  RowVectorXT _exact;
  RowVectorXT _error;

  
  int _nInput = FTerm<T>::_net->mArchitecture[0];

  // cout << _nInput << endl;
  
  for(int k=0; k < _nTrain; k++){
    _input =  _X_train -> block(0,k,_nInput,1).transpose();
    _exact =  _Y_train -> block(0,k,_nInput,1).transpose();

    // cout << _input << endl;
    
    FTerm<T>::_net->forward(_input);
    _output  = FTerm<T>::_net->_a.back() -> transpose();    

    // cout << _exact << endl;
    // cout << _output << endl;
    
    _error =  _exact - _output;
    _errors(k)  = _error.dot(_error);


  }

  // cout << "cost " <<  0.5*(_errors.sum()/_errors.size()) << endl;
  // cout << "----------------------------" << endl;
  return 0.5*(_errors.sum()/_errors.size());

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
