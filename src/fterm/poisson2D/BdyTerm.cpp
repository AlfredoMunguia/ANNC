template <typename T>
void BdyTerm<T>::init(){  
  
  
  _architecture = *FTerm<T>::_net -> getArchitecture();
  _nL           = _architecture[_architecture.size()-1];
  _nInput       = _architecture[0];
  

  for(int k=0; k <_nTrain; k++){    
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
void BdyTerm<T>::forward(){
  
  for(int k=0; k < _nTrain; k++){            
    *_va[k].front()   = _X_train  -> block(0,k,_nInput,1).transpose();
    // propagate forward (vector multiplication)
    for (int l = 1; l < _architecture.size(); l++) {

      *_va[k][l]   = *_va[k][l-1]   * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];

      //apply activation function
      for (int col = 0; col < _architecture[l]; col++){      
        _va[k][l]     -> coeffRef(col)       = FTerm<T>::_net-> activation(_va[k][l]->coeffRef(col));
        _vD[k][l-1]   -> coeffRef(col,col)   = FTerm<T>::_net-> activationDerivative(_va[k][l]->coeffRef(col));
      }
    }
  }
}


template <typename T>
void BdyTerm<T>::backward(){

  int L = _architecture.size() - 2;
  
  for(int k=0; k < _nTrain; k++){        
    *_vB[k][L]  =  *_vD[k][L];          
    for (int l = L - 1; l >= 0; l--){
      *_vB[k][l]       =   ( *_vD[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB[k][l+1]; 
    }
  }
}


template <typename T>
Matrix<T,1,Dynamic>* BdyTerm<T>::biasGradient(int l){  
  RowVectorXT data;   
  int L = _va[0].size() - 2;
  
  _gradb[l] -> setZero();  

  for(int k=0; k < _nTrain; k++){    
    data = _X_train  -> block(0,k,_nInput,1).transpose();
    *_gradb[l] +=   ( (*_va[k][L+1])(0,0) - _g_function(data(0),data(1)) ) * _vB[k][l]->transpose() ;
  }

  *_gradb[l] = (2.0 *_penalty/_nTrain)* *_gradb[l] * _perimeter;

  // cout <<  *_gradb[l] << endl;
  
  return _gradb[l];

}


template <typename T>
Matrix<T,Dynamic,Dynamic>* BdyTerm<T>::weightGradient(int l){  
  
  RowVectorXT output;
  RowVectorXT data;
  int L = _va[0].size() - 2;

  _gradW[l] -> setZero();

  for(int k=0; k < _nTrain; k++){    
    data = _X_train  -> block(0,k,_nInput,1).transpose();
    
    *_gradW[l] +=     _va[k][l]->transpose() * ( (*_va[k][L+1])(0,0) - _g_function(data(0),data(1)) )  * _vB[k][l]->transpose() ;
  }

  *_gradW[l] = (2.0 *_penalty/_nTrain)* *_gradW[l] * _perimeter;

  // cout << *_gradW[l] << endl << endl;
  return _gradW[l];
}



template <typename T>
void BdyTerm<T>::train(){
  //setTrainData();
  forward();
  backward();
  update();
}


template <typename T>
void BdyTerm<T>::update(){
  int L = _architecture.size() - 2;
  for (int l = L; l >= 0 ; l--) {
    *FTerm<T>::_net->_b[l]   -=   FTerm<T>::_net->mLearningRate  * *biasGradient(l);
    *FTerm<T>::_net->_W[l]   -=   FTerm<T>::_net->mLearningRate  * *weightGradient(l);
  }
}


template <typename T>
T BdyTerm<T>::cost(){

  RowVectorXT input;
  RowVectorXT output;

  T laplacian;  
  T exact;
  T error;

  int nInput = FTerm<T>::_net->mArchitecture[0];


  for(int k=0; k < _nTrain; k++){
    
    input   =  _X_train -> block(0,k,nInput,1).transpose();

    FTerm<T>::_net->forward(input);
    output  = FTerm<T>::_net->_a.back() -> transpose();    
    
    exact =  _g_function(input(0),input(1));
    error = output(0) - exact;

    
    _errors(k) = error*error;
    
  }

  //  cout << "error "  << _penalty  *(_errors.sum()/_errors.size())*_perimeter << endl;
  return _penalty  *(_errors.sum()/_errors.size())*_perimeter;

}


// template <typename T>
// void BdyTerm<T>::setTrainData(){
  
//   int nTrain  = _X_train -> cols();
//   int min     = 0;
//   int max     = _nTrain-1;

//   int num;
//   int j;
//   bool find;
//   vector<int> v(_nTrain);

//   for(int i=0; i < _nTrain; i++){
//     do{
//       num   = rand() * (1.0 / RAND_MAX) * (max-min+1) + min;
//       find = false;

//       for(j=0; j < i; j++)
//         if(num == v[j]){ 
//           find = true;
//           break;
//         }      
//     } while(find);
    
//     v[i] = num;
//     // v[i] = 3;
//   }


//   if(_X_train != 0 && _Y_train != 0 ){
//     delete _X_train;
//     delete _Y_train;
    
//     _X_train = 0;
//     _Y_train = 0;
//   }
    
//   _X_train = new Matrix<T,Dynamic,Dynamic> ((*_X_train)(Eigen::placeholders::all,v));
//   _Y_train = new Matrix<T,Dynamic,Dynamic> ((*_Y_train)(Eigen::placeholders::all,v));

//   // cout << *_X_trainy << endl;
//   // cout << *_Y_train << endl;

  
// }
