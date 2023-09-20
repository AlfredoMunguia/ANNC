template <typename T>
void DivTerm<T>::init(){  

  _architecture = *FTerm<T>::_net -> getArchitecture();
  _nL           = _architecture[_architecture.size()-1];
  _nInput       = _architecture[0];
  
  for(int k=0; k <_nTrain; k++){    
    for (unsigned int l = 0; l < _architecture.size(); l++) {
      _va[k].push_back(new RowVectorXT(_architecture[l]));
      _va_E[k].push_back(new RowVectorXT(_architecture[l]));
      _va_W[k].push_back(new RowVectorXT(_architecture[l]));
      _va_T[k].push_back(new RowVectorXT(_architecture[l]));
      _va_B[k].push_back(new RowVectorXT(_architecture[l]));
      
      if(l < _architecture.size() - 1){

        _vD[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));
        _vD_E[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));
        _vD_W[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));
        _vD_T[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));
        _vD_B[k].push_back(new MatrixXXT(_architecture[l+1], _architecture[l+1]));

        _vB[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
        _vB_E[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
        _vB_W[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
        _vB_T[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
        _vB_B[k].push_back(new MatrixXXT(_architecture[l+1],_nL));
        
      }
    }
  }

  for (unsigned int l = 0; l < _architecture.size(); l++) {

    _z.push_back(new RowVectorXT(_architecture[l]));
    _z_E.push_back(new RowVectorXT(_architecture[l]));
    _z_W.push_back(new RowVectorXT(_architecture[l]));
    _z_T.push_back(new RowVectorXT(_architecture[l]));
    _z_B.push_back(new RowVectorXT(_architecture[l]));
    
    if(l < _architecture.size() - 1){
      _gradb.push_back(new RowVectorXT(_architecture[l+1]));      
      _gradW.push_back(new MatrixXXT(_architecture[l],_architecture[l+1]));

      _gradb.back()->setZero();
      _gradW.back()->setZero();
    }
  }
}


template <typename T>
void DivTerm<T>::forward(){

  RowVectorXT vD_EW = RowVectorXT::Zero(_nInput);
  RowVectorXT vD_BT = RowVectorXT::Zero(_nInput);
  
  vD_EW(0) = _delta_x;
  vD_BT(1) = _delta_y;

  // cout << _nTrain << endl;
  for(int k=0; k < _nTrain; k++){            

    *_va[k].front()   = _X_train  -> block(0,k,_nInput,1).transpose();

    *_va_E[k].front() = _X_train  -> block(0,k,_nInput,1).transpose() + vD_EW ;    
    *_va_W[k].front() = _X_train  -> block(0,k,_nInput,1).transpose() - vD_EW ;
    
    *_va_T[k].front() = _X_train  -> block(0,k,_nInput,1).transpose() + vD_BT ;    
    *_va_B[k].front() = _X_train  -> block(0,k,_nInput,1).transpose() - vD_BT ;    

    
    // propagate forward (vector multiplication)
    // cout << _architecture.size() << endl;
    for (int l = 1; l < _architecture.size(); l++) {
      
      *_z[k]   = *_va[k][l-1]   * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_z_E[k] = *_va_E[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_z_W[k] = *_va_W[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_z_T[k] = *_va_T[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_z_B[k] = *_va_B[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      
      
      
      //apply activation function
      for (int col = 0; col < _architecture[l]; col++){      
        
        _va[k][l]     -> coeffRef(col)       = FTerm<T>::_net-> activation(_z[k]->coeffRef(col));
        _vD[k][l-1]   -> coeffRef(col,col)   = FTerm<T>::_net-> activationDerivative(_z[k]->coeffRef(col));
        
        _va_E[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_z_E[k][l]->coeffRef(col));
        _vD_E[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_z_E[k][l]->coeffRef(col));

        _va_W[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_z_W[k][l]->coeffRef(col));
        _vD_W[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_z_W[k][l]->coeffRef(col));

        _va_T[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_z_T[k][l]->coeffRef(col));
        _vD_T[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_z_T[k][l]->coeffRef(col));

        _va_B[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_z_B[k][l]->coeffRef(col));
        _vD_B[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_z_B[k][l]->coeffRef(col));
      }
    }
  }
}

template <typename T>
void DivTerm<T>::backward(){

  int L = _architecture.size() - 2;
  
  for(int k=0; k <_nTrain; k++){        

    *_vB[k][L]  =  *_vD[k][L];          

    *_vB_E[k][L]  =  *_vD_E[k][L];           
    *_vB_W[k][L]  =  *_vD_W[k][L];

    *_vB_T[k][L]  =  *_vD_T[k][L];           
    *_vB_B[k][L]  =  *_vD_B[k][L];
    

    
    for (int l = L - 1; l >= 0; l--){

      *_vB[k][l]       =   ( *_vD[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB[k][l+1]; 
      
      *_vB_E[k][l]     =   ( *_vD_E[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB_E[k][l+1]; 
      *_vB_W[k][l]     =   ( *_vD_W[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB_W[k][l+1]; 

      *_vB_T[k][l]     =   ( *_vD_T[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB_T[k][l+1]; 
      *_vB_B[k][l]     =   ( *_vD_B[k][l] * *FTerm<T>::_net->_W[l+1]) * *_vB_B[k][l+1]; 
    }
  }

}




template <typename T>
Matrix<T,1,Dynamic>* DivTerm<T>::biasGradient(int l){  
  RowVectorXT data;   
  RowVectorXT t1;
  RowVectorXT t2;
  RowVectorXT t3;
  RowVectorXT t4;
  RowVectorXT t5;

  int L = _va_E[0].size() - 2;
  
  _gradb[l] -> setZero();  
  
  for(int k=0; k < _nTrain; k++){    
    data = _X_train  -> block(0,k,_nInput,1).transpose();

    t1 =  (1.0/(_delta_x*_delta_x)) * (*_va_E[k][L+1]-*_va[k][L+1]) * _vB_E[k][l] -> transpose()
       -  (1.0/(_delta_x*_delta_x)) * (*_va_E[k][L+1]-*_va[k][L+1]) * _vB[k][l]   -> transpose();

    t2 =  (1.0/(_delta_x*_delta_y)) * (*_va_E[k][L+1]-*_va[k][L+1]) * _vB_T[k][l] -> transpose()
       -  (1.0/(_delta_x*_delta_y)) * (*_va_E[k][L+1]-*_va[k][L+1]) * _vB[k][l]   -> transpose();

    t3 =  (1.0/(_delta_x*_delta_y)) * (*_va_T[k][L+1]-*_va[k][L+1]) * _vB_E[k][l] -> transpose()
       -  (1.0/(_delta_x*_delta_y)) * (*_va_T[k][L+1]-*_va[k][L+1]) * _vB[k][l]   -> transpose();

    t4 =  (1.0/(_delta_y*_delta_y)) * (*_va_T[k][L+1]-*_va[k][L+1]) * _vB_T[k][l] -> transpose()
       -  (1.0/(_delta_y*_delta_y)) * (*_va_T[k][L+1]-*_va[k][L+1]) * _vB[k][l]   -> transpose();

    t5 =   _f_function(data(0),data(1)) * _vB[k][l]->transpose();

    // t1 =  (1.0/_delta_x)*(*_va_E[k][L+1]-*_va[k][L+1])*(*_vB_E[k][l]-*_vB[k][l] ).transpose()   ;   // (BlE-Bl)^t*((u_E - u)/dx)
    // t2 =  (1.0/_delta_y)*(*_va_T[k][L+1]-*_va[k][L+1])*(*_vB_T[k][l]-*_vB[k][l] ).transpose()   ;   // (BlE-Bl)^t*((u_E - u)/dy)
    // t3 =  (  _f_function(data(0),data(1)) )* _vB[k][l]->transpose();
    
    *_gradb[l] += t1 + t2 + t3 + t4 - t5;  
  }  

  *_gradb[l] = (_area/_nTrain)* *_gradb[l];
  
  return _gradb[l];

}

template <typename T>
Matrix<T,Dynamic,Dynamic>* DivTerm<T>::weightGradient(int l){  
  
  RowVectorXT data;
  // T t1;
  // T t2;
  MatrixXXT t1;
  MatrixXXT t2;
  MatrixXXT t3;
  MatrixXXT t4;
  MatrixXXT t5;
  
  int L = _va_E[0].size() - 2;
  
  _gradW[l] -> setZero();

  for(int k=0; k < _nTrain; k++){    
    data = _X_train  -> block(0,k,_nInput,1).transpose();

    // t1 =  (*_va_E[k][L+1] - *_va[k][L+1])[0]/_delta_x;
    // t1 = t1*t1;
    // t2 =  (*_va_T[k][L+1] - *_va[k][L+1])[0]/_delta_y;
    // t2 = t2*t2;
    // t1 = 4.0*(t1 + t2 - _f_function(data(0),data(1)));
  
    // t2 =    (*_va_T[k][l]-*_va[k][l]).transpose() * ((*_va_T[k][L+1] - *_va[k][L+1])/_delta_y) * (*_vB_T[k][l] - *_vB[k][l]).transpose();
    // t3 =    (*_va[k][l]).transpose() * ( _f_function(data(0),data(1)) ) * (*_vB[k][l]).transpose();
    
    t1 = (1.0/(_delta_x*_delta_x))  * _va_E[k][l]->transpose()  * ( *_va_E[k][L+1] - *_va[k][L+1] ) * _vB_E[k][l]->transpose()
       - (1.0/(_delta_x*_delta_x))  * _va[k][l]->transpose()    * ( *_va_E[k][L+1] - *_va[k][L+1] ) * _vB[k][l]->transpose();

    t2 = (1.0/(_delta_x*_delta_y))  * _va_T[k][l]->transpose()  * ( *_va_E[k][L+1] - *_va[k][L+1] ) * _vB_T[k][l]->transpose()
       - (1.0/(_delta_x*_delta_y))  *   _va[k][l]->transpose()  * ( *_va_E[k][L+1] - *_va[k][L+1] ) * _vB[k][l]->transpose();

    t3 = (1.0/(_delta_x*_delta_y))  * _va_E[k][l]->transpose()  * ( *_va_T[k][L+1] - *_va[k][L+1] ) * _vB_E[k][l]->transpose()
       - (1.0/(_delta_x*_delta_y))  *   _va[k][l]->transpose()  * ( *_va_T[k][L+1] - *_va[k][L+1] ) * _vB[k][l]->transpose();

    t4 = (1.0/(_delta_y*_delta_y))  * _va_T[k][l]->transpose()  * ( *_va_T[k][L+1] - *_va[k][L+1] ) * _vB_T[k][l]->transpose()
       - (1.0/(_delta_y*_delta_y))  *   _va[k][l]->transpose()  * ( *_va_T[k][L+1] - *_va[k][L+1] ) * _vB[k][l]->transpose();

    t5 = _va[k][l]->transpose() *  _f_function(data(0),data(1))  * _vB[k][l]->transpose();
    
    
    *_gradW[l] += t1 + t2 + t3 + t4 - t5; // + 2.0*t1*t3 + 2.0*t4*t2 + t4*t5;    
  }

  *_gradW[l] =  (_area/_nTrain)* *_gradW[l];

  // cout << *_gradW[l] << endl << endl;

  return _gradW[l];
}



template <typename T>
void DivTerm<T>::train(){
  //setTrainData();
  forward();
  backward();
  update();

}


template <typename T>
void DivTerm<T>::update(){

  int L = _architecture.size() - 2;

  for (int l = L; l >= 0 ; l--) {
    *FTerm<T>::_net->_b[l]   -=   FTerm<T>::_net->mLearningRate  * *biasGradient(l);
    *FTerm<T>::_net->_W[l]   -=   FTerm<T>::_net->mLearningRate  * *weightGradient(l);
  }
}


template <typename T>
T DivTerm<T>::cost(){

  RowVectorXT input;
  // RowVectorXT input_E;
  // RowVectorXT input_W;
  // RowVectorXT input_T;
  // RowVectorXT input_B;
  
  RowVectorXT output;
  RowVectorXT exact;
  // RowVectorXT output_E;
  // RowVectorXT output_W;
  // RowVectorXT output_T;
  // RowVectorXT output_B;

  // T laplacian;  
  // T exact;
  T error = 0;

  int nInput = FTerm<T>::_net->mArchitecture[0];

  // RowVectorXT vD_EW = RowVectorXT::Zero(nInput);
  // RowVectorXT vD_BT = RowVectorXT::Zero(nInput);
  
  // vD_EW(0) = _delta_x;
  // vD_BT(1) = _delta_y;

  // cout << _X_train->rows() <<" " << _X_train->cols() << endl;
  // cout << _Y_train->rows() <<" " << _Y_train->cols() << endl;


  for(int k=0; k < _nTrain; k++){

    input   =  _X_train -> block(0,k,nInput,1).transpose();
    exact   =  _Y_train -> block(0,k,1,1).transpose();

    // input_E = input + vD_EW;
    // // input_W = input - vD_EW;
    // input_T = input + vD_BT;
    // // input_B = input - vD_BT;


    FTerm<T>::_net->forward(input);
    output  = FTerm<T>::_net->_a.back() -> transpose();

    // cout << output << endl << endl;
    // cout << exact << endl  << endl;
    
    // FTerm<T>::_net->forward(input_E);
    // output_E  = FTerm<T>::_net->_a.back() -> transpose();    

    // // FTerm<T>::_net->forward(input_W);
    // // output_W  = FTerm<T>::_net->_a.back() -> transpose();
    // FTerm<T>::_net->forward(input_T);
    // output_T  = FTerm<T>::_net->_a.back() -> transpose();    

    // FTerm<T>::_net->forward(input_B);
    // output_B  = FTerm<T>::_net->_a.back() -> transpose();



    // laplacian = ((output_E - output)(0)/_delta_x)*((output_E - output)(0)/_delta_x)
    //           + ((output_T - output)(0)/_delta_y)*((output_T - output)(0)/_delta_y); 

    // error =  abs(laplacian - _f_function(input(0),input(1)) * output(0)  ) ;

    // laplacian = 0.5* ((output_E - output)(0)/_delta_x)*((output_E - output)(0)/_delta_x)
    //           + ((output_T - output)(0)/_delta_y)*((output_T - output)(0)/_delta_y);


    

    // error =  abs(laplacian -   _f_function(input(0),input(1)) *output(0)   ) ;
    error = abs(output(0) - exact(0));
    error = error*error;
    
    _errors(k)  = error;

  }
  
 
  return sqrt(_errors.sum()/_errors.size());

}


// template <typename T>
// void DivTerm<T>::setTrainData(){

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


//   if(_CX_train != 0 && _CY_train != 0 ){
//     delete _CX_train;
//     delete _CY_train;
    
//     _CX_train = 0;
//     _CY_train = 0;
//   }
    
//   _CX_train = new Matrix<T,Dynamic,Dynamic> ((*_X_train)(Eigen::placeholders::all,v));
//   _CY_train = new Matrix<T,Dynamic,Dynamic> ((*_Y_train)(Eigen::placeholders::all,v));


//   // cout << *_CX_train << endl;
//   // cout << *_CY_train << endl;
  
// }



template <typename T>
void DivTerm<T>::setAlpha(T alpha){
  FTerm<T>::_net->mLearningRate = alpha;
}
