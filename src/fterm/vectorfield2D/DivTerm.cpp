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
      
      *_va[k][l]   = *_va[k][l-1]   * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_va_E[k][l] = *_va_E[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_va_W[k][l] = *_va_W[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_va_T[k][l] = *_va_T[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      *_va_B[k][l] = *_va_B[k][l-1] * *FTerm<T>::_net->_W[l-1]  + *FTerm<T>::_net->_b[l-1];
      
      
      
      //apply activation function
      for (int col = 0; col < _architecture[l]; col++){      
        
        _va[k][l]     -> coeffRef(col)       = FTerm<T>::_net-> activation(_va[k][l]->coeffRef(col));
        _vD[k][l-1]   -> coeffRef(col,col)   = FTerm<T>::_net-> activationDerivative(_va[k][l]->coeffRef(col));
        
        _va_E[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_va_E[k][l]->coeffRef(col));
        _vD_E[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_va_E[k][l]->coeffRef(col));

        _va_W[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_va_W[k][l]->coeffRef(col));
        _vD_W[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_va_W[k][l]->coeffRef(col));

        _va_T[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_va_T[k][l]->coeffRef(col));
        _vD_T[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_va_T[k][l]->coeffRef(col));

        _va_B[k][l]   -> coeffRef(col)     = FTerm<T>::_net-> activation(_va_B[k][l]->coeffRef(col));
        _vD_B[k][l-1] -> coeffRef(col,col) = FTerm<T>::_net-> activationDerivative(_va_B[k][l]->coeffRef(col));
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
  RowVectorXT t6;
  RowVectorXT t7;
  RowVectorXT t8;



  int L = _va_E[0].size() - 2;
  
  _gradb[l] -> setZero();  
  
  for(int k=0; k < _nTrain; k++){    
    
    t1 =  ( 1.0 / (2.0*_delta_x*_delta_x) ) * (*_va_E[k][L+1] - *_va_W[k][L+1])[0] * _vB_E[k][l]->block(0,0,_architecture[l+1],1).transpose() ; //First component of B_E 
    t2 =  ( 1.0 / (2.0*_delta_x*_delta_x) ) * (*_va_E[k][L+1] - *_va_W[k][L+1])[0] * _vB_W[k][l]->block(0,0,_architecture[l+1],1).transpose() ; //First component of B_W 

    t3 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * (*_va_E[k][L+1] - *_va_W[k][L+1])[1] * _vB_T[k][l]->block(0,1,_architecture[l+1],1).transpose() ; //Second component of B_T
    t4 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * (*_va_E[k][L+1] - *_va_W[k][L+1])[1] * _vB_B[k][l]->block(0,1,_architecture[l+1],1).transpose() ; //Second component of B_B 

    t5 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * (*_va_T[k][L+1] - *_va_B[k][L+1])[0] * _vB_E[k][l]->block(0,0,_architecture[l+1],1).transpose() ;  
    t6 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * (*_va_T[k][L+1] - *_va_B[k][L+1])[0] * _vB_W[k][l]->block(0,0,_architecture[l+1],1).transpose() ;  

    t7 =  ( 1.0 / (2.0*_delta_y*_delta_y) ) * (*_va_T[k][L+1] - *_va_B[k][L+1])[1] * _vB_T[k][l]->block(0,1,_architecture[l+1],1).transpose() ;
    t8 =  ( 1.0 / (2.0*_delta_y*_delta_y) ) * (*_va_T[k][L+1] - *_va_B[k][L+1])[1] * _vB_B[k][l]->block(0,1,_architecture[l+1],1).transpose() ;  
    
    // t2 =  ( 1.0/_delta_y )*(*_va_T[k][L+1]-*_va[k][L+1])*(*_vB_T[k][l]-*_vB[k][l] ).transpose()   ;   // (BlE-Bl)^t*((u_E - u)/dy)
    // t3 =  (  _f_function(data(0),data(1)) )* _vB[k][l]->transpose();
    // t3 =   t1 * (*_vB_E[k][l]   - *_vB[k][l] ).transpose();
    // t4 =   t2 * (*_vB_T[k][l]   - *_vB[k][l] ).transpose();
    
    *_gradb[l] += t1 - t2 + t3 - t4 + t5 - t6 + t7 - t8;  
  }  



  *_gradb[l] = (2.0*_penalty/_nTrain)* *_gradb[l] * _area;
  
  // cout <<  *_gradb[l] << endl;
  return _gradb[l];

}

template <typename T>
Matrix<T,Dynamic,Dynamic>* DivTerm<T>::weightGradient(int l){  

  MatrixXXT t1;
  MatrixXXT t2;
  MatrixXXT t3;
  MatrixXXT t4;
  MatrixXXT t5;
  MatrixXXT t6;
  MatrixXXT t7;
  MatrixXXT t8;
  
  int L = _va_E[0].size() - 2;
  
  _gradW[l] -> setZero();

  for(int k=0; k < _nTrain; k++){    

    t1 =  ( 1.0/  (2.0*_delta_x*_delta_x) ) * _va_E[k][l]->transpose() * (*_va_E[k][L+1] - *_va_W[k][L+1])[0] * _vB_E[k][l]->block(0,0,_architecture[l+1],1).transpose() ; //First component of B_E 
    t2 =  ( 1.0 / (2.0*_delta_x*_delta_x) ) * _va_W[k][l]->transpose() * (*_va_E[k][L+1] - *_va_W[k][L+1])[0] * _vB_W[k][l]->block(0,0,_architecture[l+1],1).transpose() ; //First component of B_W 

    t3 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * _va_T[k][l]->transpose() * (*_va_E[k][L+1] - *_va_W[k][L+1])[1] * _vB_T[k][l]->block(0,1,_architecture[l+1],1).transpose() ; //Second component of B_T
    t4 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * _va_B[k][l]->transpose() * (*_va_E[k][L+1] - *_va_W[k][L+1])[1] * _vB_B[k][l]->block(0,1,_architecture[l+1],1).transpose() ; //Second component of B_B 

    t5 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * _va_E[k][l]->transpose() * (*_va_T[k][L+1] - *_va_B[k][L+1])[0] * _vB_E[k][l]->block(0,0,_architecture[l+1],1).transpose() ;  
    t6 =  ( 1.0 / (2.0*_delta_x*_delta_y) ) * _va_W[k][l]->transpose() * (*_va_T[k][L+1] - *_va_B[k][L+1])[0] * _vB_W[k][l]->block(0,0,_architecture[l+1],1).transpose() ;  

    t7 =  ( 1.0 / (2.0*_delta_y*_delta_y) ) * _va_T[k][l]->transpose() * (*_va_T[k][L+1] - *_va_B[k][L+1])[1] * _vB_T[k][l]->block(0,1,_architecture[l+1],1).transpose() ;
    t8 =  ( 1.0 / (2.0*_delta_y*_delta_y) ) * _va_B[k][l]->transpose() * (*_va_T[k][L+1] - *_va_B[k][L+1])[1] * _vB_B[k][l]->block(0,1,_architecture[l+1],1).transpose() ;  
    
    
    //cout << t1 << endl << endl ;
    //cout << " -------------- " << endl;
    
    // t1 =  (*_va_E[k][L+1] - *_va[k][L+1])[0]/_delta_x;
    // t1 = t1*t1;
    // t2 =  (*_va_T[k][L+1] - *_va[k][L+1])[0]/_delta_y;
    // t2 = t2*t2;
    // t1 = 4.0*(t1 + t2 - _f_function(data(0),data(1)));
  
    
    // t1 =    (*_va_E[k][l]-*_va[k][l]).transpose() * ((*_va_E[k][L+1] - *_va[k][L+1])/_delta_x) * (*_vB_E[k][l] - *_vB[k][l]).transpose();

    // t2 =    (*_va_T[k][l]-*_va[k][l]).transpose() * ((*_va_T[k][L+1] - *_va[k][L+1])/_delta_y) * (*_vB_T[k][l] - *_vB[k][l]).transpose();
    // t3 =    (*_va[k][l]).transpose() * ( _f_function(data(0),data(1)) ) * (*_vB[k][l]).transpose();
    
    *_gradW[l] += t1 - t2 + t3 - t4 + t5 -t6 + t7 -t8;
  }

  // cout << _penalty << endl;
  
  *_gradW[l] = (2.0*_penalty/_nTrain)* *_gradW[l] * _area;

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
  RowVectorXT input_E;
  RowVectorXT input_W;
  RowVectorXT input_T;
  RowVectorXT input_B;
  
  RowVectorXT output;
  RowVectorXT output_E;
  RowVectorXT output_W;
  RowVectorXT output_T;
  RowVectorXT output_B;

  T div;  
  T exact;
  T error = 0;

  int nInput = FTerm<T>::_net->mArchitecture[0];

  RowVectorXT vD_EW = RowVectorXT::Zero(nInput);
  RowVectorXT vD_BT = RowVectorXT::Zero(nInput);
  
  vD_EW(0) = _delta_x;
  vD_BT(1) = _delta_y;

  for(int k=0; k < _nTrain; k++){

    input   =  _X_train -> block(0,k,nInput,1).transpose();
    
    input_E = input + vD_EW;
    input_W = input - vD_EW;
    input_T = input + vD_BT;
    input_B = input - vD_BT;


    // cout <<  input << endl;
    // cout <<  input_E << endl;
    // cout <<  input_W << endl;
    // cout <<  input_T << endl;
    // cout <<  input_B << endl;
    
    
    
    // FTerm<T>::_net->forward(input);
    // output  = FTerm<T>::_net->_a.back() -> transpose();

    // cout << "--------------------" << endl;    
    FTerm<T>::_net->forward(input_E);
    output_E  = FTerm<T>::_net->_a.back() -> transpose();    
    
    FTerm<T>::_net->forward(input_W);
    output_W  = FTerm<T>::_net->_a.back() -> transpose();

    // cout << " E " <<output_E << endl;
    // cout << " W " <<output_W << endl;

    FTerm<T>::_net->forward(input_T);
    output_T  = FTerm<T>::_net->_a.back() -> transpose();    
    
    FTerm<T>::_net->forward(input_B);
    output_B  = FTerm<T>::_net->_a.back() -> transpose();

    // cout <<" T " << output_T << endl;
    // cout <<" B " << output_B << endl;
   
    
    // cout << output_E(0) << " " << output_W(0) << endl;
    // cout << output_T(1) << " " << output_B(1) << endl;
    
    div = ((output_E(0) - output_W(0))/ (2.0*_delta_x)) 
        + ((output_T(1) - output_B(1))/ (2.0*_delta_y)); 

    
    // cout << " div " << div << endl;
    // cout << "--------------------" << endl;
    error =  div*div ;

    _errors(k)  = error;

  }
  
 
  return _penalty* (_errors.sum()/_errors.size())*_area;

  //return error;

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
