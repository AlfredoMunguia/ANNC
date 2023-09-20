#include <iostream>


// namespace domain{


using namespace std;

template<typename T>
void Rectangle<T>::uniformSampleFromInner(MatrixXXT& X_test ){

  RowVectorXT xlinspace = RowVectorXT::LinSpaced(Domain<T>::_nEvalP,_a,_b);
  RowVectorXT ylinspace = RowVectorXT::LinSpaced(Domain<T>::_nEvalP,_c,_d);

  X_test.resize(2,Domain<T>::_nEvalP*Domain<T>::_nEvalP);

  int k = 0;
  for(int i=0; i < Domain<T>::_nEvalP; i++)
    for(int j=0; j < Domain<T>::_nEvalP; j++){
      X_test(0,k) = xlinspace(i);
      X_test(1,k) = ylinspace(j);
      k++;
    }    
}


template<typename T>
void Rectangle<T>::uniformSampleFromInner(MatrixXXT& X_test, int nP ){

  RowVectorXT xlinspace = RowVectorXT::LinSpaced(nP,_a,_b);
  RowVectorXT ylinspace = RowVectorXT::LinSpaced(nP,_c,_d);

  X_test.resize(2,nP*nP);

  int k = 0;
  for(int i=0; i < nP; i++)
    for(int j=0; j < nP; j++){
      X_test(0,k) = xlinspace(i);
      X_test(1,k) = ylinspace(j);
      k++;
    }    
}




template<typename T>
void Rectangle<T>::sampleFromInner(MatrixXXT& X_train, MatrixXXT& X_test ){

  RowVectorXT xlinspace = RowVectorXT::LinSpaced(Domain<T>::_nEvalP,_a,_b);
  RowVectorXT ylinspace = RowVectorXT::LinSpaced(Domain<T>::_nEvalP,_c,_d);
  T epsilon      = 10e-4;
  
  X_train  = MatrixXXT::Random(2,Domain<T>::_nIntP);

  X_train(0,all)   = (MatrixXXT::Constant(1,Domain<T>::_nIntP,1) - X_train(0,all) )*.5  ;
  X_train(0,all)   = MatrixXXT::Constant(1,Domain<T>::_nIntP, _a + epsilon) + X_train(0,all) * (_b -_a - 2.0*epsilon);
  X_train(1,all)   = (MatrixXXT::Constant(1,Domain<T>::_nIntP,1) - X_train(1,all) )*.5  ;
  X_train(1,all)   = MatrixXXT::Constant(1,Domain<T>::_nIntP, _c + epsilon) + X_train(1,all) * (_d-_c - 2.0*epsilon);

  X_test.resize(2,Domain<T>::_nEvalP*Domain<T>::_nEvalP);

  int k = 0;
  for(int i=0; i < Domain<T>::_nEvalP; i++)
    for(int j=0; j < Domain<T>::_nEvalP; j++){
      X_test(0,k) = xlinspace(i);
      X_test(1,k) = ylinspace(j);
      k++;
    }
  
}


template<typename T>
void Rectangle<T>::sampleFromInner(MatrixXXT& inner){
  inner          = MatrixXXT::Random(2,Domain<T>::_nIntP);
  
  T epsilon      = 10e-4;
  inner(0,all)   = (MatrixXXT::Constant(1,Domain<T>::_nIntP,1) - inner(0,all) )*.5  ;
  inner(0,all)   = MatrixXXT::Constant(1,Domain<T>::_nIntP, _a + epsilon) + inner(0,all) * (_b -_a - 2.0*epsilon);
  inner(1,all)   = (MatrixXXT::Constant(1,Domain<T>::_nIntP,1) - inner(1,all) )*.5  ;
  inner(1,all)   = MatrixXXT::Constant(1,Domain<T>::_nIntP, _c + epsilon) + inner(1,all) * (_d-_c - 2.0*epsilon);
}



template<typename T>
void Rectangle<T>::sampleFromInner(MatrixXXT& inner, int nIntP){
  inner          = MatrixXXT::Random(2,nIntP);
  
  T epsilon      = 10e-4;
  inner(0,all)   = (MatrixXXT::Constant(1,nIntP,1) - inner(0,all) )*.5  ;
  inner(0,all)   = MatrixXXT::Constant(1,nIntP, _a + epsilon) + inner(0,all) * (_b -_a - 2.0*epsilon);
  inner(1,all)   = (MatrixXXT::Constant(1,nIntP,1) - inner(1,all) )*.5  ;
  inner(1,all)   = MatrixXXT::Constant(1,nIntP, _c + epsilon) + inner(1,all) * (_d-_c - 2.0*epsilon);
}



template<typename T>
void Rectangle<T>::sampleFromBoundary(MatrixXXT& bdyL,  MatrixXXT& bdyB,   
                                      MatrixXXT& bdyR,  MatrixXXT& bdyT){
  
  T epsilon      = 10e-2;
  int nBdyP      = Domain<T>::_nBdyP;

  while(nBdyP%4 != 0)
    nBdyP++;
  
  Domain<T>::_nBdyP = nBdyP;
  
  bdyL      = MatrixXXT::Zero(2,nBdyP/4);
  bdyB      = MatrixXXT::Zero(2,nBdyP/4);    
  bdyR      = MatrixXXT::Zero(2,nBdyP/4);
  bdyT      = MatrixXXT::Zero(2,nBdyP/4);

  bdyL(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_a);
  bdyL(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  
  bdyB(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);
  bdyB(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_c);  

  bdyR(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_b);
  bdyR(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);

  bdyT(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);   
  bdyT(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_d);


}

template<typename T>
void Rectangle<T>::sampleFromBoundary(MatrixXXT& bdyLB, MatrixXXT& bdyB){

  T epsilon      = 10e-2;
  int nBdyP      = Domain<T>::_nBdyP;

  while(nBdyP%4 != 0)
    nBdyP++;
  
  Domain<T>::_nBdyP = nBdyP;
  
  bdyLB     = MatrixXXT::Zero(2,3*(nBdyP/4)); // Boundary less bottom
  bdyB      = MatrixXXT::Zero(2,nBdyP/4);

  MatrixXXT bdyL = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT bdyR = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT bdyT = MatrixXXT::Zero(2,nBdyP/4);

  
  bdyL(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_a);
  bdyL(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  bdyLB.block(0,0,2,nBdyP/4)  =  bdyL;
      
  bdyB(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);
  bdyB(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_c);  

  bdyR(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_b);
  bdyR(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  bdyLB.block(0,nBdyP/4,2,nBdyP/4)       =  bdyR;
  
  bdyT(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);   
  bdyT(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_d);
  bdyLB.block(0,2*(nBdyP/4),2,nBdyP/4)  = bdyT;
  
}

template<typename T>
void Rectangle<T>::sampleFromBoundary(MatrixXXT& bdy){

  T epsilon      = 10e-2;
  int nBdyP      = Domain<T>::_nBdyP;

  while(nBdyP%4 != 0)
    nBdyP++;
  
  Domain<T>::_nBdyP = nBdyP;
    
  bdy       = MatrixXXT::Zero(2,nBdyP); 

  MatrixXXT  bdyL      = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT  bdyB      = MatrixXXT::Zero(2,nBdyP/4);    
  MatrixXXT  bdyR      = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT  bdyT      = MatrixXXT::Zero(2,nBdyP/4);

  bdyL(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_a);
  bdyL(1,all) =  RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  // bdyL(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon); //RowVectorXT::Random(nBdyP/4,1); 
  // bdyL(1,all) =  RowVectorXT::Random(nBdyP/4,1); //_c+epsilon,_d-epsilon

  bdy.leftCols(nBdyP/4)       =  bdyL;
  
  //bdyB(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);
  bdyB(0,all) =  RowVectorXT::Constant(1,nBdyP/4,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;;
  bdyB(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_c);  
  bdy.block(0, nBdyP/4, 2, nBdyP/4 ) = bdyB;


  bdyR(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_b);

  //bdyR(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  bdyR(1,all) = RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  bdy.block(0, 2*(nBdyP/4), 2, nBdyP/4 ) =  bdyR;
  

  // bdyT(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);   

  bdyT(0,all) = RowVectorXT::Constant(1,nBdyP/4,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;;
  bdyT(1,all) = MatrixXXT::Constant(1,nBdyP/4,_d);
  bdy.block(0,3*(nBdyP/4), 2, nBdyP/4 ) = bdyT;

}



template<typename T>
void Rectangle<T>::sampleFromBoundary(MatrixXXT& bdy, int nBdyP){

  T epsilon      = 10e-2;

  while(nBdyP%4 != 0)
    nBdyP++;
    
  bdy       = MatrixXXT::Zero(2,nBdyP); 

  MatrixXXT  bdyL      = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT  bdyB      = MatrixXXT::Zero(2,nBdyP/4);    
  MatrixXXT  bdyR      = MatrixXXT::Zero(2,nBdyP/4);
  MatrixXXT  bdyT      = MatrixXXT::Zero(2,nBdyP/4);

  bdyL(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_a);
  bdyL(1,all) =  RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  // bdyL(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon); //RowVectorXT::Random(nBdyP/4,1); 
  // bdyL(1,all) =  RowVectorXT::Random(nBdyP/4,1); //_c+epsilon,_d-epsilon

  bdy.leftCols(nBdyP/4)       =  bdyL;
  
  //bdyB(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);
  bdyB(0,all) =  RowVectorXT::Constant(1,nBdyP/4,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;;
  bdyB(1,all) =  MatrixXXT::Constant(1,nBdyP/4,_c);  
  bdy.block(0, nBdyP/4, 2, nBdyP/4 ) = bdyB;


  bdyR(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_b);

  //bdyR(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  bdyR(1,all) = RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  bdy.block(0, 2*(nBdyP/4), 2, nBdyP/4 ) =  bdyR;
  

  // bdyT(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);   

  bdyT(0,all) = RowVectorXT::Constant(1,nBdyP/4,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;;
  bdyT(1,all) = MatrixXXT::Constant(1,nBdyP/4,_d);
  bdy.block(0,3*(nBdyP/4), 2, nBdyP/4 ) = bdyT;

}






template<typename T>
void Rectangle<T>::sampleFromBottomBoundary(MatrixXXT& bdy, int nBdyP){

  // T epsilon      = 10e-2;
  // while(nBdyP%4 != 0)
  //   nBdyP++;
    
  bdy  = MatrixXXT::Zero(2,nBdyP); 
  
  // MatrixXXT  bdyL      = MatrixXXT::Zero(2,nBdyP/4);
  // MatrixXXT  bdyB  = MatrixXXT::Zero(2,nBdyP);    
  // MatrixXXT  bdyR      = MatrixXXT::Zero(2,nBdyP/4);
  // MatrixXXT  bdyT      = MatrixXXT::Zero(2,nBdyP/4);
  // bdyL(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_a);
  // bdyL(1,all) =  RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  // bdyL(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon); //RowVectorXT::Random(nBdyP/4,1); 
  // bdyL(1,all) =  RowVectorXT::Random(nBdyP/4,1); //_c+epsilon,_d-epsilon

  // bdy.leftCols(nBdyP/4)       =  bdyL;
  
  //bdyB(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);
  bdy(0,all) =  RowVectorXT::Constant(1,nBdyP,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP) + RowVectorXT::Constant(1,nBdyP, 1.0))  ;
  bdy(1,all) =  MatrixXXT::Constant(1,nBdyP,_c);  

  // bdy         =  bdyB;
  // bdyR(0,all) =  MatrixXXT::Constant(1,nBdyP/4,_b);

  // //bdyR(1,all) =  RowVectorXT::LinSpaced(nBdyP/4,_c+epsilon,_d-epsilon);
  // bdyR(1,all) = RowVectorXT::Constant(1,nBdyP/4,_c) +  0.5*(_d -_c)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0))  ;
  // bdy.block(0, 2*(nBdyP/4), 2, nBdyP/4 ) =  bdyR;
  

  // // bdyT(0,all) =  RowVectorXT::LinSpaced(nBdyP/4,_a+epsilon,_b-epsilon);   

  // bdyT(0,all) = RowVectorXT::Constant(1,nBdyP/4,_a) +  0.5*(_b -_a)* (RowVectorXT::Random(1,nBdyP/4) + RowVectorXT::Constant(1, nBdyP/4, 1.0)) ;
  // bdyT(1,all) = MatrixXXT::Constant(1,nBdyP/4,_d);
  // bdy.block(0,3*(nBdyP/4), 2, nBdyP/4 ) = bdyT;

}


  
// }
