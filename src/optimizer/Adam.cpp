template <typename T>
Adam<T>::Adam(NeuralNetwork<T> *nn){

  _beta1 = 0.9;
  _beta2 = 0.999;
  _epsilon = 10e-8;

  Optimizer<T>::_nn  = nn;
  _architecture = Optimizer<T>::_nn ->getArchitecture();
  
  // Optimizer<T>::_loss = loss;
  // Optimizer<T>::_nTerms = nTerms;  
  // for(int i=0; i < nTerms; ++i){    
  //   // _gradb        = dynamic_cast < LSTerm<T>* >( Optimizer<T>::_loss + i ) -> getBias();
  //   // _gradW        = dynamic_cast < LSTerm<T>* >( Optimizer<T>::_loss + i ) -> getWeights();
  //   // _gradb        = dynamic_cast < Optimizer<T>::_loss+i >( Optimizer<T>::_loss + i ) -> getBias();
  //   // _gradW        = dynamic_cast < LSTerm<T>* >( Optimizer<T>::_loss + i ) -> getWeights();
  // }

    
  
  for (unsigned int l = 0; l < _architecture->size(); l++) {
    if(l < _architecture->size() - 1){
      _mtb.push_back(new RowVectorXT( (*_architecture)[l+1]));      
      _mtW.push_back(new MatrixXXT((*_architecture)[l],(*_architecture)[l+1]));
      _mtb[l] -> setZero();
      _mtW[l] -> setZero();
      
      _vtb.push_back(new RowVectorXT( (*_architecture)[l+1]));      
      _vtW.push_back(new MatrixXXT((*_architecture)[l],(*_architecture)[l+1]));
      _vtb[l] -> setZero();
      _vtW[l] -> setZero();
     
      
      _hmtb.push_back(new RowVectorXT( (*_architecture)[l+1]));      
      _hmtW.push_back(new MatrixXXT((*_architecture)[l],(*_architecture)[l+1]));
      
      _hvtb.push_back(new RowVectorXT( (*_architecture)[l+1]));      
      _hvtW.push_back(new MatrixXXT((*_architecture)[l],(*_architecture)[l+1]));

    }
  }
  
}

template <typename T>
void Adam<T>::update(vector<RowVectorXT*>* gradb,
                     vector<MatrixXXT*>* gradW, int t){
  T _alpha = .01;

  for(int l=0; l < (*_architecture).size() -1; ++l){

    *_mtb[l] = _beta1 * *_mtb[l] + (1.0-_beta1)* *((*gradb)[l]);
    *_mtW[l] = _beta1 * *_mtW[l] + (1.0-_beta1)* *((*gradW)[l]);

    *_vtb[l]  = _beta2 * *_vtb[l] + (1.0-_beta2)* ((*gradb)[l])->cwiseAbs2();
    *_vtW[l]  = _beta2 * *_vtW[l] + (1.0-_beta2)* ((*gradW)[l])->cwiseAbs2();

    *_hmtb[l] = *_mtb[l]/(1.0-pow(_beta1,t+1));
    *_hmtW[l] = *_mtW[l]/(1.0-pow(_beta1,t+1));

    *_hvtb[l] = *_vtb[l]/(1.0-pow(_beta2,t+1));
    *_hvtW[l] = *_vtW[l]/(1.0-pow(_beta2,t+1));

    
    Optimizer<T>::_nn->_b[l]->array() = Optimizer<T>::_nn->_b[l]->array() -_alpha*_hmtb[l]->array() / (_hvtb[l]->array().sqrt() + _epsilon) ;
    Optimizer<T>::_nn->_W[l]->array() = Optimizer<T>::_nn->_W[l]->array() -_alpha*_hmtW[l]->array() / (_hvtW[l]->array().sqrt() + _epsilon) ;

    
  }
  // cout << *(*_gradb)[0] << endl;

}

