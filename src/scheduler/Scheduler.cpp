template <typename T>
Scheduler<T>::Scheduler(NeuralNetwork<T> *nn, int step_size, T gamma){

  Scheduler<T>::_nn  =  nn;
  _step_size         =  step_size;
  _gamma             =  gamma;
  _lr                =  Scheduler<T>::_nn->getLearningRate();
  _update            =  true;
  _factor            =  0;
  _n                 =  0;
}

template <typename T>
void Scheduler<T>::update(int t){
      
  if(t == (_factor+1)*_step_size ){
    
    _lr *=  pow(10,--_n);
    _update = true;
    
    cout << t << "  " << _lr << endl;
    Scheduler<T>::_nn->setLearningRate(_lr);
    _factor++;
    
  }
  
  // cout << _lr * pow(10,n) << endl;  
  // Scheduler<T>::_nn->setLearningRate(_lr);
}
