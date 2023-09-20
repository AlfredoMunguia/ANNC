#include <iostream> 
#include <fstream> 

template <typename T>
NeuralNetwork<T>::NeuralNetwork() {
  mConfusion = nullptr;
}


template <typename T>
NeuralNetwork<T>::~NeuralNetwork() {
  if (mConfusion)
    delete mConfusion;
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(vector<int> architecture, T learningRate /*= LEARNING_RATE*/, Activation activation /*= TANH*/) {

  init(architecture, learningRate, activation);
  _input.resize(mArchitecture[0]);
  _output.resize(mArchitecture[mArchitecture.size()-1]);

}


template <typename T>
void NeuralNetwork<T>::init(vector<int> architecture, T learningRate /*= LEARNING_RATE*/, Activation activation/*= TANH*/) {

  mArchitecture = architecture;
  mLearningRate = learningRate;
  mActivation   = activation;
  
  mNeurons.clear();
  mErrors.clear();
  mWeights.clear();

  _a.clear();
  // _delta.clear();
  _b.clear();
  _W.clear();
  _D.clear();

  int nL   = architecture[architecture.size()-1];
  _nInput  = architecture[0];
  _nOutput = nL;
  
  for (unsigned int i = 0; i < architecture.size(); i++) {
    _a.push_back(new RowVectorXT(architecture[i]));      

    if(i < architecture.size() - 1){

      _b.push_back(new RowVectorXT(architecture[i+1]));      
      _W.push_back(new MatrixXXT(architecture[i], architecture[i+1]));

      _D.push_back(new MatrixXXT(architecture[i+1], architecture[i+1]));
      _B.push_back(new MatrixXXT(architecture[i+1],nL));

      // initialize weights and bias
      _W.back()->setRandom();
      _b.back()->setRandom();
      
    }
  }

  _Lambda.resize(nL,nL);
  _Lambda.diagonal() = RowVectorXT::Ones(nL);
  
  mConfusion = new MatrixXXT(architecture.back(), architecture.back());
  mConfusion->setZero();
}


template <typename T>
T NeuralNetwork<T>::activation(T x) {
  if (mActivation == TANH)
    return tanh(x);
  if (mActivation == SIGMOID)
     return 1.0 / (1.0 + exp(-x));
  return 0;
}


template <typename T>
T NeuralNetwork<T>::activationDerivative(T x) {
  if (mActivation == TANH)
    return 1 - tanh(x) * tanh(x);
  if (mActivation == SIGMOID)
    return x * (1.0 - x);
  return 0;
}


template <typename T> void NeuralNetwork<T>::forward(RowVectorXT& input) {
  // set first layer input
  *_a.front() = input;
    
  // propagate forward (vector multiplication)
  for (int i = 1; i < mArchitecture.size(); i++) {
    *_a[i] = *_a[i-1] * *_W[i-1]  + *_b[i-1];

    // cout << *_a[i] << endl; 
    //apply activation function
    for (int col = 0; col < mArchitecture[i]; col++){      
      _a[i]   -> coeffRef(col)     = activation(_a[i]->coeffRef(col));
      // cout << "<--->" << *_a[i] << endl << endl;
      _D[i-1] -> coeffRef(col,col) = activationDerivative(_a[i]->coeffRef(col));
    }      
  }    
}


template <typename T> void NeuralNetwork<T>::backward() {
  int L = mArchitecture.size() - 2;
  *_B[L]       =  *_D[L];           
  for (int i = L - 1; i >= 0; i--){
    *_B[i]     =   ( *_D[i] * *_W[i+1]) * *_B[i+1];      
  }  
}

template<typename T> void NeuralNetwork<T>::eval(MatrixXXT& input, MatrixXXT& output) {

  RowVectorXT inputByLayer(mArchitecture[0]);
  RowVectorXT outputByLayer(mArchitecture[mArchitecture.size()-1]);

  output.resize( mArchitecture[mArchitecture.size()-1] ,  input.cols() );
  
  for(int j=0; j < input.cols(); ++j){
    
    for(int i=0; i < input.rows(); ++i){
      inputByLayer.coeffRef(i) = input(i,j);
    }
    
    forward(inputByLayer);
    outputByLayer = *_a.back();
    
    for(int i=0; i < output.rows(); ++i){
      output(i,j) = outputByLayer.coeffRef(i);
    }            
  }
}


template<typename T>
void NeuralNetwork<T>::eval(RowVectorXT& input, RowVectorXT& output) {
  forward(input);
  output = *_a.back();
}


template <typename T>
void NeuralNetwork<T>::test(RowVectorXT& input, RowVectorXT& output) {
  forward(input);
  // calculate last layer errors
  _error = output - *_a.back();
}


template <typename T>
void NeuralNetwork<T>::resetConfusion() {
  if (mConfusion)
    mConfusion->setZero();
}


template <typename T>
void NeuralNetwork<T>::evaluate(RowVectorXT& output) {
  T desired = 0, actual = 0;
  mConfusion->coeffRef(vote(output, desired),vote(*mNeurons.back(), actual))++;
}

template <typename T>
void NeuralNetwork<T>::confusionMatrix(RowVectorXT*& precision, RowVectorXT*& recall) {
  int rows = (int)mConfusion->rows();
  int cols = (int)mConfusion->cols();
  
  precision = new RowVectorXT(cols);
  for (int col = 0; col < cols; col++) {
    T colSum = 0;
    for (int row = 0; row < rows; row++)
      colSum += mConfusion->coeffRef(row, col);
    precision->coeffRef(col) = mConfusion->coeffRef(col, col) / colSum;
  }
  
  recall = new RowVectorXT(rows);
  for (int row = 0; row < rows; row++) {
    T rowSum = 0;
    for (int col = 0; col < cols; col++)
      rowSum += mConfusion->coeffRef(row, col);
    recall->coeffRef(row) = mConfusion->coeffRef(row, row) / rowSum;
  }
  
  // convert confusion to percentage 
  for (int row = 0; row < rows; row++) {
    T rowSum = 0;
    for (int col = 0; col < cols; col++)
      rowSum += mConfusion->coeffRef(row, col);
    for (int col = 0; col < cols; col++)
      mConfusion->coeffRef(row, col) = mConfusion->coeffRef(row, col) * 100 / rowSum;
  }
}


// template <typename T>
// void NeuralNetwork<T>::train() {   
//   int r = rand()%_nTrain;  
//   _input  =  _X_train(all,r).transpose();
//   _output =  _Y_train(all,r).transpose();  
//   train(_input,_output);  
// }


// template <typename T>
// void NeuralNetwork<T>::train() {
  
//   for(int i=0; i < loss.size(); i++){
//     loss[i]->getXTrain();
//     forward(loss[i]);   
//     backward(loss[i]);  
//     update(loss[i]);
//   }
// }


// template <typename T>
// void NeuralNetwork<T>::train(RowVectorXT& input, RowVectorXT& output) {
//   forward(input);  
//   backward();      
//   update(output);  
// }

// template <typename T>
// T NeuralNetwork<T>::cost(){

//   RowVectorXT approx;
//   RowVectorXT exact;
//   RowVectorXT error;
    
//   for(int i=0; i < _nTrain; ++i){
//     approx = _X_train(all,i).transpose();

//     forward(approx);    
//     approx     = *_a.back();
//     exact      = _Y_train(all,i).transpose();
//     error      = exact-approx;
//     _errors(i)  = (error).dot(error);
//   }

//   return 0.5*(_errors.sum()/ _errors.size());
  
// }


// template <typename T>
// T NeuralNetwork<T>::cost(){

//   T cost = 0.0;
  
//   for(int i=0; i < loss.size(); i++){
    
//     costForward(loss[i]);    
//     cost += loss[i]->cost();
//   }
  
//   return cost;  
// }



// template <typename T>
// void NeuralNetwork<T>::setLossTerm(FTerm<T>* term){
//   loss.push_back(term);
// }


// mean square error
template <typename T>
T NeuralNetwork<T>::mse() {    
  return sqrt((_error).dot((_error)) / _error.size());
}



template <typename T>
int NeuralNetwork<T>::vote(T& value) {
  auto it = mNeurons.back();
  return vote(*it, value);
}

template <typename T>
int NeuralNetwork<T>::vote(RowVectorXT& v, T& value) {
  int index = 0;
  for (int i = 1; i < v.cols(); i++)
    if (v[i] > v[index])
      index = i;
  value = v[index];
  return index;
}

template <typename T>
T NeuralNetwork<T>::output(int col) {
  auto it = mNeurons.back();
  return (*it)[col];
}

template <typename T>
void NeuralNetwork<T>::save(const char* filename) {
  stringstream tplgy;
  for (auto it = mArchitecture.begin(), _end = mArchitecture.end(); it != _end; it++)
    tplgy << *it << (it != _end - 1 ? "," : "");
  
  stringstream wts;
  for (auto it = mWeights.begin(), _end = mWeights.end(); it != _end; it++)
    wts << **it << (it != _end - 1 ? "," : "") << endl;
  
  ofstream file(filename);
  file << "learningRate: " << mLearningRate << endl;
  file << "architecture: " << tplgy.str() << endl;
  file << "activation: " << mActivation << endl;
  file << "weights: " << endl << wts.str() << endl;
  file.close();
}

template <typename T>
bool NeuralNetwork<T>::load(const char* filename) {
  mArchitecture.clear();
  
  ifstream file(filename);
  if (!file.is_open())
    return false;
  string line, name, value;
  if (!getline(file, line, '\n'))
    return false;
  stringstream lr(line);
  
  // read learning rate
  getline(lr, name, ':');
  if (name != "learningRate")
    return false;
  if (!getline(lr, value, '\n'))
    return false;
  mLearningRate = atof(value.c_str());
  
  // read topoplogy
  getline(file, line, '\n');
  stringstream ss(line);
  getline(ss, name, ':');
  if (name != "architecture")
    return false;
  while (getline(ss, value, ','))
    mArchitecture.push_back(atoi(value.c_str()));
  
  // read activation
  getline(file, line, '\n');
  stringstream sss(line);
  getline(sss, name, ':');
  if (name != "activation")
    return false;
  if (!getline(sss, value, '\n'))
    return false;
  mActivation = (Activation)atoi(value.c_str());
  
  // initialize using read architecture
  init(mArchitecture, mLearningRate, mActivation);
  
  // read weights
  getline(file, line, '\n');
  stringstream we(line);
  getline(we, name, ':');
  if (! (name.compare("weights") == 0) )
    return false;
  
  string matrix;
  for (int i = 0; i < mArchitecture.size(); i++)
    if (getline(file, matrix, ',')) {
      stringstream ss(matrix);
      int row = 0;
      while (getline(ss, value, '\n'))
        if (!value.empty()) {
          stringstream word(value);
          int col = 0;
          while (getline(word, value, ' '))
            if (!value.empty())
              mWeights[i]->coeffRef(row, col++) = atof(value.c_str());
          row++;
        }
    }
  
  file.close();
  return true;
}



