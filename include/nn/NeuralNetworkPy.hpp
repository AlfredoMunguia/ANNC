#include <Eigen/Eigen> 
#include <vector> 
#include <iostream> 
#include <fstream> 

using namespace std;
using namespace Eigen;

namespace nn{

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
  
  
#define LEARNING_RATE   0.01

  template <typename T>  
  class NeuralNetwork {
    
    typedef Matrix<T,1,Dynamic> RowVectorXT;
    typedef Matrix<T,Dynamic,Dynamic> MatrixXXT;
    
  public:
    enum Activation { TANH, SIGMOID } mActivation;
    // network learning rate
    T mLearningRate;
    // network layers architecture (input-hidden-output) 
    vector<int> mArchitecture;

    // network layers vectors
    vector<RowVectorXT*> mNeurons;
    vector<RowVectorXT*> mErrors;    
    vector<MatrixXXT*> mWeights;

    // network layers vectors
    vector<RowVectorXT*> _a;    
    RowVectorXT _error;
    
    // neurons' output errors    
    // vector<RowVectorXT*> _delta;
    vector<RowVectorXT*> _b;
    
    // connections' weights
    vector<MatrixXXT*> _W;
    vector<MatrixXXT*> _D;
    vector<MatrixXXT*> _B;


    MatrixXXT _Lambda;

    RowVectorXT _input;
    RowVectorXT _output;
    RowVectorXT _errors;

    int         _nTrain;
    int         _nCTrain;    
    int         _nInput;
    int         _nOutput;

    // confusion matrix
    MatrixXXT* mConfusion;

    // constructors
    NeuralNetwork();    
    NeuralNetwork(vector<int> architecture,T learningRate = LEARNING_RATE,Activation = TANH);

    ~NeuralNetwork();    

    void init(vector<int> architecture,T learningRate = LEARNING_RATE,Activation = TANH);

    // load from file
    bool load(const char* file);
    // save to file
    void save(const char* file);
    // data forward propagation
    void forward(RowVectorXT & input);

    // backward propagation
    void backward();
    
    // void update();    
    // void update(RowVectorXT & output);    
    // void update(FTerm<T>* fterm);
    
    // void setLossTerm(FTerm<T>*);

    void eval(MatrixXXT& input, MatrixXXT& output);
    void eval(RowVectorXT& input, RowVectorXT& output);
              
    T activation(T x);    
    T activationDerivative(T x);

    void setTrainData(MatrixXXT& X_train, MatrixXXT& Y_train, int nTrain);
    
    // train the neural network given an input
    // void train(RowVectorXT & input, RowVectorXT& output);

    // train the neural network 
    //void train();

    // train all train data
    // void train();

    vector<int> * getArchitecture(){return &mArchitecture;}
    
    //cost function
    T cost();

    //void setTrainData();
    
    // test the neural network given an input
    void test(RowVectorXT& input, RowVectorXT& output);
  
    void resetConfusion();
    void evaluate(RowVectorXT& output);
    void confusionMatrix(RowVectorXT*& precision, RowVectorXT*& recall);

    // get max output index
    int vote(T& value);
    int vote(RowVectorXT& v, T& value);
  
    // get last layer output
    T output(int col);
    
    // get output layer mean squere error
    T mse();

    //friend class FTerm;

    void setLearningRate(T lr){mLearningRate = lr;}
    T    getLearningRate(){return mLearningRate;}
    
  };



#include "../src/nn/NeuralNetwork.cpp"

#endif
  
}
