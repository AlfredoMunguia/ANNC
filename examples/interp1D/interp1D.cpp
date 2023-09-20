// main.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <nn/NeuralNetwork.hpp>
#include <fterm/FTerm.hpp>
#include <fterm/interp1D/LSTerm.hpp>
#include <utils/WriteFile.hpp>
#include <utils/GNUplot.hpp>
#include <time.h>
#include <ctime>
#include <optimizer/Adam.hpp>
#include <scheduler/Scheduler.hpp>


using namespace nn;
//using namespace domain;
using namespace utils;
using namespace fterm;
using namespace optimizer;
using namespace scheduler;


double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks=clock1-clock2;
  double diffms=(diffticks*10)/CLOCKS_PER_SEC;
  return diffms;
}

template <typename T>
void train(FTerm<T>* loss, NeuralNetwork<T> *net,  int nMaxIter= 1000, T tol = 1e-2) {
  
  T error ;
  int i=0;
 
  clock_t begin = clock();

  Adam<T> adam(net);
  // Scheduler<T> scheduler(net,200);
  
  cout << "-----Start training---------" << endl;  
  do {
   
    loss[0].train();  

    dynamic_cast < LSTerm<T>* >(loss) -> gradient();  
    adam.update(dynamic_cast < LSTerm<T>* >(loss) -> getBias(),
                dynamic_cast < LSTerm<T>* >(loss) -> getWeights(),i);
    // scheduler.update(i);
    
    error = loss[0].cost();
    if(i%100 == 0)
      cout << "Iteration " << i << "  Error: " << error  << endl;
    
    i++;
  }while(i < nMaxIter);
  
  cout << "-------Finish training---------" << endl;
  

  clock_t end = clock();
  //cout << "Sum: " << sum << " Time elapsed: " << double(diffclock(end,begin)) << " ms" << endl;
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  cout << "Train elapased time " <<  elapsed_secs << "sec." << endl;
}


template<typename T>
void test(NeuralNetwork<T>& net, Matrix<T,Dynamic,Dynamic> &X_test, Matrix<T,Dynamic,Dynamic> &Y_test) {
  
  int n,m;
  RowVectorXf input(1);
  RowVectorXf output(1);
  
  n = X_test.rows();
  m = X_test.cols();

  Y_test.resize(n,m);


  for(int i=0; i < n; i++){
    for(int j=0; j < m; j++){
      input(0) = X_test(i,j);
      net.eval(input,output);
      Y_test(i,j) = output(0);
    }
  }

}


int main() {

  int nMaxIter = 20000;
  float tol    = 1e-3;
  int nTrain   = 40; //1024;
  int nEval    = 100;
  float a      = -M_PI;
  float b      = M_PI;

  float epsilon = 1e-4;
  
  float penalty1 = 100;
  float penalty2 = 100;
  
  
  //plotter 
  GNUplot plotter;  
  string pnamefile = "./data/interp1D_prediction_data.txt";
  string lnamefile = "./data/interp1D_learning_data.txt";
  string enamefile = "./data/interp1D_exact_data.txt";
  string s;
  

  // typedef Matrix<float,Dynamic,Dynamic> MatrixXXf;
  // typedef Matrix<float,Dynamic> RowVectorXXf;


  Matrix<float,Dynamic,Dynamic> X_test,Y_test,Y_exact;
  MatrixXf merror;
  float error;
  MatrixXf X_train,Y_train;



  X_train = a*RowVectorXf::Ones(nTrain) + (b-a)*( 0.5 * (RowVectorXf::Random(nTrain) + RowVectorXf::Ones(nTrain)) );
  // X_train.resize(1,10);
  // X_train(0,0) = -1.0;
  // X_train(0,1) =  0.0;
  Y_train = X_train.array().sin();

  // cout << X_train << endl;
  // cout << Y_train << endl;
  
  WriteFile<float> writter;
  
  vector<int> architecture = {1, 4, 4, 1};
  NeuralNetwork<float> net(architecture,0.01, NeuralNetwork<float>::Activation::TANH);

  FTerm<float>* loss [1];  
  FTerm<float>* lsterm = new LSTerm<float>(&X_train,&Y_train,&net);  

  loss[0] = lsterm;

  train<float>(*loss,&net,nMaxIter,tol);
   
  X_test  = RowVectorXf::LinSpaced(nEval,a,b);
  // cout << X_test << endl;
  Y_exact = X_test.array().sin(); 
  
  test<float>(net,X_test,Y_test);
  
  // cout << Y_test << endl;

  writter.write(X_train,Y_train,lnamefile);
  writter.write(X_test,Y_test,pnamefile);
  writter.write(X_test,Y_exact,enamefile);

  merror =  Y_test - Y_exact;
  merror = merror * merror.transpose() / merror.size();
  
  cout << "Relative error "<<  sqrt(merror(0)) << endl;

  
  // // net.save("./data/params.txt");  


  // // // cout << endl << "Neurons:" << endl;
  // // // for(int i = 0; i < net.mNeurons.size(); i++)
  // // //   cout <
  // // // *net.mNeurons[i] << endl;
  // // // cout << endl << "Weights:" << endl;
  // // // for (int i = 0; i < net.mWeights.size(); i++)
  // // //   cout << *net.mWeights[i] << endl;
  


  s = "set title "+string("\"")+"Interpolation "+string("\"")+";" + "plot "+ string("\'")+ pnamefile+string("\'")+" w p pt 1"+ ","+ string("\'")+ enamefile+string("\'")+"  w p pt 6 ";
  
  plotter(s.c_str());
  int pause = cin.get();

    

  return 0;
}
