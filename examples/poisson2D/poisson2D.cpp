// main.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>

#include <nn/NeuralNetwork.hpp>
#include <fterm/FTerm.hpp>
#include <fterm/poisson2D/DivTerm.hpp>
#include <fterm/poisson2D/BdyTerm.hpp>
#include <domain/Rectangle.hpp>
#include <utils/WriteFile.hpp>
#include <utils/GNUplot.hpp>
#include <time.h>
#include <ctime>

using namespace nn;
using namespace domain;
using namespace utils;
using namespace fterm;

typedef Matrix<float,Dynamic,Dynamic> MatrixXXf;


double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks=clock1-clock2;
  double diffms=(diffticks*10)/CLOCKS_PER_SEC;
  return diffms;
}

template<typename T>
T f_function(T x, T y) {
    
  T z = (2.0*(1.25 + cos(5.4*y)) * pow(108.0*x-36.0,2) ) / ( pow(6.0+6.0*pow(3.0*x-1.0,2),3) )
    - (108.0*(1.25 + cos(5.4*y))) / (pow( 6.0+6.0*pow(3.0*x-1,2),2))
    - (29.16*cos(5.4*y)) / (6.0+6.0*pow(3.0*x-1.0,2));
  
  return z;
}

template<typename T>
T g_function(T x, T y) {  
  return  (1.25 + cos(5.4*y))  / (6.0+6.0*(3.0*x-1.0)*(3.0*x-1.0)); 
}

template<typename T, typename F>
void get_Y_train(Matrix<T,Dynamic,Dynamic> & X_train,  Matrix<T,Dynamic,Dynamic> & Y_train, F f ){
  
  int n = X_train.cols();  
  Y_train.resize(1,n);

  for(int i=0; i < n; i++)
    Y_train(0,i) = f(X_train(0,i),X_train(1,i));  
}

template<typename T, typename F>
void get_Y_exact(Matrix<T,Dynamic,Dynamic> & X_train,  Matrix<T,Dynamic,Dynamic> & Y_exact, F f ){
  
  int n = X_train.cols();  
  Y_exact.resize(1,n);

  for(int i=0; i < n; i++)
    Y_exact(0,i) = f(X_train(0,i),X_train(1,i));  
}


FTerm<float>* loss[1];  

MatrixXXf inner_X_train,inner_Y_train;
MatrixXXf bdy_X_train,bdy_Y_train;  
MatrixXXf X_test,Y_test,Y_exact;


template <typename T>
void train(Rectangle <float> & rectangle, FTerm<float> * divterm,
           FTerm<float> * bdyterm, int nMaxIter= 1000, T tol = 1e-2) {
  
  T error ;
  int i=0;
  int n = -3.0;
  T alpha;
  clock_t begin = clock();
  
  
  cout << "-----Start training---------" << endl;
  do {
     
    loss[0]->train();
    // loss[1]->train();
    error =  loss[0] -> cost();
    // error += loss[1] -> cost();
    
    
    // cout << "---- Resampling ----" << endl;
    rectangle.sampleFromInner(inner_X_train);
    // rectangle.sampleFromBoundary(bdy_X_train);
    get_Y_train(inner_X_train,inner_Y_train,f_function<float>);
    // get_Y_train(bdy_X_train,bdy_Y_train,g_function<float>);

     
    if(i%100 == 0)
      {
        cout << "set Alpha" << endl;
        alpha = float(pow(10.0,n));
        n++;
        dynamic_cast < DivTerm<float>* >(divterm) -> setAlpha(alpha);
        //     cout << "---- Resampling ----" << endl;
        //     rectangle.sampleFromInner(inner_X_train);
        //     // rectangle.sampleFromBoundary(bdy_X_train);
        //     get_Y_train(inner_X_train,inner_Y_train,f_function<float>);
        //     // get_Y_train(bdy_X_train,bdy_Y_train,g_function<float>);
        
        //     // cout << "-----------------" << endl;       
        //     // cout << inner_X_train << endl;
        //     // cout << bdy_X_train << endl;
        //     // cout << "-----------------" << endl;
        
        //     divterm -> setTrainData(&inner_X_train,&inner_Y_train);
        //     // bdyterm -> setTrainData(&bdy_X_train,&bdy_Y_train);
        
        divterm -> setTrainData(&inner_X_train,&inner_Y_train);
        // bdyterm -> setTrainData(&bdy_X_train,&bdy_Y_train);
      }
     
     // if(i%10 == 0)
    cout << "Iteration " << i << "  Error: " << error  << endl;
    
    i++;
  }while( i < nMaxIter);
  
  
  cout << "-------Finish training---------" << endl;
  

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  cout << "Train elapased time " <<  elapsed_secs << "sec." << endl;
}

template<typename T>
T rmse(Matrix<float,Dynamic,Dynamic> &Y_test, Matrix<float,Dynamic,Dynamic> &Y_exact) {  

  T error;
  Matrix<float,Dynamic,Dynamic> errors = Y_test - Y_exact; 
  errors = errors.cwiseAbs2();  
  error = sqrt( errors.sum()/errors.size() );
  // cout << error << endl;
  return error;
}


int main() {

  int nMaxIter = 1000;
  float tol   = 1e-3;
  int nEvalP  = 50;
  int nIntP   = 100;
  int nBdyP   = 20;
  
  float a     = 0;
  float b     = 1;
  float c     = 0;
  float d     = 1;
  float delta_x = 0.001;
  float delta_y = 0.001;

  int n;
  int m;

  
  //plotter 
  GNUplot plotter;  
  string pnamefile = "./data/poisson_prediction_data.txt";
  string lnamefile = "./data/poisson_learning_data.txt";
  string enamefile = "./data/poisson_exact_data.txt";
  string s;
  
  //Problem Domain
  Rectangle <float> rectangle(a,b,c,d,nEvalP,nIntP,nBdyP);
  
  WriteFile<float> writter;

  rectangle.sampleFromInner(inner_X_train);
  rectangle.sampleFromBoundary(bdy_X_train);

  get_Y_train(inner_X_train,inner_Y_train,f_function<float>);
  get_Y_train(bdy_X_train,bdy_Y_train,g_function<float>);


  vector<int> architecture = {2,4,4,4,1};
  // NeuralNetwork<float> net(architecture,0.01, NeuralNetwork<float>::Activation::TANH);
  NeuralNetwork<float> net(architecture,0.01, NeuralNetwork<float>::Activation::SIGMOID);


  
  float penalty = 500;

  FTerm<float> * divterm = new DivTerm<float>(&inner_X_train,&inner_Y_train,&net,delta_x,delta_y,f_function,rectangle.area());  
  FTerm<float> * bdyterm = new BdyTerm<float>(&bdy_X_train,&bdy_Y_train,&net,g_function,rectangle.perimeter(),penalty);  
  
  loss[0] = divterm;
  // loss[1] = bdyterm;
  
  train<float>(rectangle,divterm,bdyterm,nMaxIter,tol);
  net.save("./data/params.txt");  
  
  rectangle.uniformSampleFromInner(X_test);
  net.eval(X_test,Y_test);
  get_Y_exact(X_test,Y_exact,g_function<float>);
  
    
  writter.write(inner_X_train,inner_Y_train,lnamefile);
  writter.write(X_test,Y_test,pnamefile);
  writter.write(X_test,Y_exact,enamefile);
  
  cout << "RMSE: " << rmse<float>(Y_test,Y_exact) << endl;
  
  // cout << endl << "Neurons:" << endl;
  // for(int i = 0; i < net.mNeurons.size(); i++)
  //   cout << *net.mNeurons[i] << endl;
  // cout << endl << "Weights:" << endl;
  // for (int i = 0; i < net.mWeights.size(); i++)
  //   cout << *net.mWeights[i] << endl;
  

  s = "set title "+string("\"")+"Clasification Zones"+string("\"")+";"+ "splot "+ string("\'")+ pnamefile+string("\'")+" using 1:2:3 with points,"+ string("\'")+ enamefile+string("\'")+" using 1:2:3 with points"+";";

  // s = "set title "+string("\"")+"Clasification Zones"+string("\"")+";"+ "set palette model RGB defined "+ string("(")+"0"+string(" ")+ string("\"")+"red"+string("\"")+string(",")+"1"+string(" ")+string("\"")+"blue"+string("\"")+string(")") + ";" + "splot "+ string("\'")+ pnamefile+string("\'")+" using 1:2:3 with points pt 2 palette  notitle,"+ string("\'")+ lnamefile+string("\'")+" using 1:2:3 with points pt 7  ps 1 palette title "+string("\"")+"train data"+string("\"");
  
  plotter(s.c_str());
  int pause = cin.get();

  return 0;
}
