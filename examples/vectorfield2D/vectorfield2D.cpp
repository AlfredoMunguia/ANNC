// main.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>

#include <nn/NeuralNetwork.hpp>
#include <fterm/FTerm.hpp>
#include <fterm/vectorfield2D/LSTerm.hpp>
#include <fterm/vectorfield2D/DivTerm.hpp>
#include <fterm/vectorfield2D/BdyTerm.hpp>
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
void u0(T x, T y, T& u0, T& u1) {
  u0 = x;
  u1 = 0.0;
}

template<typename T>
void ue(T x, T y, T& u0, T& u1) {  
  u0 = x;
  u1 = -y;
}


template<typename T, typename F>
void get_Y(Matrix<T,Dynamic,Dynamic> & X,  Matrix<T,Dynamic,Dynamic> & Y, F f ){
  
  int n = X.cols();  
  Y.resize(2,n);

  for(int i=0; i < n; i++)
    f(X(0,i),X(1,i),Y(0,i),Y(1,i) );  

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


FTerm<float>* loss[1];  

MatrixXXf inner_X_train_1, inner_Y_train_1;
MatrixXXf inner_X_train_2, inner_Y_train_2;
MatrixXXf bdy_X_train, bdy_Y_train;  

MatrixXXf X_test,Y_test,Y_exact;
int nIntP_1   = 1024;
int nIntP_2   = 500;
int nBdyP     = 500;
  


template <typename T>
void train(Rectangle <float> & rectangle, int nMaxIter= 1000, T tol = 1e-2) {
  
  T error ;
  int i=0;
  clock_t begin = clock();

  
  cout << "-----Start training---------" << endl;
  do {
    
    loss[0]->train();

    // loss[1]->train();     
    // loss[2]->train();     
    
    error =  loss[0] -> cost();     
    // error += loss[1] -> cost();
    // error += loss[2] -> cost();

    // if(i%100 == 0)
      {
        
        // rectangle.sampleFromInner(inner_X_train_1,nIntP_1);  
        // get_Y(inner_X_train_1,inner_Y_train_1,u0<float>);
        // // cout << inner_X_train_1 << endl;
        // // cout << inner_Y_train_1 << endl;
        // loss[0] -> setTrainData(&inner_X_train_1,&inner_Y_train_1);
        
        // rectangle.sampleFromInner(inner_X_train_2,nIntP_2);
        // loss[1] -> setTrainData(&inner_X_train_2);
        
        
        // rectangle.sampleFromBottomBoundary(bdy_X_train,nBdyP);
        // loss[2] -> setTrainData(&bdy_X_train);

      }
     
    // if(i%100 == 0)
      cout << "Iteration " << i << "  Error: " << error  << endl;
    
    i++;
  }while( i < nMaxIter);

  
  cout << "-------Finish training---------" << endl;
  

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  cout << "Train elapased time " <<  elapsed_secs << "sec." << endl;
}



int main() {



  int nMaxIter  = 1000;

  float tol     = 1e-3;
  int nEvalP    = 50;


  float a     = 0;
  float b     = 1;
  float c     = 0;
  float d     = 1;
  float delta_x = 0.01;
  float delta_y = 0.01;

  int n;
  int m;
  
  //plotter 
  GNUplot plotter;  
  string pnamefile = "./data/vfield2D_prediction_data.txt";
  string lnamefile = "./data/vfield2D_learning_data.txt";
  string enamefile = "./data/vfield2D_exact_data.txt";
  string s;
  
  //Problem Domain
  Rectangle <float> rectangle(a,b,c,d);  
  WriteFile<float> writter;

  rectangle.sampleFromInner(inner_X_train_1,nIntP_1);  
  rectangle.sampleFromInner(inner_X_train_2,nIntP_2);
  rectangle.sampleFromBottomBoundary(bdy_X_train,nBdyP);  

  // cout << bdy_X_train.transpose() << endl;
  

  
  get_Y(inner_X_train_1,inner_Y_train_1,ue<float>);
  // get_Y_train(bdy_X_train,bdy_Y_train,g_function<float>);

  vector<int> architecture = {2,2,3,2};
  NeuralNetwork<float> net(architecture,0.01, NeuralNetwork<float>::Activation::SIGMOID);
  
  float penalty_1 = 100;
  float penalty_2 = 100;

  FTerm<float> * lsterm  = new LSTerm<float>(&inner_X_train_1,&inner_Y_train_1,&net,ue);  
  // FTerm<float> * divterm = new DivTerm<float>(&inner_X_train_2,&net,delta_x,delta_y,rectangle.area(),penalty_1);  
  // FTerm<float> * bdyterm = new BdyTerm<float>(&bdy_X_train,&net,rectangle.perimeter()/4.0,penalty_2);  
  
  loss[0] = lsterm;
  // loss[1] = divterm;
  // loss[2] = bdyterm;
  
  train<float>(rectangle,nMaxIter,tol);

  // net.save("./data/params.txt");  
  
  rectangle.uniformSampleFromInner(X_test,nEvalP);  
  net.eval(X_test,Y_test);

  RowVectorXf x_test;
  RowVectorXf y_test;

  // cout << X_test.rows() <<" " << X_test.cols() << endl;
  x_test = X_test.col(561);
  cout << x_test << endl;
  net.eval(x_test,y_test); 
  cout << y_test << endl;
  
  // cout << X_test << endl;
  // cout << Y_test << endl;


  // get_Y(X_test,Y_exact,ue<float>);

  
  // // cout << Y_test << endl;
  // // cout << Y_exact << endl;

  
  // // writter.write(inner_X_train,inner_Y_train,lnamefile);
  // writter.write(X_test,Y_test,pnamefile);
  // writter.write(X_test,Y_exact,enamefile);
  
  // cout << "RMSE: " << rmse<float>(Y_test,Y_exact) << endl;
  
  // // cout << endl << "Neurons:" << endl;
  // // for(int i = 0; i < net.mNeurons.size(); i++)
  // //   cout << *net.mNeurons[i] << endl;
  // // cout << endl << "Weights:" << endl;
  // // for (int i = 0; i < net.mWeights.size(); i++)
  // //   cout << *net.mWeights[i] << endl;
  

  //plot 'vfield2D_prediction_data.txt' u 1:2:(0.1*$3):(0.1*$4) w vector
  
  // s = "set title "+string("\"")+"Clasification Zones"+string("\"")+";"+ "splot "+ string("\'")+ pnamefile+string("\'")+" using 1:2:3 with points,"+ string("\'")+ enamefile+string("\'")+" using 1:2:3 with points"+";";

  // // s = "set title "+string("\"")+"Clasification Zones"+string("\"")+";"+ "set palette model RGB defined "+ string("(")+"0"+string(" ")+ string("\"")+"red"+string("\"")+string(",")+"1"+string(" ")+string("\"")+"blue"+string("\"")+string(")") + ";" + "splot "+ string("\'")+ pnamefile+string("\'")+" using 1:2:3 with points pt 2 palette  notitle,"+ string("\'")+ lnamefile+string("\'")+" using 1:2:3 with points pt 7  ps 1 palette title "+string("\"")+"train data"+string("\"");
  
  // plotter(s.c_str());
  // int pause = cin.get();

  return 0;
}
