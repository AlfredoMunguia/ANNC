#include <fterm/FTerm.hpp>
#include <Eigen/Eigen>
#include <nn/NeuralNetwork.hpp>

namespace scheduler{
  
  using namespace fterm;
  using namespace std;
  
  
#ifndef SCHEDULER_H
#define SCHEDULER_H

  template <typename T>
  class Scheduler{
    int  _step_size;
    T    _gamma;
    T    _lr;
    bool _update;
    int  _factor;
    int  _n;
    
  protected:
    NeuralNetwork<T>* _nn;


  public:
    Scheduler(NeuralNetwork<T> * nn, int step_size = 30, T gamma = 0.1);
    void update(int);

    
  };
  
  
#include "../src/scheduler/Scheduler.cpp"
  
  
#endif
  
}
