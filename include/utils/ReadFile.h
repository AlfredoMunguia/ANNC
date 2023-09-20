
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <Eigen>
#include <stdlib.h>

namespace utils{

  using namespace std;
  using namespace Eigen;

 // typedef Eigen::MatrixXf Matrix;
  typedef Eigen::RowVectorXd RowVector;
  //typedef Eigen::VectorXd ColVector;
  typedef float Scalar;
  
  template <typename T>
  class ReadFile{    

  private:
    string _filename;
    vector<RowVector*> _data;
    
    
  public:
      ReadFile() {}
    ReadFile(string filename);
    string readFileIntoString(const string& path);
    //void readCSV(vector<RowVector*> &);    
    void readCSV(vector<RowVector*>&, Matrix<T, Dynamic, Dynamic>&, Matrix<T, Dynamic, Dynamic>&, string filename);

  };
#include "../src/utils/ReadFile.cpp"
}
