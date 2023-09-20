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
  // typedef Eigen::RowVectorXd RowVector;
  // typedef Eigen::VectorXd ColVector;
  // typedef float Scalar;  
  // typedef  Matrix<T,Dynamic,Dynamic> MatrixXXT;
  // typedef Matrix<T,1,Dynamic> RowVectorXT;

  template <typename T>
  class WriteFile{
    
  private:
    string _filename;
    
  public:
    WriteFile(){}
    WriteFile(string);
    // void eigentoData(MatrixXf& src, char* pathAndName);      
    // void genData();
    void write(Matrix<T,Dynamic,Dynamic>&, Matrix<T,Dynamic,Dynamic>&);
    void write(Matrix<T,Dynamic,Dynamic>&, Matrix<T,Dynamic,Dynamic>&,string);
    void writeCSV(Matrix<T, Dynamic, Dynamic>& X_test, Matrix<T, Dynamic, Dynamic>& Y_test, string filename);
  };

#include "../src/utils/WriteFile.cpp"
  
}
