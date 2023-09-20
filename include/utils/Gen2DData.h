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
  class Gen2DData{
    
  private:
    string _filename;
    
  public:
      Gen2DData(){}
      Gen2DData(string filename);
    // void eigentoData(MatrixXf& src, char* pathAndName);      
    // void genData();

    void gen2Ddata(Matrix<T, Dynamic, Dynamic>& X_train, Matrix<T, Dynamic, Dynamic>& Y_train, float epsilon, float limiteRang1, float limiteRang2);
  };

#include "../src/utils/Gen2DData.cpp"  
}
