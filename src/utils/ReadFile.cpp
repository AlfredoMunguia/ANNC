//#include <utils/ReadFile.hpp>
#include <string>

//namespace utils{

  //using namespace std;

  
  template <typename T>
  ReadFile<T>::ReadFile(std::string filename) {
      _filename = filename;
  };

  template <typename T>
  void ReadFile<T>::readCSV(vector<RowVector*>& data, Matrix<T, Dynamic, Dynamic>& X_train, Matrix<T, Dynamic, Dynamic>& Y_train, string filename) {

      _filename = filename;
      data.clear();
      std::ifstream file(_filename);
      std::string line, word;

      // determine number of columns in file
      getline(file, line, '\n');
      std::stringstream ss(line);
      std::vector<Scalar> parsed_vec;


      while (getline(ss, word, ',')) {
          // cout << word << endl;
          parsed_vec.push_back(Scalar(std::stof(&word[0])));
      }

      unsigned int cols = parsed_vec.size();
      // cout << cols << endl;
      data.push_back(new RowVector(cols));

      for (unsigned int i = 0; i < cols; i++) {
          // data.back()->coeffRef(1, i) = parsed_vec[i];
          data.back()->coeffRef(0, i) = parsed_vec[i];
      }

      if (file.is_open()) {     // read the file
          unsigned int j = 0;
          while (getline(file, line, '\n')) {
              std::stringstream ss(line);
              // cout << line << endl;
              data.push_back(new RowVector(1, cols));
              unsigned int i = 0;
              unsigned int z = 2;

              while (getline(ss, word, ',')) {
                  data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                  // cout <<"comparacion:   Scalar   " << Scalar(std::stof(&word[0])) << "    data  " << data.back()->coeffRef(i) << endl;
                  if (i < 2) {
                      X_train(i, j) = data.back()->coeffRef(i);
                  }
                  else {
                      Y_train(i - 2, j) = data.back()->coeffRef(i);
                  }
                  i++;
              }
              j++;

          }
      }
  }

  template <typename T>
  string ReadFile<T>::readFileIntoString(const string& path)
  {
      auto ss = ostringstream{};
      ifstream input_file(path);
      if (!input_file.is_open()) {
          cerr << "Could not open the file - '"
              << path << "'" << endl;
          exit(EXIT_FAILURE);
      }
      ss << input_file.rdbuf();
      //      cout << ss.str() << endl;  // imprime string de archivo 
      return ss.str();
  }
  //}
