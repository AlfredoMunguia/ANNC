//#include <utils/WriteFile.hpp>

fstream MyFile;

//namespace utils{
template <typename T>
WriteFile<T>::WriteFile(string filename){
  _filename = filename;
}


// void WriteFile::genData()
//   {
//     std::ofstream file1(_filename + "-in");
//     std::ofstream file2(_filename + "-out");

//     for (uint r = 0; r < 1000; r++) {
//       float x = rand() / float(RAND_MAX);
//       float y = rand() / float(RAND_MAX);
//       file1 << x << ", " << y << std::endl;
//       file2 << 2 * x + 10 + y << std::endl;
//     }
//     file1.close();
//     file2.close();
//   }


// void WriteFile::eigentoData(MatrixXf& src, char* pathAndName)
// {
//   ofstream fichier(pathAndName, ios::out | ios::trunc);  
//   if(fichier)  // si l'ouverture a réussi
//     {   
//       // instructions
//       fichier << "Here is the matrix src:\n" << src << "\n";
//       fichier.close();  // on referme le fichier
//     }
//   else  // sinon
//     {
//       cerr << "Erreur à l'ouverture !" << endl;
//         }
// }

template <typename T>
void WriteFile<T>::write(Matrix<T,Dynamic,Dynamic>& X_test, Matrix<T,Dynamic,Dynamic>& Y_test){
  
  std::ofstream file(_filename);
  Matrix<T,Dynamic,Dynamic> x_test = X_test.transpose();
  Matrix<T,Dynamic,Dynamic> y_test = Y_test.transpose();
  Matrix<T,Dynamic,Dynamic> mToSave(x_test.rows(),x_test.cols()+y_test.cols());

  mToSave << x_test,y_test;
  file << mToSave << endl;
  file.close();
}

template <typename T>
void WriteFile<T>::write(Matrix<T,Dynamic,Dynamic>& X_test, Matrix<T,Dynamic,Dynamic>& Y_test, string filename){
  
  std::ofstream file(filename);
  Matrix<T,Dynamic,Dynamic> x_test = X_test.transpose();
  Matrix<T,Dynamic,Dynamic> y_test = Y_test.transpose();
  Matrix<T,Dynamic,Dynamic> mToSave(x_test.rows(),x_test.cols()+y_test.cols());

  mToSave << x_test,y_test;
  file << mToSave << endl;
  file.close();
}

template <typename T>
void WriteFile<T>::writeCSV(Matrix<T, Dynamic, Dynamic>& X_train, Matrix<T, Dynamic, Dynamic>& Y_train, string filename) {


	std::ofstream file(filename);
	Matrix<T, Dynamic, Dynamic> x_train = X_train.transpose();
	Matrix<T, Dynamic, Dynamic> y_train = Y_train.transpose();
	//Matrix<T, Dynamic, Dynamic> mToSave(x_train.rows(), x_train.cols() + y_train.cols());

	for (int j = 0; j < x_train.cols(); j++) {
		file <<j<< ",";
	}
	for (int j = 0; j < y_train.cols(); j++) {
		if (j == y_train.cols() - 1) file <<j<< endl;
		else  file << j <<",";
	}

    for (int i = 0; i < x_train.rows(); i++) {
        for (int j = 0; j < x_train.cols(); j++) {
		     file << X_train(j, i) << ",";
        }
        for (int j = 0; j < y_train.cols(); j++) {
			if (j == y_train.cols()-1) file << Y_train(j, i) << endl;
			else  file << Y_train(j, i) << ",";
        }
     }

	file.close();


}




  
//}
