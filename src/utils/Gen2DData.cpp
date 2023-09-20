
#include <string>

//namespace utils{
template <typename T>
Gen2DData<T>::Gen2DData(std::string filename){
  _filename = filename;
}



template <typename T>
void Gen2DData<T>::gen2Ddata(Matrix<T, Dynamic, Dynamic>& X_train, Matrix<T, Dynamic, Dynamic>& Y_train,float epsilon,float limiteRang1, float limiteRang2) {

    int n, m, p, q, r, s;
    float o,u;
    n = X_train.rows();
    m = X_train.cols();// +1;
    o = 0.00;
    u = 0.00;

    //epsilon = 0.1;
    limiteRang1 = limiteRang1 - epsilon; //a
    limiteRang2 = limiteRang2 + epsilon;  //b
    p = Y_train.rows();
    q = Y_train.cols();// +1;
    

    r = q / p;
    s = 0;

    srand(time(0));
    for (int i = 0; i < n; i++) {  // genera puntos de entrenamiento aleatorios 

        s = 0;
        for (int j = 0; j < m; j++) {

            if (j - s == r) {
               // cout << "Ciclo if: " << s << endl << endl;
                o = limiteRang2 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (limiteRang1 - limiteRang2)));
                for (int t = 0; t < n; t++) {
                    for (int v = 0; v < m; v++) {
                        if ((o + epsilon) >= X_train(t, v) && (o - epsilon) <= X_train(t, v) || (o + epsilon) >= X_train(t, v) && (o - epsilon) <= X_train(t, v)) {
                           // cout << "Interseccion encomntrada o: " << o+epsilon << " contra X_train(t, v): " << X_train(t, v) <<" y " << o-epsilon<< " " << (o+epsilon )-(o-epsilon)<< endl;
                            o = limiteRang2 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (limiteRang1 - limiteRang2)));
                            v = v - 1;
                        }
                    }
                }
                X_train(i, j) = o;
                s = r + s;
              // cout << "xxxxxxxxxxxxxxx: " << X_train(i, j) << endl << endl;
            }
            else {
               // cout << "Ciclo else: " << s << endl << endl;
                X_train(i, j) = o + epsilon + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.00 - epsilon)));
                //for (int t = 0; t < n; t++) {
                //    for (int v = 0; v < m; v++) {
                //        if ((u - epsilon) <= X_train(t, v) && (u + epsilon) >= X_train(t, v)) {  //
                //            cout << "Interseccion uuuuuuu: " << u << " contra X_train(t, v): " << X_train(t, v) << endl;
                //            u = o + epsilon + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (0.00 - epsilon)));
                //            v = v - 1;
                //        }
                //    }
                //}
               // X_train(i, j) = u;
                //X_train(i, j) = rand() / static_cast<float>(RAND_MAX);
               // X_train(i, j) = o+epsilon;
              // cout << "y: " << X_train(i, j) << endl;
            }


        }
    }

    s = 0;
    for (int i = 0; i < p; i++) { // genera etiquetas para esos puntos
        s = r + s;
        for (int j = 0; j < q; j++) {
            if (j < s && j >= s - r) {
                Y_train(i, j) = 1;
            }
            else
                Y_train(i, j) = 0;
        }
    }
}