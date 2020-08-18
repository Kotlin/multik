#ifndef CPP_CPP_LINALG_H_
#define CPP_CPP_LINALG_H_

#include "cblas.h"


void matrix_power() {

}

void svd() {

}

void norm() {

}

void det() {

}

void matrix_rank() {

}

void matrix_solve() {

}

void inv() {

}

void eigen_values(double *A, int n) {
  const char Nchar = 'N';
  double *eigReal = new double[n];
  double *eigImag = new double[n];
  const int lwork = 5 * n;
  double *work = new double[lwork];
  int info;
  double *lv, *rv;
  const int one = 1;
// перезаписывает матрицу
//  dgeev(&Nchar, &Nchar, &n, A, &n, eigReal, eigImag, nullptr, &one, nullptr, &one, work, &lwork, &info);

  delete [] eigReal;
  delete [] eigImag;
  delete [] work;
}

void matrix_dot(double *A, int n, int m, int k, double *B, double *C) {
  char Nchar = 'N';
  double alpha = 1.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n,m, k, alpha, A, n, B, k, alpha, C, n);
}

#endif //CPP_CPP_LINALG_H_
