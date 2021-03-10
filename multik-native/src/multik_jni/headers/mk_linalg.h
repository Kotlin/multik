/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#ifndef CPP_CPP_LINALG_H_
#define CPP_CPP_LINALG_H_

#include "cblas.h"
#include "lapacke.h"

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

  delete[] eigReal;
  delete[] eigImag;
  delete[] work;
}

float vector_dot_float(int n, float *A, int strA, float *B, int strB) {
  return cblas_sdot(n, A, strA, B, strB);
}

double vector_dot_double(int n, double *A, int strA, double *B, int strB) {
  return cblas_ddot(n, A, strA, B, strB);
}

int solve_linear_system_float(int n, int nrhs, float *A, int strA, float *b, int strB) {
  int ipiv[n];

  return LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, A, strA, ipiv, b, strB);
}

int solve_linear_system_double(int n, int nrhs, double *A, int strA, double *b, int strB) {
  int ipiv[n];

  return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, strA, ipiv, b, strB);
}

void matrix_dot_float(float *A, int m, int n, int k, float *B, float *C) {
  float alpha = 1.0;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, alpha, C, n);
}

void matrix_dot_double(double *A, int m, int n, int k, double *B, double *C) {
  double alpha = 1.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, alpha, C, n);
}

void matrix_dot_vector_float(float *A, int m, int n, float *B, float *C) {
  float alpha = 1.0;
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, B, alpha, alpha, C, alpha);
}

void matrix_dot_vector_double(double *A, int m, int n, double *B, double *C) {
  double alpha = 1.0;
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, B, alpha, alpha, C, alpha);
}

#endif //CPP_CPP_LINALG_H_
