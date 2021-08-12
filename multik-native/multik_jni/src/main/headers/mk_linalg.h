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

float vector_dot(int n, float *A, int strA, float *B, int strB) {
  return cblas_sdot(n, A, strA, B, strB);
}

double vector_dot(int n, double *A, int strA, double *B, int strB) {
  return cblas_ddot(n, A, strA, B, strB);
}

openblas_complex_float vector_dot_complex(int n, float *A, int strA, float *B, int strB) {
  return cblas_cdotc(n, A, strA, B, strB);
}

openblas_complex_double vector_dot_complex(int n, double *A, int strA, double *B, int strB) {
  return cblas_zdotc(n, A, strA, B, strB);
}

int solve_linear_system_float(int n, int nrhs, float *A, int strA, float *b, int strB) {
  int ipiv[n];

  return LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, A, strA, ipiv, b, strB);
}

int solve_linear_system_double(int n, int nrhs, double *A, int strA, double *b, int strB) {
  int ipiv[n];

  return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, strA, ipiv, b, strB);
}

int inverse_matrix_float(int n, float *A, int strA) {
  int ipiv[n];

  LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, A, strA, ipiv);

  return LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, A, strA, ipiv);
}

int inverse_matrix_double(int n, double *A, int strA) {
  int ipiv[n];

  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, strA, ipiv);

  return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, strA, ipiv);
}

void matrix_dot(bool trans_a, float *A, int m, int n, int k, bool trans_b, float *B, float *C) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, A, k, B, n, beta, C, n);
}

void matrix_dot(bool trans_a, double *A, int m, int n, int k, bool trans_b, double *B, double *C) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, alpha, A, k, B, n, beta, C, n);
}

void matrix_dot_complex(bool trans_a, float *A, int m, int n, int k, bool trans_b, float *B, void *C) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_cgemm(CblasRowMajor, transA, transB, m, n, k, &alpha, A, k, B, n, &beta, C, n);
}

void matrix_dot_complex(bool trans_a, double *A, int m, int n, int k, bool trans_b, double *B, double *C) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_zgemm(CblasRowMajor, transA, transB, m, n, k, &alpha, A, k, B, n, &beta, C, n);
}

void matrix_dot(bool trans_a, float *A, int m, int n, float *B, float *C) {
  float alpha = 1.0, beta = 0.0;
  //TODO: incx?
  blasint incy = 1;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_sgemv(CblasRowMajor, transA, m, n, alpha, A, n, B, 1, beta, C, incy);
}

void matrix_dot(bool trans_a, double *A, int m, int n, double *B, double *C) {
  double alpha = 1.0, beta = 0.0;
  //TODO: incx?
  blasint incy = 1;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_dgemv(CblasRowMajor, transA, m, n, alpha, A, n, B, 1, beta, C, incy);
}

void matrix_dot_complex(bool trans_a, float *A, int m, int n, float *B, float *C) {
  float alpha = 1.0, beta = 0.0;
  //TODO: incx?
  blasint incy = 1;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_cgemv(CblasRowMajor, transA, m, n, &alpha, A, n, B, 1, &beta, C, incy);
}

void matrix_dot_complex(bool trans_a, double *A, int m, int n, double *B, double *C) {
  double alpha = 1.0, beta = 0.0;
  //TODO: incx?
  blasint incy = 1;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_zgemv(CblasRowMajor, transA, m, n, &alpha, A, n, B, 1, &beta, C, incy);
}

#endif //CPP_CPP_LINALG_H_
