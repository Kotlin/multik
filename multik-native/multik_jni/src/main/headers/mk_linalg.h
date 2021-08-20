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

float vector_dot(int n, float *X, int incx, float *Y, int incy) {
  return cblas_sdot(n, X, incx, Y, incy);
}

double vector_dot(int n, double *X, int incx, double *Y, int incy) {
  return cblas_ddot(n, X, incx, Y, incy);
}

openblas_complex_float vector_dot_complex(int n, float *X, int incx, float *Y, int incy) {
  return cblas_cdotc(n, X, incx, Y, incy);
}

openblas_complex_double vector_dot_complex(int n, double *X, int incx, double *Y, int incy) {
  return cblas_zdotc(n, X, incx, Y, incy);
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

void matrix_dot(bool trans_a, int offsetA, float *A, int lda, int m, int n, int k,
				bool trans_b, int offsetB, float *B, int ldb, float *C) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, A + offsetA, lda, B + offsetB, ldb, beta, C, n);
}

void matrix_dot(bool trans_a, int offsetA, double *A, int lda, int m, int n, int k,
				bool trans_b, int offsetB, double *B, int ldb, double *C) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, alpha, A + offsetA, lda, B + offsetB, ldb, beta, C, n);
}

void matrix_dot_complex(bool trans_a, int offsetA, float *A, int lda, int m, int n, int k,
						bool trans_b, int offsetB, float *B, int ldb, float *C) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_cgemm(CblasRowMajor, transA, transB, m, n, k, &alpha, A + offsetA, lda, B + offsetB, ldb, &beta, C, n);
}

void matrix_dot_complex(bool trans_a, int offsetA, double *A, int lda, int m, int n, int k,
						bool trans_b, int offsetB, double *B, int ldb, double *C) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;
  CBLAS_TRANSPOSE transB;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;
  (trans_b) ? transB = CblasTrans : transB = CblasNoTrans;

  cblas_zgemm(CblasRowMajor, transA, transB, m, n, k, &alpha, A + offsetA, lda, B + offsetB, ldb, &beta, C, n);
}

void matrix_dot(bool trans_a, int offsetA, float *A, int lda, int m, int n, float *X, int incx, float *Y) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_sgemv(CblasRowMajor, transA, m, n, alpha, A + offsetA, lda, X, incx, beta, Y, 1);
}

void matrix_dot(bool trans_a, int offsetA, double *A, int lda, int m, int n, double *X, int incx, double *Y) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_dgemv(CblasRowMajor, transA, m, n, alpha, A + offsetA, lda, X, incx, beta, Y, 1);
}

void matrix_dot_complex(bool trans_a, int offsetA, float *A, int lda, int m, int n, float *X, int incx, float *Y) {
  float alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_cgemv(CblasRowMajor, transA, m, n, &alpha, A + offsetA, lda, X, incx, &beta, Y, 1);
}

void matrix_dot_complex(bool trans_a, int offsetA, double *A, int lda, int m, int n, double *X, int incx, double *Y) {
  double alpha = 1.0, beta = 0.0;
  CBLAS_TRANSPOSE transA;

  (trans_a) ? transA = CblasTrans : transA = CblasNoTrans;

  cblas_zgemv(CblasRowMajor, transA, m, n, &alpha, A + offsetA, lda, X, incx, &beta, Y, 1);
}

#endif //CPP_CPP_LINALG_H_
