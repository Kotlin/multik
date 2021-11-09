/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#ifndef CPP_CPP_LINALG_H_
#define CPP_CPP_LINALG_H_

#include "cblas.h"
#include "lapacke.h"
#include "algorithm"
#include <cstring>

float vector_dot(int n, float *X, int incx, float *Y, int incy) {
  return cblas_sdot(n, X, incx, Y, incy);
}

double vector_dot(int n, double *X, int incx, double *Y, int incy) {
  return cblas_ddot(n, X, incx, Y, incy);
}

openblas_complex_float vector_dot_complex(int n, float *X, int incx, float *Y, int incy) {
  return cblas_cdotu(n, X, incx, Y, incy);
}

openblas_complex_double vector_dot_complex(int n, double *X, int incx, double *Y, int incy) {
  return cblas_zdotu(n, X, incx, Y, incy);
}

int qr_matrix(int m, int n, float *AQ, int lda, float *R) {
  int mn = std::min(m, n);
  float tau[mn];

  int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, AQ, lda, tau);
  if (info != 0)
	return info;

  for (size_t row = 0; row < mn; ++row) {
	size_t index = (n + 1) * row;
	std::memcpy(R + index, AQ + index, (n - row) * sizeof(float));
  }

  return LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, mn, mn, AQ, lda, tau);
}

int qr_matrix(int m, int n, double *AQ, int lda, double *R) {
  int mn = std::min(m, n);
  double tau[mn];

  int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, AQ, lda, tau);
  if (info != 0)
	return info;

  for (size_t row = 0; row < mn; ++row) {
	size_t index = (n + 1) * row;
	std::memcpy(R + index, AQ + index, (n - row) * sizeof(double));
  }

  return LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, mn, mn, AQ, lda, tau);
}

int qr_matrix_complex(int m, int n, float *AQ, int lda, float *R) {
  int mn = std::min(m, n);
  lapack_complex_float tau[mn];
  lapack_complex_float *aq = (lapack_complex_float *)AQ;
  lapack_complex_float *r = (lapack_complex_float *)R;

  int info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, aq, lda, tau);
  if (info != 0)
	return info;

  for (size_t row = 0; row < mn; ++row) {
	size_t index = (n + 1) * row;
	std::memcpy(r + index, aq + index, (n - row) * sizeof(lapack_complex_float));
  }

  return LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, mn, mn, aq, lda, tau);
}

int qr_matrix_complex(int m, int n, double *AQ, int lda, double *R) {
  int mn = std::min(m, n);
  lapack_complex_double tau[mn];
  lapack_complex_double *aq = (lapack_complex_double *)AQ;
  lapack_complex_double *r = (lapack_complex_double *)R;

  int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, aq, lda, tau);
  if (info != 0)
	return info;

  for (size_t row = 0; row < mn; ++row) {
	size_t index = (n + 1) * row;
	std::memcpy(r + index, aq + index, (n - row) * sizeof(lapack_complex_double));
  }

  return LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, mn, mn, aq, lda, tau);
}

int plu_matrix(int m, int n, float *A, int lda, int *IPIV) {
  int num_threads = openblas_get_num_threads(); // TODO (fast fix for single threaded mode. Remove, wrap set in kotlin)
  openblas_set_num_threads(1);
  int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, IPIV);
  openblas_set_num_threads(num_threads);
  return info;
}

int plu_matrix(int m, int n, double *A, int lda, int *IPIV) {
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, IPIV);
  openblas_set_num_threads(num_threads);
  return info;
}

int plu_matrix_complex(int m, int n, float *A, int lda, int *IPIV) {
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  lapack_complex_float *a = (lapack_complex_float *)A;
  int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, IPIV);
  openblas_set_num_threads(num_threads);
  return info;
}

int plu_matrix_complex(int m, int n, double *A, int lda, int *IPIV) {
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  lapack_complex_double *a = (lapack_complex_double *)A;
  int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, IPIV);
  openblas_set_num_threads(num_threads);
  return info;
}

int solve_linear_system(int n, int nrhs, float *A, int lda, float *b, int ldb) {
  int ipiv[n];
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv, b, ldb);
  openblas_set_num_threads(num_threads);
  return info;
}

int solve_linear_system(int n, int nrhs, double *A, int lda, double *b, int ldb) {
  int ipiv[n];
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv, b, ldb);
  openblas_set_num_threads(num_threads);
  return info;
}

int solve_linear_system_complex(int n, int nrhs, float *A, int lda, float *B, int ldb) {
  int ipiv[n];
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  lapack_complex_float *a = (lapack_complex_float *)A;
  lapack_complex_float *b = (lapack_complex_float *)B;
  int info = LAPACKE_cgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
  openblas_set_num_threads(num_threads);
  return info;
}

int solve_linear_system_complex(int n, int nrhs, double *A, int lda, double *B, int ldb) {
  int ipiv[n];
  int num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  lapack_complex_double *a = (lapack_complex_double *)A;
  lapack_complex_double *b = (lapack_complex_double *)B;
  int info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
  openblas_set_num_threads(num_threads);
  return info;
}

int inverse_matrix(int n, float *A, int lda) {
  int ipiv[n];
  plu_matrix(n, n, A, lda, ipiv);
  return LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, A, lda, ipiv);
}

int inverse_matrix(int n, double *A, int lda) {
  int ipiv[n];
  plu_matrix(n, n, A, lda, ipiv);
  return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, lda, ipiv);
}

int inverse_matrix_complex(int n, float *A, int lda) {
  int ipiv[n];
  plu_matrix_complex(n, n, A, lda, ipiv);
  lapack_complex_float *a = (lapack_complex_float *)A;
  return LAPACKE_cgetri(LAPACK_ROW_MAJOR, n, a, lda, ipiv);
}

int inverse_matrix_complex(int n, double *A, int lda) {
  int ipiv[n];
  plu_matrix_complex(n, n, A, lda, ipiv);
  lapack_complex_double *a = (lapack_complex_double *)A;
  return LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, a, lda, ipiv);
}

int eigen(int n, float *A, float *W, char computeV, float *VR) {
  lapack_complex_float *a = (lapack_complex_float *)A;
  lapack_complex_float *w = (lapack_complex_float *)W;
  lapack_complex_float *vr = (lapack_complex_float *)VR;

  return LAPACKE_cgeev(LAPACK_ROW_MAJOR, 'N', computeV, n, a, n, w, nullptr, n, vr, n);
}

int eigen(int n, double *A, double *W, char computeV, double *VR) {
  lapack_complex_double *a = (lapack_complex_double *)A;
  lapack_complex_double *w = (lapack_complex_double *)W;
  lapack_complex_double *vr = (lapack_complex_double *)VR;

  return LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', computeV, n, a, n, w, nullptr, n, vr, n);
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
