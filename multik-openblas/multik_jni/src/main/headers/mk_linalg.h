#include "stdbool.h"

#ifndef CPP_MULTIK_JNI_SRC_MAIN_HEADERS_MK_LINALG_H_
#define CPP_MULTIK_JNI_SRC_MAIN_HEADERS_MK_LINALG_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float real, imag; } mk_complex_float;
typedef struct { double real, imag; } mk_complex_double;

float vector_dot_float(int n, int offsetX, float *X, int incx, int offsetY, float *Y, int incy);

double vector_dot_double(int n, int offsetX, double *X, int incx, int offsetY, double *Y, int incy);

mk_complex_float vector_dot_complex_float(int n, int offset_x, float *X, int incx, int offset_y, float *Y, int incy);

mk_complex_double vector_dot_complex_double(int n, int offsetX, double *X, int incx, int offsetY, double *Y, int incy);

float norm_matrix_float(char norm, int m, int n, float *A, int lda);

double norm_matrix_double(char norm, int m, int n, double *A, int lda);

int qr_matrix_float(int m, int n, float *AQ, int lda, float *R);

int qr_matrix_double(int m, int n, double *AQ, int lda, double *R);

int qr_matrix_complex_float(int m, int n, float *AQ, int lda, float *R);

int qr_matrix_complex_double(int m, int n, double *AQ, int lda, double *R);

int plu_matrix_float(int m, int n, float *A, int lda, int *IPIV);

int plu_matrix_double(int m, int n, double *A, int lda, int *IPIV);

int plu_matrix_complex_float(int m, int n, float *A, int lda, int *IPIV);

int plu_matrix_complex_double(int m, int n, double *A, int lda, int *IPIV);

//int svd_matrix_float(int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt);

//int svd_matrix_double(int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt);

//int svd_matrix_complex_float(int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt);

//int svd_matrix_complex_double(int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt);

int solve_linear_system_float(int n, int nrhs, float *A, int lda, float *b, int ldb);

int solve_linear_system_double(int n, int nrhs, double *A, int lda, double *b, int ldb);

int solve_linear_system_complex_float(int n, int nrhs, float *A, int lda, float *B, int ldb);

int solve_linear_system_complex_double(int n, int nrhs, double *A, int lda, double *B, int ldb);

int inverse_matrix_float(int n, float *A, int lda);

int inverse_matrix_double(int n, double *A, int lda);

int inverse_matrix_complex_float(int n, float *A, int lda);

int inverse_matrix_complex_double(int n, double *A, int lda);

// TODO(_concat_fortran_string)
//int eigen_float(int n, float *A, float *W, char computeV, float *VR);

//int eigen_double(int n, double *A, double *W, char computeV, double *VR);

void matrix_dot_float(bool trans_a, int offsetA, float *A, int lda, int m, int n, int k,
					  bool trans_b, int offsetB, float *B, int ldb, float *C);

void matrix_dot_double(bool trans_a, int offsetA, double *A, int lda, int m, int n, int k,
					   bool trans_b, int offsetB, double *B, int ldb, double *C);

void matrix_dot_complex_float(bool trans_a, int offsetA, float *A, int lda, int m, int n, int k,
							  bool trans_b, int offsetB, float *B, int ldb, float *C);

void matrix_dot_complex_double(bool trans_a, int offsetA, double *A, int lda, int m, int n, int k,
							   bool trans_b, int offsetB, double *B, int ldb, double *C);

void matrix_dot_vector_float(bool trans_a, int offsetA, float *A, int lda, int m, int n, int offsetX, float *X, int incx, float *Y);

void matrix_dot_vector_double(bool trans_a, int offsetA, double *A, int lda, int m, int n, int offsetX, double *X, int incx, double *Y);

void matrix_dot_complex_vector_float(bool trans_a, int offsetA, float *A, int lda, int m, int n, int offsetX, float *X, int incx, float *Y);

void matrix_dot_complex_vector_double(bool trans_a, int offsetA, double *A, int lda, int m, int n, int offsetX, double *X, int incx, double *Y);

#ifdef __cplusplus
}
#endif
#endif //CPP_MULTIK_JNI_SRC_MAIN_HEADERS_MK_LINALG_H_
