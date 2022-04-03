#include <stdint.h>

#ifndef MULTIK_JNI_SRC_MAIN_HEADERS_MK_MATH_WRAPPER_H_
#define MULTIK_JNI_SRC_MAIN_HEADERS_MK_MATH_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

int argmax(const void *arr, int offset, int size, int dim, const int *shape, const int *strides, int type);

int argmin(const void *arr, int offset, int size, int dim, const int *shape, const int *strides, int type);

void array_exp_float(float *arr, int size);

void array_exp_double(double *arr, int size);

void array_exp_complex_float(float *arr, int size);

void array_exp_complex_double(double *arr, int size);

void array_log_float(float *arr, int size);

void array_log_double(double *arr, int size);

void array_log_complex_float(float *arr, int size);

void array_log_complex_double(double *arr, int size);

void array_sin_float(float *arr, int size);

void array_sin_double(double *arr, int size);

void array_sin_complex_float(float *arr, int size);

void array_sin_complex_double(double *arr, int size);

void array_cos_float(float *arr, int size);

void array_cos_double(double *arr, int size);

void array_cos_complex_float(float *arr, int size);

void array_cos_complex_double(double *arr, int size);

int8_t array_max_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int16_t array_max_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int32_t array_max_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int64_t array_max_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
float array_max_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides);
double array_max_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides);

int8_t array_min_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int16_t array_min_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int32_t array_min_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int64_t array_min_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
float array_min_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides);
double array_min_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides);

int8_t array_sum_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int16_t array_sum_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int32_t array_sum_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
int64_t array_sum_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides);
float array_sum_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides);
double array_sum_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides);

void array_cumsum(const void *arr, void *out, int offset, int size,
				  int dim, const int *shape, const int *strides, int type);

#ifdef __cplusplus
}
#endif
#endif //MULTIK_JNI_SRC_MAIN_HEADERS_MK_MATH_WRAPPER_H_
