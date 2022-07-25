/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include <cmath>
#include <cstring>
#include "mk_math.h"

void index_increment(int *, const int *, int);
int array_ptr(int, const int *, const int *, int);

template<typename T>
int array_argmax(const T *arr, int offset, int size);
template<typename T>
int array_argmax(const T *arr, int offset, int size, int dim, const int *shape, const int *strides);
template<typename T>
int array_argmin(const T *arr, int offset, int size);
template<typename T>
int array_argmin(const T *arr, int offset, int size, int dim, const int *shape, const int *strides);
template<typename T>
T array_max(const T *arr, int offset, int size);
template<typename T>
T array_max(const T *arr, int offset, int size, int dim, const int *shape, const int *strides);
template<typename T>
T array_min(const T *arr, int offset, int size);
template<typename T>
T array_min(const T *arr, int offset, int size, int dim, const int *shape, const int *strides);
template<typename T>
T array_sum(const T *arr, int offset, int size);
template<typename T>
T array_sum(const T *arr, int offset, int size, int dim, const int *shape, const int *strides);
template<typename T>
void array_cumsum(const T *arr, T *out, int offset, int size);
template<typename T>
void array_cumsum(const T *arr, T *out, int offset, int size, int dim, const int *shape, const int *strides);

int argmax(const void *arr, int offset, int size, int dim, const int *shape, const int *strides, int type) {
  int ret = 0;

  if (strides == nullptr) {
	switch (type) {
	  case 1: {
		auto *array = (int8_t *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  case 2: {
		auto *array = (int16_t *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  case 3: {
		auto *array = (int32_t *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  case 4: {
		auto *array = (int64_t *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  case 5: {
		auto *array = (float *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  case 6: {
		auto *array = (double *)arr;
		ret = array_argmax(array, offset, size);
		break;
	  }
	  default:break;
	}
  } else {
	switch (type) {
	  case 1: {
		auto *array = (int8_t *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 2: {
		auto *array = (int16_t *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 3: {
		auto *array = (int32_t *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 4: {
		auto *array = (int64_t *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 5: {
		auto *array = (float *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 6: {
		auto *array = (double *)arr;
		ret = array_argmax(array, offset, size, dim, shape, strides);
		break;
	  }
	  default:break;
	}
  }

  return ret;
}

int argmin(const void *arr, int offset, int size, int dim, const int *shape, const int *strides, int type) {
  int ret = 0;

  if (strides == nullptr) {
	switch (type) {
	  case 1: {
		auto *array = (int8_t *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  case 2: {
		auto *array = (int16_t *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  case 3: {
		auto *array = (int32_t *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  case 4: {
		auto *array = (int64_t *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  case 5: {
		auto *array = (float *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  case 6: {
		auto *array = (double *)arr;
		ret = array_argmin(array, offset, size);
		break;
	  }
	  default:break;
	}
  } else {
	switch (type) {
	  case 1: {
		auto *array = (int8_t *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 2: {
		auto *array = (int16_t *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 3: {
		auto *array = (int32_t *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 4: {
		auto *array = (int64_t *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 5: {
		auto *array = (float *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  case 6: {
		auto *array = (double *)arr;
		ret = array_argmin(array, offset, size, dim, shape, strides);
		break;
	  }
	  default:break;
	}
  }

  return ret;
}

template<typename T>
int array_argmax(const T *arr, int offset, int size) {
  int ret = 0;
  T max = *(arr + offset);
  for (int i = 0; i < size; ++i) {
	if (max < arr[i]) {
	  max = arr[i];
	  ret = i;
	}
  }
  return ret;
}

template<typename T>
int array_argmax(const T *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int ret = 0;
  T max = *(arr + offset);
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int kI = 0; kI < size; ++kI) {
	int p = array_ptr(offset, index, strides, dim);
	if (max < arr[p]) {
	  max = arr[p];
	  ret = kI;
	}
	index_increment(index, shape, dim);
  }
  delete[] index;
  return ret;
}

template<typename T>
int array_argmin(const T *arr, int offset, int size) {
  int ret = 0;
  T min = *(arr + offset);
  for (int i = 0; i < size; ++i) {
	if (min > arr[i]) {
	  min = arr[i];
	  ret = i;
	}
  }
  return ret;
}

template<typename T>
int array_argmin(const T *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int ret = 0;
  T min = *(arr + offset);
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int kI = 0; kI < size; ++kI) {
	int p = array_ptr(offset, index, strides, dim);
	if (min > arr[p]) {
	  min = arr[p];
	  ret = kI;
	}
	index_increment(index, shape, dim);
  }
  delete[] index;
  return ret;
}

void array_exp_float(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::exp(arr[i]);
  }
}

void array_exp_double(double *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::exp(arr[i]);
  }
}

void array_exp_complex_float(float *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	float expReal = std::exp(arr[i]);
	arr[i] = expReal * std::cos(arr[i + 1]);
	arr[i + 1] = expReal * std::sin(arr[i + 1]);
  }
}

void array_exp_complex_double(double *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	double expReal = std::exp(arr[i]);
	arr[i] = expReal * std::cos(arr[i + 1]);
	arr[i + 1] = expReal * std::sin(arr[i + 1]);
  }
}

void array_log_float(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::log(arr[i]);
  }
}

void array_log_double(double *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::log(arr[i]);
  }
}

void array_log_complex_float(float *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	float abs = std::sqrt(arr[i] * arr[i] + arr[i + 1] + arr[i + 1]);
	float angle = std::atan2(arr[i + 1], arr[i]);
	arr[i] = abs;
	arr[i + 1] = angle;
  }
}

void array_log_complex_double(double *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	double abs = std::sqrt(arr[i] * arr[i] + arr[i + 1] + arr[i + 1]);
	double angle = std::atan2(arr[i + 1], arr[i]);
	arr[i] = abs;
	arr[i + 1] = angle;
  }
}

void array_sin_float(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::sin(arr[i]);
  }
}

void array_sin_double(double *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::sin(arr[i]);
  }
}

void array_sin_complex_float(float *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	float cosRe = std::cos(arr[i]);
	arr[i] = std::sin(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = cosRe * std::sinh(arr[i + 1]);
  }
}

void array_sin_complex_double(double *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	double cosRe = std::cos(arr[i]);
	arr[i] = std::sin(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = cosRe * std::sinh(arr[i + 1]);
  }
}

void array_cos_float(float *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::cos(arr[i]);
  }
}

void array_cos_double(double *arr, int size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::cos(arr[i]);
  }
}

void array_cos_complex_float(float *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	float sinRe = std::sin(arr[i]);
	arr[i] = std::cos(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = sinRe * std::sinh(arr[i + 1]);
  }
}

void array_cos_complex_double(double *arr, int size) {
  for (int i = 0; i < size; i += 2) {
	double sinRe = std::sin(arr[i]);
	arr[i] = std::cos(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = sinRe * std::sinh(arr[i + 1]);
  }
}

int8_t array_max_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int8_t ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

int16_t array_max_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int16_t ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

int32_t array_max_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int32_t ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

int64_t array_max_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int64_t ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

float array_max_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  float ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

double array_max_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  double ret = (strides != nullptr) ? array_max(arr, offset, size, dim, shape, strides) : array_max(arr, offset, size);
  return ret;
}

int8_t array_min_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int8_t ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

int16_t array_min_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int16_t ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

int32_t array_min_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int32_t ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

int64_t array_min_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int64_t ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

float array_min_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  float ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

double array_min_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  double ret = (strides != nullptr) ? array_min(arr, offset, size, dim, shape, strides) : array_min(arr, offset, size);
  return ret;
}

template<typename T>
T array_max(const T *arr, int offset, int size) {
  T max = *(arr + offset);
  const T *end = arr + size;
  for (const T *p = arr + offset; p < end; ++p) {
	if (max < *p) {
	  max = *p;
	}
  }
  return max;
}

template<typename T>
T array_max(const T *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  T max = *(arr + offset);
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int kI = 0; kI < size; ++kI) {
	int p = array_ptr(offset, index, strides, dim);
	if (max < arr[p]) {
	  max = arr[p];
	}
	index_increment(index, shape, dim);
  }
  delete[] index;
  return max;
}

template<typename T>
T array_min(const T *arr, int offset, int size) {
  T min = *(arr + offset);
  const T *end = arr + size;
  for (const T *p = arr + offset; p < end; ++p) {
	if (min > *p) {
	  min = *p;
	}
  }
  return min;
}

template<typename T>
T array_min(const T *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  T min = *(arr + offset);
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int kI = 0; kI < size; ++kI) {
	int p = array_ptr(offset, index, strides, dim);
	if (min > arr[p]) {
	  min = arr[p];
	}
	index_increment(index, shape, dim);
  }
  delete[] index;
  return min;
}

int8_t array_sum_int8(const int8_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int8_t ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

int16_t array_sum_int16(const int16_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int16_t ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

int32_t array_sum_int32(const int32_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int32_t ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

int64_t array_sum_int64(const int64_t *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  int64_t ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

float array_sum_float(const float *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  float ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

double array_sum_double(const double *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  double ret = (strides != nullptr) ? array_sum(arr, offset, size, dim, shape, strides) : array_sum(arr, offset, size);
  return ret;
}

template<typename T>
T array_sum(const T *arr, int offset, int size) {
  T accum = 0;
  T compens = 0;
  const T *end = arr + size;
  for (const T *p = arr + offset; p < end; ++p) {
	T y = *p - compens;
	T t = accum + y;
	compens = (t - accum) - y;
	accum = t;
  }
  return accum;
}

template<typename T>
T array_sum(const T *arr, int offset, int size, int dim, const int *shape, const int *strides) {
  double accum = 0;
  double compens = 0;
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int i = 0; i < size; ++i) {
	int p = array_ptr(offset, index, strides, dim);
	double y = arr[p] - compens;
	double t = accum + y;
	compens = (t - accum) - y;
	accum = t;
	index_increment(index, shape, dim);
  }
  delete[] index;
  return static_cast<T>(accum);
}

void array_cumsum(const void *arr, void *out, int offset, int size,
				  int dim, const int *shape, const int *strides, int type) {
  switch (type) {
	case 1: {
	  (strides != nullptr) ? array_cumsum((int8_t *)arr, (int8_t *)out, offset, size, dim, shape, strides)
						   : array_cumsum((int8_t *)arr, (int8_t *)out, offset, size);
	  break;
	}
	case 2: {
	  (strides != nullptr) ? array_cumsum((int16_t *)arr, (int16_t *)out, offset, size, dim, shape, strides)
						   : array_cumsum((int16_t *)arr, (int16_t *)out, offset, size);
	  break;
	}
	case 3: {
	  (strides != nullptr) ? array_cumsum((int32_t *)arr, (int32_t *)out, offset, size, dim, shape, strides)
						   : array_cumsum((int32_t *)arr, (int32_t *)out, offset, size);
	  break;
	}
	case 4: {
	  (strides != nullptr) ? array_cumsum((int64_t *)arr, (int64_t *)out, offset, size, dim, shape, strides)
						   : array_cumsum((int64_t *)arr, (int64_t *)out, offset, size);
	  break;
	}
	case 5: {
	  (strides != nullptr) ? array_cumsum((float *)arr, (float *)out, offset, size, dim, shape, strides)
						   : array_cumsum((float *)arr, (float *)out, offset, size);
	  break;
	}
	case 6: {
	  (strides != nullptr) ? array_cumsum((double *)arr, (double *)out, offset, size, dim, shape, strides)
						   : array_cumsum((double *)arr, (double *)out, offset, size);
	  break;
	}
	default:break;
  }
}

template<typename T>
void array_cumsum(const T *arr, T *out, int offset, int size) {
  double accum = 0;
  double compens = 0;
  for (int i = offset; i < size + offset; ++i) {
	double y = arr[i] - compens;
	double t = accum + y;
	compens = (t - accum) - y;
	accum = t;
	*(out + i) = static_cast<T>(accum);
  }
}

template<typename T>
void array_cumsum(const T *arr, T *out, int offset, int size, int dim, const int *shape, const int *strides) {
  double accum = 0;
  double compens = 0;
  int *index = new int[dim];
  memset(index, 0, dim * 4);
  for (int i = 0; i < size; ++i) {
	int p = array_ptr(offset, index, strides, dim);
	double y = arr[p] - compens;
	double t = accum + y;
	compens = (t - accum) - y;
	accum = t;
	*(out + i) = static_cast<T>(accum);
	index_increment(index, shape, dim);
  }
  delete[] index;
}

int array_ptr(int offset, const int *index, const int *strides, int size) {
  int ptr = offset;
  for (const int *ind = index, *str = strides; ind < index + size; ++ind, ++str) {
	ptr += (*str) * (*ind);
  }
  return ptr;
}

void index_increment(int *index, const int *shape, int size) {
  for (int *ind = index + size - 1, *sh = const_cast<int *>(shape + size - 1); ind >= index; --ind, --sh) {
	int tmp = (*ind) + 1;
	if (tmp >= *sh && ind != index) {
	  *ind = 0;
	} else {
	  *ind = tmp;
	  break;
	}
  }
}
