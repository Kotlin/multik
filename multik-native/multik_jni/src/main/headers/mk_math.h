/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include <iostream>
#include <cmath>
#include <string.h>

#ifndef CPP_HEADERS_MK_MATH_H_
#define CPP_HEADERS_MK_MATH_H_

void index_increment(int *, const int *, int);
int array_ptr(int, const int *, const int *, int);

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

void array_exp(float *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::exp(arr[i]);
  }
}

void array_exp(double *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::exp(arr[i]);
  }
}

void array_exp_complex(float *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	float expReal = std::exp(arr[i]);
	arr[i] = expReal * std::cos(arr[i + 1]);
	arr[i + 1] = expReal * std::sin(arr[i + 1]);
  }
}

void array_exp_complex(double *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	double expReal = std::exp(arr[i]);
	arr[i] = expReal * std::cos(arr[i + 1]);
	arr[i + 1] = expReal * std::sin(arr[i + 1]);
  }
}

void array_log(float *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::log(arr[i]);
  }
}

void array_log(double *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::log(arr[i]);
  }
}

void array_log_complex(float *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	float abs = std::sqrt(arr[i] * arr[i] + arr[i + 1] + arr[i + 1]);
	float angle = std::atan2(arr[i + 1], arr[i]);
	arr[i] = abs;
	arr[i + 1] = angle;
  }
}

void array_log_complex(double *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	double abs = std::sqrt(arr[i] * arr[i] + arr[i + 1] + arr[i + 1]);
	double angle = std::atan2(arr[i + 1], arr[i]);
	arr[i] = abs;
	arr[i + 1] = angle;
  }
}

void array_sin(float *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::sin(arr[i]);
  }
}

void array_sin(double *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::sin(arr[i]);
  }
}

void array_sin_complex(float *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	float cosRe = std::cos(arr[i]);
	arr[i] = std::sin(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = cosRe * std::sinh(arr[i + 1]);
  }
}

void array_sin_complex(double *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	double cosRe = std::cos(arr[i]);
	arr[i] = std::sin(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = cosRe * std::sinh(arr[i + 1]);
  }
}

void array_cos(float *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::cos(arr[i]);
  }
}

void array_cos(double *arr, size_t size) {
  for (int i = 0; i < size; ++i) {
	arr[i] = std::cos(arr[i]);
  }
}

void array_cos_complex(float *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	float sinRe = std::sin(arr[i]);
	arr[i] = std::cos(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = sinRe * std::sinh(arr[i + 1]);
  }
}

void array_cos_complex(double *arr, size_t size) {
  for (int i = 0; i < size; i += 2) {
	double sinRe = std::sin(arr[i]);
	arr[i] = std::cos(arr[i]) * std::cosh(arr[i + 1]);
	arr[i + 1] = sinRe * std::sinh(arr[i + 1]);
  }
}

template<typename T>
T array_max(const T *arr, int offset, int size) {
  T max = *(arr + offset);
  const T *end = arr + size;
  for (const T *p = arr; p < end; ++p) {
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
  for (const T *p = arr; p < end; ++p) {
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

template<typename T>
T array_sum(const T *arr, int size) {
  T accum = 0;
  T compens = 0;
  const T *end = arr + size;
  for (const T *p = arr; p < end; ++p) {
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

template<typename T>
void array_cumsum(const T *arr, T *out, int size) {
  double accum = 0;
  double compens = 0;
  for (int i = 0; i < size; ++i) {
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

#endif //CPP_HEADERS_MK_MATH_H_
