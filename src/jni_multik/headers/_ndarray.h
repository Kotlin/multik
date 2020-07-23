#pragma once

#include "cmath"
#include <cstdarg>
#include <vector>

class Ndarray;
class Buffer;
typedef enum Datatype {
  J_INT8 = 1,
  J_INT16 = 2,
  J_INT32 = 3,
  J_INT64 = 4,
  J_FLOAT32 = 5,
  J_FLOAT64 = 6,
} Datatype;

class Buffer {
 public:
  Buffer() : data_(nullptr), size_(0) {};
  Buffer(void *ptr, size_t size) : data_(ptr), size_(size) {};

  ~Buffer() {
    delete &data_;
  }

  void *data() const { return data_; }

  size_t size() const { return size_; }

 private:
  void *data_;
  size_t size_;
};

class Ndarray {
 public:
  Ndarray(Buffer *buffer, std::vector<int> &shape, int dtype)
      : data_(buffer), shape_(shape), dtype_(Datatype(dtype)) {
    size_ = 1;
    for (int d: shape_) {
      size_ *= d;
    }

    stride_ = std::vector(shape_);
    stride_[stride_.size() - 1] = 1;
    for (int i = stride_.size() - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
  }

  ~Ndarray() {
    delete data_;
  }

  const void *get_value(int idx, ...) const {
    int p = 0;
    va_list indices;
    va_start(indices, idx);
    for (int i = 0; i < idx; ++i) {
      p += va_arg(indices, int) * stride_[i];
    }
    va_end(indices);
    return (char *) data_->data() + p;
  }

  void *get_value(int idx, ...) {
    int p = 0;
    va_list indices;
    va_start(indices, idx);
    for (int i = 0; i < idx; ++i) {
      p += va_arg(indices, int) * stride_[i];
    }
    va_end(indices);
    return (char *) data_->data() + p;

  }

  Buffer *getBuffer() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  const void *operator[](int index) const { return (char *) data_->data() + index; }

  void *operator[](int index) { return (char *) data_->data() + index; }

 private:
  Buffer *data_;
  std::vector<int> shape_;
  std::vector<int> stride_;
  int size_;
  Datatype dtype_;
};
