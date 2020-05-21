#include "ndarray.h"
#include <iostream>

class Buffer {
 public:
  Buffer(size_t size, void *data) : size_(size), data_(data) {}

  ~Buffer() = default;

  void *data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  template<typename T>
  T *base() const {
    return reinterpret_cast<T *>(data());
  }

 private:
  void *data_;
  size_t size_;
};

class Ndarray {
 public:
  explicit Ndarray(Buffer *buffer) : buffer_(buffer) {}//, shape_(shape) {}
  ~Ndarray() = default;

// private:
  Buffer *buffer_;
  //int *shape_;
};

/*
 * Class:     org_jetbrains_multik_jni_Basic
 * Method:    allocate
 * Signature: (Ljava/nio/Buffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_jetbrains_multik_jni_Basic_allocate
    (JNIEnv *env, jobject object, jobject buffer) {
  void *address = env->GetDirectBufferAddress(buffer);
  size_t size = env->GetDirectBufferCapacity(buffer);
  Buffer *nav_buffer = new Buffer(size, address);
  Ndarray *ndarray = new Ndarray(nav_buffer);
  return reinterpret_cast<jlong>(ndarray);
}

/*
 * Class:     org_jetbrains_multik_jni_Basic
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_Basic_delete
    (JNIEnv *env, jobject object, jlong handle) {

}

