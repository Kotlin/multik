#include "headers/ndarray.h"
#include "headers/_ndarray.h"
#include <iostream>
//#include <nsimd/cxx_adv_api.hpp>

/*
 * Class:     org_jetbrains_multik_jni_Basic
 * Method:    allocate
 * Signature: (Ljava/nio/Buffer;)J
 */
JNIEXPORT jlong JNICALL Java_org_jetbrains_multik_jni_Basic_allocate
    (JNIEnv *env, jobject object, jobject buffer, jintArray shape, jint datatype) {
  void *address = env->GetDirectBufferAddress(buffer);
  size_t size = env->GetDirectBufferCapacity(buffer);

  const int num_dims = env->GetArrayLength(shape);
  std::vector<int> dims(num_dims);
  if (num_dims > 0) {
    jint *shape_elems = env->GetIntArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i)
      dims[i] = shape_elems[i];
    env->ReleaseIntArrayElements(shape, shape_elems, JNI_ABORT);
  }

  Buffer *nav_buffer = new Buffer(address, size);
  Ndarray *ndarray = new Ndarray(nav_buffer, dims, datatype);

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

//template<typename T> T
void dot(Ndarray *a, Ndarray *b) {
  int *stride = new int[2]{3, 1};
  int *shape = new int[2]{3, 3};
  int r = 0;
  for (int *p_a = (int *) a->getBuffer()->data(); p_a < (int *) a->getBuffer()->data() + a->size();
       p_a += stride[0]) {
    for (int *p_b = (int *) b->getBuffer()->data();
         p_b < (int *) b->getBuffer()->data() + shape[1] && p_b < (int *) b->getBuffer()->data() + b->size();
         ++p_b) {
      int a1 = *(p_a);
      int a2 = *(p_a + stride[1]);
      int a3 = *(p_a + stride[1] + stride[1]);
      int b1 = *p_b;
      int b2 = *(p_b + stride[0]);
      int b3 = *(p_b + stride[0] + stride[0]);
      r = a1 * b1 + a2 * b2 + a3 * b3;
      r = 0;
    }
  }
}

/*
 * Class:     org_jetbrains_multik_jni_Basic
 * Method:    delete
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_Basic_dot
    (JNIEnv *env, jobject object, jlong handle_a, jlong handle_b) {
  dot((Ndarray *) handle_a, (Ndarray *) handle_b);
}

template<typename T>
T vdot_nsimd(T *first0, T *last0, T *first1) {
//  nsimd::pack<T> v(0);
//
//  size_t len = size_t(nsimd::len(nsimd::pack<T>()));
//  for (; first0 + len < last0; first0 += len, first1 += len) {
//    // Load current values
//    auto v0 = nsimd::loada<nsimd::pack<T> >(first0);
//    auto v1 = nsimd::loada<nsimd::pack<T> >(first1);
//    // Computation
//    v = nsimd::fma(v0, v1, v);
////      v = v + (v0 * v1);
//  }
//
//  T r = nsimd::addv(v); // horizontal SIMD vector summation
//
////  r += dot(first0, last0, first1);
//
//  return r;
}

