/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include "jni_JniMath.h"
#include "mk_math.h"

template<typename T>
jobject getNewJObject(JNIEnv *env, T value, const char *class_name, const char *jtype) {
  jobject ret;
  jclass cls = env->FindClass(class_name);
  jmethodID init = env->GetMethodID(cls, "<init>", jtype);
  if (nullptr == init) return nullptr;
  ret = env->NewObject(cls, init, value);
  return ret;
}


/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    argMax
 * Signature: (Ljava/lang/Object;II[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_argMax
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int arg = 0;
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }

  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  arg = argmax(varr, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return arg;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    argMin
 * Signature: (Ljava/lang/Object;II[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_argMin
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int arg = 0;
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  arg = argmin(varr, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return arg;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    exp
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_exp___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    exp
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_exp___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    expC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_expC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    expC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_expC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    log
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_log___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    log
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_log___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    logC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_logC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    logC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_logC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sin
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sin___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sin
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sin___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sinC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sinC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sinC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sinC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cos
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cos___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cos
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cos___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cosC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cosC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cosC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cosC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([BII[I[I)B
 */
JNIEXPORT jbyte JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3BII_3I_3I
	(JNIEnv *env, jobject jobj, jbyteArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int8_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int8_t ret = array_max_int8(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([SII[I[I)S
 */
JNIEXPORT jshort JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3SII_3I_3I
	(JNIEnv *env, jobject jobj, jshortArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int16_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int16_t ret = array_max_int16(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([III[I[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3III_3I_3I
	(JNIEnv *env, jobject jobj, jintArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int32_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int32_t ret = array_max_int32(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([JII[I[I)J
 */
JNIEXPORT jlong JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3JII_3I_3I
	(JNIEnv *env, jobject jobj, jlongArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int64_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int64_t ret = array_max_int64(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([FII[I[I)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3FII_3I_3I
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (float *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  float ret = array_max_float(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_max
 * Signature: ([DII[I[I)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1max___3DII_3I_3I
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (double *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  double ret = array_max_double(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([BII[I[I)B
 */
JNIEXPORT jbyte JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3BII_3I_3I
	(JNIEnv *env, jobject jobj, jbyteArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int8_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int8_t ret = array_min_int8(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([SII[I[I)S
 */
JNIEXPORT jshort JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3SII_3I_3I
	(JNIEnv *env, jobject jobj, jshortArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int16_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int16_t ret = array_min_int16(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([III[I[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3III_3I_3I
	(JNIEnv *env, jobject jobj, jintArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int32_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int32_t ret = array_min_int32(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([JII[I[I)J
 */
JNIEXPORT jlong JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3JII_3I_3I
	(JNIEnv *env, jobject jobj, jlongArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int64_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int64_t ret = array_min_int64(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([FII[I[I)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3FII_3I_3I
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (float *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  float ret = array_min_float(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    array_min
 * Signature: ([DII[I[I)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_array_1min___3DII_3I_3I
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (double *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  double ret = array_min_double(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([BII[I[I)B
 */
JNIEXPORT jbyte JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3BII_3I_3I
	(JNIEnv *env, jobject jobj, jbyteArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int8_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int8_t ret = array_sum_int8(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([SII[I[I)S
 */
JNIEXPORT jshort JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3SII_3I_3I
	(JNIEnv *env, jobject jobj, jshortArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int16_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int16_t ret = array_sum_int16(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([III[I[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3III_3I_3I
	(JNIEnv *env, jobject jobj, jintArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int32_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int32_t ret = array_sum_int32(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([JII[I[I)J
 */
JNIEXPORT jlong JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3JII_3I_3I
	(JNIEnv *env, jobject jobj, jlongArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (int64_t *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  int64_t ret = array_sum_int64(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([FII[I[I)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3FII_3I_3I
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (float *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  float ret = array_sum_float(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    sum
 * Signature: ([DII[I[I)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_sum___3DII_3I_3I
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint offset, jint size, jintArray jshape, jintArray jstrides) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto varr = (double *)env->GetPrimitiveArrayCritical((jarray)j_arr, nullptr);

  double ret = array_sum_double(varr, offset, size, dim, shape, strides);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)j_arr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([BII[I[I[BI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3BII_3I_3I_3BI
	(JNIEnv *env, jobject jobj, jbyteArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jbyteArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 1; // byte type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (int8_t *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (int8_t *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type); // TODO(change signature?)

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([SII[I[I[SI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3SII_3I_3I_3SI
	(JNIEnv *env, jobject jobj, jshortArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jshortArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 2; // short type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (int16_t *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (int16_t *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([III[I[I[II)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3III_3I_3I_3II
	(JNIEnv *env, jobject jobj, jintArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jintArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 3; // int type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (int32_t *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (int32_t *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([JII[I[I[JI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3JII_3I_3I_3JI
	(JNIEnv *env, jobject jobj, jlongArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jlongArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 4; // long type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (int64_t *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (int64_t *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([FII[I[I[FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3FII_3I_3I_3FI
	(JNIEnv *env, jobject jobj, jfloatArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jfloatArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 5; // float type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (float *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (float *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_math_JniMath
 * Method:    cumSum
 * Signature: ([DII[I[I[DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_openblas_math_JniMath_cumSum___3DII_3I_3I_3DI
	(JNIEnv *env, jobject jobj, jdoubleArray jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jdoubleArray jout, jint axis) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;
  int type = 6; // double type

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  auto arr = (double *)env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);
  auto out = (double *)env->GetPrimitiveArrayCritical((jarray)jout, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}
