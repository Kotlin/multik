/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include <map>
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
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    argMax
 * Signature: (Ljava/lang/Object;II[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_argMax
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
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    argMin
 * Signature: (Ljava/lang/Object;II[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_argMin
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
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    exp
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_exp___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    exp
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_exp___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    expC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_expC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    expC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_expC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_exp_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    log
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_log___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    log
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_log___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    logC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_logC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    logC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_logC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_log_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    sin
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_sin___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    sin
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_sin___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    sinC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_sinC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    sinC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_sinC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_sin_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    cos
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_cos___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    cos
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_cos___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    cosC
 * Signature: ([FI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_cosC___3FI
	(JNIEnv *env, jobject jobj, jfloatArray j_arr, jint size) {
  auto arr = (float *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_complex_float(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    cosC
 * Signature: ([DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_cosC___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray j_arr, jint size) {
  auto arr = (double *)(env->GetPrimitiveArrayCritical(j_arr, nullptr));
  array_cos_complex_double(arr, size);
  env->ReleasePrimitiveArrayCritical(j_arr, arr, 0);
  return true;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    array_max
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_max
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int dim;
  jobject ret = nullptr;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  switch (type) {
	case 1: {
	  int8_t tmp = array_max_int8((int8_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Byte", "(B)V");
	  break;
	}
	case 2: {
	  int16_t tmp = array_max_int16((int16_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Short", "(S)V");
	  break;
	}
	case 3: {
	  int32_t tmp = array_max_int32((int32_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Integer", "(I)V");
	  break;
	}
	case 4: {
	  int64_t tmp = array_max_int64((int64_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Long", "(J)V");
	  break;
	}
	case 5: {
	  float tmp = array_max_float((float *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Float", "(F)V");
	  break;
	}
	case 6: {
	  double tmp = array_max_double((double *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Double", "(D)V");
	  break;
	}
	default:break;
  }

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    array_min
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_min
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int dim;
  jobject ret = nullptr;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  switch (type) {
	case 1: {
	  int8_t tmp = array_min_int8((int8_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Byte", "(B)V");
	  break;
	}
	case 2: {
	  int16_t tmp = array_min_int16((int16_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Short", "(S)V");
	  break;
	}
	case 3: {
	  int32_t tmp = array_min_int32((int32_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Integer", "(I)V");
	  break;
	}
	case 4: {
	  int64_t tmp = array_min_int64((int64_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Long", "(J)V");
	  break;
	}
	case 5: {
	  float tmp = array_min_float((float *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Float", "(F)V");
	  break;
	}
	case 6: {
	  double tmp = array_min_double((double *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Double", "(D)V");
	  break;
	}
	default:break;
  }

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    sum
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_sum
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int dim;
  jobject ret = nullptr;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  switch (type) {
	case 1: {
	  int8_t tmp = array_sum_int8((int8_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Byte", "(B)V");
	  break;
	}
	case 2: {
	  int16_t tmp = array_sum_int16((int16_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Short", "(S)V");
	  break;
	}
	case 3: {
	  int32_t tmp = array_sum_int32((int32_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Integer", "(I)V");
	  break;
	}
	case 4: {
	  int64_t tmp = array_sum_int64((int64_t *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Long", "(J)V");
	  break;
	}
	case 5: {
	  float tmp = array_sum_float((float *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Float", "(F)V");
	  break;
	}
	case 6: {
	  double tmp = array_sum_double((double *)varr, offset, size, dim, shape, strides);
	  ret = getNewJObject(env, tmp, "java/lang/Double", "(D)V");
	  break;
	}
	default:break;
  }

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_math_JniMath
 * Method:    cumSum
 * Signature: (Ljava/lang/Object;II[I[ILjava/lang/Object;II)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_kotlinx_multik_jni_math_JniMath_cumSum
	(JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size,
	 jintArray jshape, jintArray jstrides, jobject jout, jint axis, jint type) {
  int dim = 0;
  int *strides = nullptr;
  int *shape;

  if (jstrides != nullptr) {
	dim = env->GetArrayLength(jshape);
	strides = (int *)(env->GetPrimitiveArrayCritical(jstrides, nullptr));
	shape = (int *)(env->GetPrimitiveArrayCritical(jshape, nullptr));
  }
  void *arr = (env->GetPrimitiveArrayCritical((jarray)jarr, nullptr));
  void *out = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  array_cumsum(arr, out, offset, size, dim, shape, strides, type);

  if (strides != nullptr) {
	env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
	env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  }
  env->ReleasePrimitiveArrayCritical((jarray)jarr, arr, 0);
  env->ReleasePrimitiveArrayCritical((jarray)jout, out, 0);

  return true;
}
