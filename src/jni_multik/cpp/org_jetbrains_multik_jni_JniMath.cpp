#include "headers/org_jetbrains_multik_jni_JniMath.h"
#include "headers/mk_math.h"

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    argMax
 * Signature: (Ljava/lang/Object;[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_multik_jni_JniMath_argMax
    (JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int arg = 0;
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmax(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);

  return arg;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    argMin
 * Signature: (Ljava/lang/Object;II[I[II)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_multik_jni_JniMath_argMin
    (JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  int arg = 0;
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      arg = array_argmin(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);

  return arg;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    exp
 * Signature: (Ljava/lang/Object;II[I[ILjava/lang/Object;I)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_multik_jni_JniMath_exp
    (JNIEnv *env,
     jobject jobj,
     jobject jarr,
     jint offset,
     jint size,
     jintArray jshape,
     jintArray jstrides,
     jdoubleArray jout,
     jint type) {
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  double *out = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical(jout, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_exp(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
  env->ReleasePrimitiveArrayCritical(jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    log
 * Signature: (Ljava/lang/Object;II[I[ILjava/lang/Object;I)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_multik_jni_JniMath_log
    (JNIEnv *env,
     jobject jobj,
     jobject jarr,
     jint offset,
     jint size,
     jintArray jshape,
     jintArray jstrides,
     jdoubleArray jout,
     jint type) {
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  double *out = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical(jout, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_log(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
  env->ReleasePrimitiveArrayCritical(jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    sin
 * Signature: (Ljava/lang/Object;II[I[I[DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_multik_jni_JniMath_sin
    (JNIEnv *env,
     jobject jobj,
     jobject jarr,
     jint offset,
     jint size,
     jintArray jshape,
     jintArray jstrides,
     jdoubleArray jout,
     jint type) {
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  double *out = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical(jout, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_sin(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
  env->ReleasePrimitiveArrayCritical(jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    cos
 * Signature: (Ljava/lang/Object;II[I[I[DI)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_multik_jni_JniMath_cos
    (JNIEnv *env,
     jobject jobj,
     jobject jarr,
     jint offset,
     jint size,
     jintArray jshape,
     jintArray jstrides,
     jdoubleArray jout,
     jint type) {
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  double *out = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical(jout, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      array_cos(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
  env->ReleasePrimitiveArrayCritical(jout, out, 0);

  return true;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    max
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_multik_jni_JniMath_max
    (JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  jobject ret = nullptr;
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Byte");
      jmethodID byteInit = env->GetMethodID(cls, "<init>", "(B)V");
      if (nullptr == byteInit) return nullptr;
      ret = env->NewObject(cls, byteInit, tmp);

      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Short");
      jmethodID shortInit = env->GetMethodID(cls, "<init>", "(S)V");
      if (nullptr == shortInit) return nullptr;
      ret = env->NewObject(cls, shortInit, tmp);

      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int32_t tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Integer");
      jmethodID integerInit = env->GetMethodID(cls, "<init>", "(I)V");
      if (nullptr == integerInit) return nullptr;
      ret = env->NewObject(cls, integerInit, tmp);

      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int64_t tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Long");
      jmethodID longInit = env->GetMethodID(cls, "<init>", "(J)V");
      if (nullptr == longInit) return nullptr;
      ret = env->NewObject(cls, longInit, tmp);

      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      float tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Float");
      jmethodID floatInit = env->GetMethodID(cls, "<init>", "(F)V");
      if (nullptr == floatInit) return nullptr;
      ret = env->NewObject(cls, floatInit, tmp);

      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      double tmp = array_max(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Double");
      jmethodID doubleInit = env->GetMethodID(cls, "<init>", "(D)V");
      if (nullptr == doubleInit) return nullptr;
      ret = env->NewObject(cls, doubleInit, tmp);

      break;
    }
    default:break;
  }

  return ret;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    min
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_multik_jni_JniMath_min
    (JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  jobject ret = nullptr;
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Byte");
      jmethodID byteInit = env->GetMethodID(cls, "<init>", "(B)V");
      if (nullptr == byteInit) return nullptr;
      ret = env->NewObject(cls, byteInit, tmp);

      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Short");
      jmethodID shortInit = env->GetMethodID(cls, "<init>", "(S)V");
      if (nullptr == shortInit) return nullptr;
      ret = env->NewObject(cls, shortInit, tmp);

      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int32_t tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Integer");
      jmethodID integerInit = env->GetMethodID(cls, "<init>", "(I)V");
      if (nullptr == integerInit) return nullptr;
      ret = env->NewObject(cls, integerInit, tmp);

      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int64_t tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Long");
      jmethodID longInit = env->GetMethodID(cls, "<init>", "(J)V");
      if (nullptr == longInit) return nullptr;
      ret = env->NewObject(cls, longInit, tmp);

      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      float tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Float");
      jmethodID floatInit = env->GetMethodID(cls, "<init>", "(F)V");
      if (nullptr == floatInit) return nullptr;
      ret = env->NewObject(cls, floatInit, tmp);

      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      double tmp = array_min(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Double");
      jmethodID doubleInit = env->GetMethodID(cls, "<init>", "(D)V");
      if (nullptr == doubleInit) return nullptr;
      ret = env->NewObject(cls, doubleInit, tmp);

      break;
    }
    default:break;
  }

  return ret;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    sum
 * Signature: (Ljava/lang/Object;II[I[II)Ljava/lang/Number;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_multik_jni_JniMath_sum
    (JNIEnv *env, jobject jobj, jobject jarr, jint offset, jint size, jintArray jshape, jintArray jstrides, jint type) {
  jobject ret = nullptr;
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Byte");
      jmethodID byteInit = env->GetMethodID(cls, "<init>", "(B)V");
      if (nullptr == byteInit) return nullptr;
      ret = env->NewObject(cls, byteInit, tmp);

      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int8_t tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Short");
      jmethodID shortInit = env->GetMethodID(cls, "<init>", "(S)V");
      if (nullptr == shortInit) return nullptr;
      ret = env->NewObject(cls, shortInit, tmp);

      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int32_t tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Integer");
      jmethodID integerInit = env->GetMethodID(cls, "<init>", "(I)V");
      if (nullptr == integerInit) return nullptr;
      ret = env->NewObject(cls, integerInit, tmp);

      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      int64_t tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Long");
      jmethodID longInit = env->GetMethodID(cls, "<init>", "(J)V");
      if (nullptr == longInit) return nullptr;
      ret = env->NewObject(cls, longInit, tmp);

      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      float tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Float");
      jmethodID floatInit = env->GetMethodID(cls, "<init>", "(F)V");
      if (nullptr == floatInit) return nullptr;
      ret = env->NewObject(cls, floatInit, tmp);

      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      double tmp = array_sum(arr, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
      env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);

      jclass cls = env->FindClass("java/lang/Double");
      jmethodID doubleInit = env->GetMethodID(cls, "<init>", "(D)V");
      if (nullptr == doubleInit) return nullptr;
      ret = env->NewObject(cls, doubleInit, tmp);

      break;
    }
    default:break;
  }

  return ret;
}

/*
 * Class:     org_jetbrains_multik_jni_JniMath
 * Method:    cumSum
 * Signature: (Ljava/lang/Object;II[I[ILjava/lang/Object;II)Z
 */
JNIEXPORT jboolean JNICALL Java_org_jetbrains_multik_jni_JniMath_cumSum
    (JNIEnv *env,
     jobject jobj,
     jobject jarr,
     jint offset,
     jint size,
     jintArray jshape,
     jintArray jstrides,
     jobject jout,
     jint axis,
     jint type) {
  int dim = env->GetArrayLength(jshape);
  int *shape = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jshape, nullptr));
  int *strides = reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(jstrides, nullptr));
  switch (type) {
    case 1: {
      auto *arr = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<int8_t *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    case 2: {
      auto *arr = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<int16_t *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    case 3: {
      auto *arr = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<int32_t *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    case 4: {
      auto *arr = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<int64_t *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    case 5: {
      auto *arr = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<float *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    case 6: {
      auto *arr = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jarr, nullptr));
      auto *out = reinterpret_cast<double *>(env->GetPrimitiveArrayCritical((jarray) jout, nullptr));
      array_cumsum(arr, out, offset, size, dim, shape, strides);
      env->ReleasePrimitiveArrayCritical((jarray) jarr, arr, 0);
      env->ReleasePrimitiveArrayCritical((jarray) jout, out, 0);
      break;
    }
    default:break;
  }

  env->ReleasePrimitiveArrayCritical(jshape, shape, 0);
  env->ReleasePrimitiveArrayCritical(jstrides, strides, 0);

  return true;
}
