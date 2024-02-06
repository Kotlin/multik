#include "jni.h"

jobject newComplexDouble(JNIEnv *env, double re, double im) {
  jclass cls = env->FindClass("org/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble_jvmKt");
  if (cls == nullptr) {
    // TODO(exception handling)
  }

  jmethodID methodID = env->GetStaticMethodID(cls, "ComplexDouble", "(DD)Lorg/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble;");
  if (methodID == nullptr) {
    // TODO(exception handling)
  }

  jobject ret = env->CallStaticObjectMethod(cls, methodID, re, im);
  return ret;
}
