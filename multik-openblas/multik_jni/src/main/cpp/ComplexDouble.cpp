#include "jni.h"

static jmethodID newComplexDoubleID = nullptr;

jobject newComplexDouble(JNIEnv *env, double re, double im) {
  jobject ret;
  jclass cls = env->FindClass("org/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble");
  newComplexDoubleID = env->GetMethodID(cls, "<init>", "(DD)V");
  if (newComplexDoubleID == nullptr) return nullptr;
  ret = env->NewObject(cls, newComplexDoubleID, re, im);
  return ret;
}