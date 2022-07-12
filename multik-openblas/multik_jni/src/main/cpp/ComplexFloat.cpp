#include "jni.h"

static jmethodID newComplexFloatID = nullptr;

jobject newComplexFloat(JNIEnv *env, float re, float im) {
  jobject ret;
  jclass cls = env->FindClass("org/jetbrains/kotlinx/multik/ndarray/complex/ComplexFloat");
  newComplexFloatID = env->GetMethodID(cls, "<init>", "(FF)V");
  if (newComplexFloatID == nullptr) return nullptr;
  ret = env->NewObject(cls, newComplexFloatID, re, im);
  return ret;
}
