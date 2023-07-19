#include "jni.h"

static jmethodID newComplexDoubleID = nullptr;

jobject newComplexDouble(JNIEnv *env, double re, double im) {
  jobject ret;
  jclass cls = env->FindClass("org/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble");
  jfieldID companionField = env->GetStaticFieldID(cls, "Companion", "Lorg/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble$Companion;");
  jobject companionObject = env->GetStaticObjectField(cls, companionField);
  jclass companionClass = env->GetObjectClass(companionObject);
  newComplexDoubleID = env->GetMethodID(companionClass, "invoke", "(DD)Lorg/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble;");
  ret = env->CallObjectMethod(companionObject, newComplexDoubleID, re, im);
  return ret;
}
