#include "jni_JniLinAlg.h"
#include "mk_linalg.h"

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    pow
 * Signature: ([FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_pow___3FI_3F
	(JNIEnv *env, jobject jobj, jfloatArray mat, jint n, jfloatArray result) {
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    pow
 * Signature: ([DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_pow___3DI_3D
	(JNIEnv *env, jobject jobj, jdoubleArray mat, jint n, jdoubleArray result) {
}


/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    norm
 * Signature: ([FI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_norm___3FI
	(JNIEnv *env, jobject jobj, jfloatArray mat, jint p) {
  return NULL;
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    norm
 * Signature: ([DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_norm___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray mat, jint p) {
  return NULL;
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([FII[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_dot___3FII_3FI_3F
	(JNIEnv *env, jobject jobj, jfloatArray j_a, jint m, jint n, jfloatArray j_b, jint k, jfloatArray j_c) {
  float *A = (float *)env->GetPrimitiveArrayCritical(j_a, 0);
  float *B = (float *)env->GetPrimitiveArrayCritical(j_b, 0);
  float *C = (float *)env->GetPrimitiveArrayCritical(j_c, 0);

  matrix_dot_float(A, m, n, k, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([DII[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_dot___3DII_3DI_3D
	(JNIEnv *env, jobject jobj, jdoubleArray j_a, jint m, jint n, jdoubleArray j_b, jint k, jdoubleArray j_c) {
  double *A = (double *)env->GetPrimitiveArrayCritical(j_a, 0);
  double *B = (double *)env->GetPrimitiveArrayCritical(j_b, 0);
  double *C = (double *)env->GetPrimitiveArrayCritical(j_c, 0);

  matrix_dot_double(A, m, n, k, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([FII[F[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_dot___3FII_3F_3F
	(JNIEnv *env, jobject jobj, jfloatArray j_a, jint m, jint n, jfloatArray j_b, jfloatArray j_c) {
  float *A = (float *)env->GetPrimitiveArrayCritical(j_a, 0);
  float *B = (float *)env->GetPrimitiveArrayCritical(j_b, 0);
  float *C = (float *)env->GetPrimitiveArrayCritical(j_c, 0);

  matrix_dot_vector_float(A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([DII[D[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_dot___3DII_3D_3D
	(JNIEnv *env, jobject jobj, jdoubleArray j_a, jint m, jint n, jdoubleArray j_b, jdoubleArray j_c) {
  double *A = (double *)env->GetPrimitiveArrayCritical(j_a, 0);
  double *B = (double *)env->GetPrimitiveArrayCritical(j_b, 0);
  double *C = (double *)env->GetPrimitiveArrayCritical(j_c, 0);

  matrix_dot_vector_double(A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}