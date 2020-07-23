#include "headers/org_jetbrains_multik_jni_JniLinAlg.h"
#include "headers/mk_linalg.h"
#include <iostream>

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    pow
 * Signature: (Lorg/jetbrains/multik/core/Ndarray;I)Lorg/jetbrains/multik/core/Ndarray;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_pow
    (JNIEnv *env, jobject jobj, jobject matrix, jint n) {
}


/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    svd
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_svd
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    norm
 * Signature: (Lorg/jetbrains/multik/core/Ndarray;I)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_norm
    (JNIEnv *env, jobject jobj, jobject matrix, jint p) {
}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    cond
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_cond
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    det
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_det
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    matRank
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_matRank
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    solve
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_solve
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    inv
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_inv
    (JNIEnv *env, jobject object) {

}

/*
 * Class:     org_jetbrains_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([DII[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_multik_jni_JniLinAlg_dot___3DII_3DI_3D
    (JNIEnv *env, jobject jobj, jdoubleArray j_a, jint n, jint m, jdoubleArray j_b, jint k, jdoubleArray j_c) {
  double *A = (double *) env->GetPrimitiveArrayCritical(j_a, 0);
  double *B = (double *) env->GetPrimitiveArrayCritical(j_b, 0);
  double *C = (double *) env->GetPrimitiveArrayCritical(j_c, 0);

  matrix_dot(A, n, k, m, B, C);

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
    (JNIEnv *env, jobject jobj, jdoubleArray j_a, jint n, jint m, jdoubleArray j_b, jdoubleArray j_c) {
}