/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include "jni_JniLinAlg.h"
#include "mk_linalg.h"

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    pow
 * Signature: ([FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_pow___3FI_3F
	(JNIEnv *env, jobject jobj, jfloatArray mat, jint n, jfloatArray result) {
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    pow
 * Signature: ([DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_pow___3DI_3D
	(JNIEnv *env, jobject jobj, jdoubleArray mat, jint n, jdoubleArray result) {
}


/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    norm
 * Signature: ([FI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_norm___3FI
	(JNIEnv *env, jobject jobj, jfloatArray mat, jint p) {
  return NULL;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    norm
 * Signature: ([DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_norm___3DI
	(JNIEnv *env, jobject jobj, jdoubleArray mat, jint p) {
  return NULL;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    inv
 * Signature: (I[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_inv__I_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jint strA) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_float(n, A, strA);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    inv
 * Signature: (I[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_inv__I_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jint strA) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_double(n, A, strA);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    solve
 * Signature: (II[FI[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_solve__II_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jfloatArray j_a, jint strA, jfloatArray j_b, jint strB) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_float(n, nrhs, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    solve
 * Signature: (II[DI[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_solve__II_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jdoubleArray j_a, jint strA, jdoubleArray j_b, jint strB) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_double(n, nrhs, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMM
 * Signature: (Z[FIIZ[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMM__Z_3FIIZ_3FI_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jfloatArray j_a, jint m, jint n,
	 jboolean trans_b, jfloatArray j_b, jint k, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot(trans_a, A, m, n, k, trans_b, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMM
 * Signature: (Z[DIIZ[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMM__Z_3DIIZ_3DI_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jdoubleArray j_a, jint m, jint n,
	 jboolean trans_b, jdoubleArray j_b, jint k, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot(trans_a, A, m, n, k, trans_b, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMMC
 * Signature: (Z[FIIZ[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMMC__Z_3FIIZ_3FI_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jfloatArray j_a, jint m, jint n,
	 jboolean trans_b, jfloatArray j_b, jint k, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex(trans_a, A, m, n, k, trans_b, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMMC
 * Signature: (Z[DIIZ[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMMC__Z_3DIIZ_3DI_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jdoubleArray j_a, jint m, jint n,
	 jboolean trans_b, jdoubleArray j_b, jint k, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex(trans_a, A, m, n, k, trans_b, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMV
 * Signature: (Z[FII[F[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMV__Z_3FII_3F_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jfloatArray j_a, jint m, jint n, jfloatArray j_b, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot(trans_a, A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMV
 * Signature: (Z[DII[D[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMV__Z_3DII_3D_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jdoubleArray j_a,
	 jint m, jint n, jdoubleArray j_b, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot(trans_a, A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMVC
 * Signature: (Z[FII[F[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMVC__Z_3FII_3F_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jfloatArray j_a, jint m, jint n, jfloatArray j_b, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex(trans_a, A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotMVC
 * Signature: (Z[DII[D[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotMVC__Z_3DII_3D_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jdoubleArray j_a,
	 jint m, jint n, jdoubleArray j_b, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex(trans_a, A, m, n, B, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotVV
 * Signature: (I[FI[FI)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotVV__I_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jint strA, jfloatArray j_b, jint strB) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  float ret = vector_dot(n, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotVV
 * Signature: (I[DI[DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotVV__I_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jint strA, jdoubleArray j_b, jint strB) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  double ret = vector_dot(n, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotVVC
 * Signature: (I[FI[FI)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotVVC__I_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jint strA, jfloatArray j_b, jint strB) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  // TODO: float ret = vector_dot_complex(n, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

//  return ret;
  return NULL;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg
 * Method:    dotVVC
 * Signature: (I[DI[DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_linalg_JniLinAlg_dotVVC__I_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jint strA, jdoubleArray j_b, jint strB) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  // TODO: double ret = vector_dot_complex(n, A, strA, B, strB);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

//  return ret;
  return NULL;
}
