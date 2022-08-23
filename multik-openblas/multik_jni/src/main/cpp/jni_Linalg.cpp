/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include "jni_JniLinAlg.h"
#include "mk_linalg.h"
#include <ComplexFloat.h>
#include <ComplexDouble.h>

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    pow
 * Signature: ([FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_pow___3FI_3F
	(JNIEnv *env, jobject jobj, jfloatArray mat, jint n, jfloatArray result) {
  // TODO(Optimize pow)
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    pow
 * Signature: ([DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_pow___3DI_3D
	(JNIEnv *env, jobject jobj, jdoubleArray mat, jint n, jdoubleArray result) {
  // TODO(Optimize pow)
}


/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    norm
 * Signature: (CII[FI)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_norm__CII_3FI
	(JNIEnv *env, jobject jobj, jchar jnorm, jint m, jint n, jfloatArray jarr, jint lda) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(jarr, nullptr);
  float ret = norm_matrix_float((char)jnorm, m, n, A, lda);
  env->ReleasePrimitiveArrayCritical(jarr, A, 0);
  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    norm
 * Signature: (CII[DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_norm__CII_3DI
	(JNIEnv *env, jobject jobj, jchar jnorm, jint m, jint n, jdoubleArray jarr, jint lda) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(jarr, nullptr);
  double ret = norm_matrix_double((char)jnorm, m, n, A, lda);
  env->ReleasePrimitiveArrayCritical(jarr, A, 0);
  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    inv
 * Signature: (I[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_inv__I_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jint lda) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_float(n, A, lda);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    inv
 * Signature: (I[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_inv__I_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jint lda) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_double(n, A, lda);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    invC
 * Signature: (I[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_invC__I_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jint lda) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_complex_float(n, A, lda);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    invC
 * Signature: (I[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_invC__I_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jint lda) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);

  int info = inverse_matrix_complex_double(n, A, lda);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    qr
 * Signature: (II[FI[F)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_qr__II_3FI_3F
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_aq, jint lda, jfloatArray j_r) {
  auto *AQ = (float *)env->GetPrimitiveArrayCritical(j_aq, nullptr);
  auto *R = (float *)env->GetPrimitiveArrayCritical(j_r, nullptr);

  int info = qr_matrix_float(m, n, AQ, lda, R);

  env->ReleasePrimitiveArrayCritical(j_aq, AQ, 0);
  env->ReleasePrimitiveArrayCritical(j_r, R, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    qr
 * Signature: (II[DI[D)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_qr__II_3DI_3D
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_aq, jint lda, jdoubleArray j_r) {
  auto *AQ = (double *)env->GetPrimitiveArrayCritical(j_aq, nullptr);
  auto *R = (double *)env->GetPrimitiveArrayCritical(j_r, nullptr);

  int info = qr_matrix_double(m, n, AQ, lda, R);

  env->ReleasePrimitiveArrayCritical(j_aq, AQ, 0);
  env->ReleasePrimitiveArrayCritical(j_r, R, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    qrC
 * Signature: (II[FI[F)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_qrC__II_3FI_3F
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_aq, jint lda, jfloatArray j_r) {
  auto *AQ = (float *)env->GetPrimitiveArrayCritical(j_aq, nullptr);
  auto *R = (float *)env->GetPrimitiveArrayCritical(j_r, nullptr);

  int info = qr_matrix_complex_float(m, n, AQ, lda, R);

  env->ReleasePrimitiveArrayCritical(j_aq, AQ, 0);
  env->ReleasePrimitiveArrayCritical(j_r, R, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    qrC
 * Signature: (II[DI[D)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_qrC__II_3DI_3D
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_aq, jint lda, jdoubleArray j_r) {
  auto *AQ = (double *)env->GetPrimitiveArrayCritical(j_aq, nullptr);
  auto *R = (double *)env->GetPrimitiveArrayCritical(j_r, nullptr);

  int info = qr_matrix_complex_double(m, n, AQ, lda, R);

  env->ReleasePrimitiveArrayCritical(j_aq, AQ, 0);
  env->ReleasePrimitiveArrayCritical(j_r, R, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    plu
 * Signature: (II[FI[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_plu__II_3FI_3I
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_a, jint lda, jintArray j_ipiv) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *IPIV = (int *)env->GetPrimitiveArrayCritical(j_ipiv, nullptr);

  int info = plu_matrix_float(m, n, A, lda, IPIV);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_ipiv, IPIV, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    plu
 * Signature: (II[DI[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_plu__II_3DI_3I
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_a, jint lda, jintArray j_ipiv) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *IPIV = (int *)env->GetPrimitiveArrayCritical(j_ipiv, nullptr);

  int info = plu_matrix_double(m, n, A, lda, IPIV);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_ipiv, IPIV, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    pluC
 * Signature: (II[FI[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_pluC__II_3FI_3I
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_a, jint lda, jintArray j_ipiv) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *IPIV = (int *)env->GetPrimitiveArrayCritical(j_ipiv, nullptr);

  int info = plu_matrix_complex_float(m, n, A, lda, IPIV);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_ipiv, IPIV, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    pluC
 * Signature: (II[DI[I)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_pluC__II_3DI_3I
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_a, jint lda, jintArray j_ipiv) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *IPIV = (int *)env->GetPrimitiveArrayCritical(j_ipiv, nullptr);

  int info = plu_matrix_complex_double(m, n, A, lda, IPIV);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_ipiv, IPIV, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    svd
 * Signature: (II[FI[F[FI[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_svd__II_3FI_3F_3FI_3FI
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_a, jint lda,
	 jfloatArray j_s, jfloatArray j_u, jint ldu, jfloatArray j_vt, jint ldvt) {
//  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *S = (float *)env->GetPrimitiveArrayCritical(j_s, nullptr);
//  auto *U = (float *)env->GetPrimitiveArrayCritical(j_u, nullptr);
//  auto *VT = (float *)env->GetPrimitiveArrayCritical(j_vt, nullptr);
//
//  int info = svd_matrix_float(m, n, A, lda, S, U, ldu, VT, ldvt);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_s, S, 0);
//  env->ReleasePrimitiveArrayCritical(j_u, U, 0);
//  env->ReleasePrimitiveArrayCritical(j_vt, VT, 0);
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    svd
 * Signature: (II[DI[D[DI[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_svd__II_3DI_3D_3DI_3DI
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_a, jint lda,
	 jdoubleArray j_s, jdoubleArray j_u, jint ldu, jdoubleArray j_vt, jint ldvt) {
//  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *S = (double *)env->GetPrimitiveArrayCritical(j_s, nullptr);
//  auto *U = (double *)env->GetPrimitiveArrayCritical(j_u, nullptr);
//  auto *VT = (double *)env->GetPrimitiveArrayCritical(j_vt, nullptr);
//
//  int info = svd_matrix_double(m, n, A, lda, S, U, ldu, VT, ldvt);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_s, S, 0);
//  env->ReleasePrimitiveArrayCritical(j_u, U, 0);
//  env->ReleasePrimitiveArrayCritical(j_vt, VT, 0);
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    svdC
 * Signature: (II[FI[F[FI[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_svdC__II_3FI_3F_3FI_3FI
	(JNIEnv *env, jobject jobj, jint m, jint n, jfloatArray j_a, jint lda,
	 jfloatArray j_s, jfloatArray j_u, jint ldu, jfloatArray j_vt, jint ldvt) {
//  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *S = (float *)env->GetPrimitiveArrayCritical(j_s, nullptr);
//  auto *U = (float *)env->GetPrimitiveArrayCritical(j_u, nullptr);
//  auto *VT = (float *)env->GetPrimitiveArrayCritical(j_vt, nullptr);
//
//  int info = svd_matrix_complex_float(m, n, A, lda, S, U, ldu, VT, ldvt);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_s, S, 0);
//  env->ReleasePrimitiveArrayCritical(j_u, U, 0);
//  env->ReleasePrimitiveArrayCritical(j_vt, VT, 0);
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    svdC
 * Signature: (II[DI[D[DI[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_svdC__II_3DI_3D_3DI_3DI
	(JNIEnv *env, jobject jobj, jint m, jint n, jdoubleArray j_a, jint lda,
	 jdoubleArray j_s, jdoubleArray j_u, jint ldu, jdoubleArray j_vt, jint ldvt) {
//  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *S = (double *)env->GetPrimitiveArrayCritical(j_s, nullptr);
//  auto *U = (double *)env->GetPrimitiveArrayCritical(j_u, nullptr);
//  auto *VT = (double *)env->GetPrimitiveArrayCritical(j_vt, nullptr);
//
//  int info = svd_matrix_complex_double(m, n, A, lda, S, U, ldu, VT, ldvt);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_s, S, 0);
//  env->ReleasePrimitiveArrayCritical(j_u, U, 0);
//  env->ReleasePrimitiveArrayCritical(j_vt, VT, 0);
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    eig
 * Signature: (I[F[FC[F)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_eig__I_3F_3FC_3F
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_a, jfloatArray j_w, jchar compute_v, jfloatArray j_v) {
//  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *W = (float *)env->GetPrimitiveArrayCritical(j_w, nullptr);
//  float *V;
//  if (j_v == nullptr) {
//	V = nullptr;
//  } else {
//	V = (float *)env->GetPrimitiveArrayCritical(j_v, nullptr);
//  }
//
//  int info = eigen_float(n, A, W, (char)compute_v, V);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_w, W, 0);
//  if (j_v != nullptr) {
//	env->ReleasePrimitiveArrayCritical(j_v, V, 0);
//  }
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    eig
 * Signature: (I[D[DC[D)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_eig__I_3D_3DC_3D
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_a, jdoubleArray j_w, jchar compute_v, jdoubleArray j_v) {
//  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
//  auto *W = (double *)env->GetPrimitiveArrayCritical(j_w, nullptr);
//  double *V;
//  if (j_v == nullptr) {
//	V = nullptr;
//  } else {
//	V = (double *)env->GetPrimitiveArrayCritical(j_v, nullptr);
//  }
//
//  int info = eigen_double(n, A, W, (char)compute_v, V);
//
//  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
//  env->ReleasePrimitiveArrayCritical(j_w, W, 0);
//  if (j_v != nullptr) {
//	env->ReleasePrimitiveArrayCritical(j_v, V, 0);
//  }
//
//  return info;
  return -1;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    solve
 * Signature: (II[FI[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_solve__II_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jfloatArray j_a, jint lda, jfloatArray j_b, jint ldb) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_float(n, nrhs, A, lda, B, ldb);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    solve
 * Signature: (II[DI[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_solve__II_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jdoubleArray j_a, jint lda, jdoubleArray j_b, jint ldb) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_double(n, nrhs, A, lda, B, ldb);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    solveC
 * Signature: (II[FI[FI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_solveC__II_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jfloatArray j_a, jint lda, jfloatArray j_b, jint ldb) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_complex_float(n, nrhs, A, lda, B, ldb);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    solveC
 * Signature: (II[DI[DI)I
 */
JNIEXPORT jint JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_solveC__II_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jint nrhs, jdoubleArray j_a, jint lda, jdoubleArray j_b, jint ldb) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);

  int info = solve_linear_system_complex_double(n, nrhs, A, lda, B, ldb);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);

  return info;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMM
 * Signature: (ZI[FIIIZI[FII[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMM__ZI_3FIIIZI_3FII_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jfloatArray j_a, jint m, jint k, jint lda,
	 jboolean trans_b, jint offset_b, jfloatArray j_b, jint n, jint ldb, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_float(trans_a, offset_a, A, lda, m, n, k, trans_b, offset_b, B, ldb, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMM
 * Signature: (ZI[DIIIZI[DII[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMM__ZI_3DIIIZI_3DII_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jdoubleArray j_a, jint m, jint k, jint lda,
	 jboolean trans_b, jint offset_b, jdoubleArray j_b, jint n, jint ldb, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_double(trans_a, offset_a, A, lda, m, n, k, trans_b, offset_b, B, ldb, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMMC
 * Signature: (ZI[FIIIZI[FII[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMMC__ZI_3FIIIZI_3FII_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jfloatArray j_a, jint m, jint k, jint lda,
	 jboolean trans_b, jint offset_b, jfloatArray j_b, jint n, jint ldb, jfloatArray j_c) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (float *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex_float(trans_a, offset_a, A, lda, m, n, k, trans_b, offset_b, B, ldb, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMMC
 * Signature: (ZI[DIIIZI[DII[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMMC__ZI_3DIIIZI_3DII_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jdoubleArray j_a, jint m, jint k, jint lda,
	 jboolean trans_b, jint offset_b, jdoubleArray j_b, jint n, jint ldb, jdoubleArray j_c) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_b, nullptr);
  auto *C = (double *)env->GetPrimitiveArrayCritical(j_c, nullptr);

  matrix_dot_complex_double(trans_a, offset_a, A, lda, m, n, k, trans_b, offset_b, B, ldb, C);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_b, B, 0);
  env->ReleasePrimitiveArrayCritical(j_c, C, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMV
 * Signature: (ZI[FIII[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMV__ZI_3FIII_3FI_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jfloatArray j_a, jint m, jint n, jint lda,
	 jfloatArray j_x, jint incx, jfloatArray j_y) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *X = (float *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (float *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  matrix_dot_vector_float(trans_a, offset_a, A, lda, m, n, X, incx, Y);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMV
 * Signature: (ZI[DIII[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMV__ZI_3DIII_3DI_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jdoubleArray j_a, jint m, jint n, jint lda,
	 jdoubleArray j_x, jint incx, jdoubleArray j_y) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *X = (double *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (double *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  matrix_dot_vector_double(trans_a, offset_a, A, lda, m, n, X, incx, Y);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMVC
 * Signature: (ZI[FIII[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMVC__ZI_3FIII_3FI_3F
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jfloatArray j_a, jint m, jint n, jint lda,
	 jfloatArray j_x, jint incx, jfloatArray j_y) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *X = (float *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (float *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  matrix_dot_complex_vector_float(trans_a, offset_a, A, lda, m, n, X, incx, Y);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotMVC
 * Signature: (ZI[DIII[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotMVC__ZI_3DIII_3DI_3D
	(JNIEnv *env, jobject jobj, jboolean trans_a, jint offset_a, jdoubleArray j_a, jint m, jint n, jint lda,
	 jdoubleArray j_x, jint incx, jdoubleArray j_y) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_a, nullptr);
  auto *X = (double *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (double *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  matrix_dot_complex_vector_double(trans_a, offset_a, A, lda, m, n, X, incx, Y);

  env->ReleasePrimitiveArrayCritical(j_a, A, 0);
  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotVV
 * Signature: (I[FI[FI)F
 */
JNIEXPORT jfloat JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotVV__I_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_x, jint incx, jfloatArray j_y, jint incy) {
  auto *X = (float *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (float *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  float ret = vector_dot_float(n, X, incx, Y, incy);

  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotVV
 * Signature: (I[DI[DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotVV__I_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_x, jint incx, jdoubleArray j_y, jint incy) {
  auto *X = (double *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *Y = (double *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  double ret = vector_dot_double(n, X, incx, Y, incy);

  env->ReleasePrimitiveArrayCritical(j_x, X, 0);
  env->ReleasePrimitiveArrayCritical(j_y, Y, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotVVC
 * Signature: (I[FI[FI)Lorg/jetbrains/kotlinx/multik/ndarray/complex/ComplexFloat;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotVVC__I_3FI_3FI
	(JNIEnv *env, jobject jobj, jint n, jfloatArray j_x, jint incx, jfloatArray j_y, jint incy) {
  auto *A = (float *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *B = (float *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  mk_complex_float cf = vector_dot_complex_float(n, A, incx, B, incy);
  jobject ret = newComplexFloat(env, cf.real, cf.imag);

  env->ReleasePrimitiveArrayCritical(j_x, A, 0);
  env->ReleasePrimitiveArrayCritical(j_y, B, 0);

  return ret;
}

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg
 * Method:    dotVVC
 * Signature: (I[DI[DI)Lorg/jetbrains/kotlinx/multik/ndarray/complex/ComplexDouble;
 */
JNIEXPORT jobject JNICALL Java_org_jetbrains_kotlinx_multik_openblas_linalg_JniLinAlg_dotVVC__I_3DI_3DI
	(JNIEnv *env, jobject jobj, jint n, jdoubleArray j_x, jint incx, jdoubleArray j_y, jint incy) {
  auto *A = (double *)env->GetPrimitiveArrayCritical(j_x, nullptr);
  auto *B = (double *)env->GetPrimitiveArrayCritical(j_y, nullptr);

  mk_complex_double cd = vector_dot_complex_double(n, A, incx, B, incy);
  jobject ret = newComplexDouble(env, cd.real, cd.imag);

  env->ReleasePrimitiveArrayCritical(j_x, A, 0);
  env->ReleasePrimitiveArrayCritical(j_y, B, 0);

  return ret;
}
