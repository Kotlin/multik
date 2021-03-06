/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_jetbrains_kotlinx_multik_jni_JniLinAlg */

#ifndef _Included_org_jetbrains_kotlinx_multik_jni_JniLinAlg
#define _Included_org_jetbrains_kotlinx_multik_jni_JniLinAlg
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    pow
 * Signature: ([FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_pow___3FI_3F
	(JNIEnv *, jobject, jfloatArray, jint, jfloatArray);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    pow
 * Signature: ([DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_pow___3DI_3D
	(JNIEnv *, jobject, jdoubleArray, jint, jdoubleArray);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    norm
 * Signature: ([FI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_norm___3FI
	(JNIEnv *, jobject, jfloatArray, jint);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    norm
 * Signature: ([DI)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_norm___3DI
	(JNIEnv *, jobject, jdoubleArray, jint);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([FII[FI[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_dot___3FII_3FI_3F
	(JNIEnv *, jobject, jfloatArray, jint, jint, jfloatArray, jint, jfloatArray);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([DII[DI[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_dot___3DII_3DI_3D
	(JNIEnv *, jobject, jdoubleArray, jint, jint, jdoubleArray, jint, jdoubleArray);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([FII[F[F)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_dot___3FII_3F_3F
	(JNIEnv *, jobject, jfloatArray, jint, jint, jfloatArray, jfloatArray);

/*
 * Class:     org_jetbrains_kotlinx_multik_jni_JniLinAlg
 * Method:    dot
 * Signature: ([DII[D[D)V
 */
JNIEXPORT void JNICALL Java_org_jetbrains_kotlinx_multik_jni_JniLinAlg_dot___3DII_3D_3D
	(JNIEnv *, jobject, jdoubleArray, jint, jint, jdoubleArray, jdoubleArray);

#ifdef __cplusplus
}
#endif
#endif
