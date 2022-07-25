/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include "JniStat.h"
#include "mk_stat.h"

/*
 * Class:     org_jetbrains_kotlinx_multik_openblas_stat_JniStat
 * Method:    median
 * Signature: (Ljava/lang/Object;II)D
 */
JNIEXPORT jdouble JNICALL Java_org_jetbrains_kotlinx_multik_openblas_stat_JniStat_median
	(JNIEnv *env, jobject jobj, jobject jarr, jint size, jint type) {

  void *varr = env->GetPrimitiveArrayCritical((jarray)jarr, nullptr);

  double ret = array_median(varr, size, type);

  env->ReleasePrimitiveArrayCritical((jarray)jarr, varr, 0);

  return ret;
}