/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

#include <stdint.h>

#ifndef MULTIK_JNI_MULTIK_JNI_SRC_MAIN_HEADERS_MK_STAT_H_
#define MULTIK_JNI_MULTIK_JNI_SRC_MAIN_HEADERS_MK_STAT_H_
#ifdef __cplusplus
extern "C" {
#endif

double array_median(const void *arr, int size, int type);

#ifdef __cplusplus
}
#endif
#endif //MULTIK_JNI_MULTIK_JNI_SRC_MAIN_HEADERS_MK_STAT_H_
