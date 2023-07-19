/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.stat

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.StableRef
import org.jetbrains.kotlinx.multik.cinterop.array_median

@OptIn(ExperimentalForeignApi::class)
internal actual object JniStat {
    actual fun median(arr: Any, size: Int, dtype: Int): Double =
        array_median(StableRef.create(arr).asCPointer(), size, dtype)
}