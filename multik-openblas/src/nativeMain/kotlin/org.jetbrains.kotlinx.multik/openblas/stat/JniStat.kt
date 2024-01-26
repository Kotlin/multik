/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.stat

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import org.jetbrains.kotlinx.multik.cinterop.array_median

@OptIn(ExperimentalForeignApi::class)
internal actual object JniStat {
    actual fun median(arr: Any, size: Int, dtype: Int): Double = when (arr) {
        is DoubleArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        is FloatArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        is IntArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        is LongArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        is ByteArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        is ShortArray -> arr.usePinned { array_median(it.addressOf(0), size, dtype) }
        else -> throw Exception("Only primitive arrays are supported for Kotlin/Native `median`")
    }
}