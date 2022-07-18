/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.stat


internal expect object JniStat {
    fun median(arr: Any, size: Int, dtype: Int): Double
}