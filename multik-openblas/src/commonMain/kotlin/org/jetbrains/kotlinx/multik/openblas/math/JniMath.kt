/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.math

internal expect object JniMath {
    fun argMax(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int
    fun argMin(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): Int

    fun exp(arr: FloatArray, size: Int): Boolean
    fun exp(arr: DoubleArray, size: Int): Boolean
    fun expC(arr: FloatArray, size: Int): Boolean
    fun expC(arr: DoubleArray, size: Int): Boolean

    fun log(arr: FloatArray, size: Int): Boolean
    fun log(arr: DoubleArray, size: Int): Boolean
    fun logC(arr: FloatArray, size: Int): Boolean
    fun logC(arr: DoubleArray, size: Int): Boolean

    fun sin(arr: FloatArray, size: Int): Boolean
    fun sin(arr: DoubleArray, size: Int): Boolean
    fun sinC(arr: FloatArray, size: Int): Boolean
    fun sinC(arr: DoubleArray, size: Int): Boolean

    fun cos(arr: FloatArray, size: Int): Boolean
    fun cos(arr: DoubleArray, size: Int): Boolean
    fun cosC(arr: FloatArray, size: Int): Boolean
    fun cosC(arr: DoubleArray, size: Int): Boolean

    fun <T : Number> array_max(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    fun <T : Number> array_min(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    fun <T : Number> sum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, dtype: Int): T
    fun cumSum(arr: Any, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: Any, axis: Int = -1, dtype: Int): Boolean
}