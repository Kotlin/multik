/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
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

    fun array_max(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    fun array_max(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    fun array_max(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    fun array_max(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    fun array_max(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    fun array_max(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    fun array_min(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    fun array_min(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    fun array_min(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    fun array_min(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    fun array_min(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    fun array_min(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    fun sum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Byte
    fun sum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Short
    fun sum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Int
    fun sum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Long
    fun sum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Float
    fun sum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?): Double

    fun cumSum(arr: ByteArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ByteArray, axis: Int = -1): Boolean
    fun cumSum(arr: ShortArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: ShortArray, axis: Int = -1): Boolean
    fun cumSum(arr: IntArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: IntArray, axis: Int = -1): Boolean
    fun cumSum(arr: LongArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: LongArray, axis: Int = -1): Boolean
    fun cumSum(arr: FloatArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: FloatArray, axis: Int = -1): Boolean
    fun cumSum(arr: DoubleArray, offset: Int, size: Int, shape: IntArray, strides: IntArray?, out: DoubleArray, axis: Int = -1): Boolean
}