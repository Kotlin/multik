/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*

public fun <T : Number> D1Array<T>.append(n: D1Array<T>): D1Array<T> {
    val totalSize: Int = this.size + n.size
    val appendedArray = when (this.dtype) {
        DataType.IntDataType -> {
            val dest = IntArray(totalSize)
            System.arraycopy(this.data.getIntArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getIntArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewIntArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.ByteDataType -> {
            val dest = ByteArray(totalSize)
            System.arraycopy(this.data.getByteArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getByteArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewByteArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.ShortDataType -> {
            val dest = ShortArray(totalSize)
            System.arraycopy(this.data.getShortArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getShortArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewShortArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.LongDataType -> {
            val dest = LongArray(totalSize)
            System.arraycopy(this.data.getLongArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getLongArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewLongArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.FloatDataType -> {
            val dest = FloatArray(totalSize)
            System.arraycopy(this.data.getFloatArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getFloatArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewFloatArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.DoubleDataType -> {
            val dest = DoubleArray(totalSize)
            System.arraycopy(this.data.getDoubleArray(), 0, dest, 0, this.size)
            System.arraycopy(n.data.getDoubleArray(), 0, dest, this.size, n.size)
            D1Array(data = MemoryViewDoubleArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
    }
    return appendedArray as D1Array<T>
}

public fun <T : Number> D1Array<T>.append(n: T): D1Array<T> {
    val totalSize: Int = this.size + 1
    val appendedArray = when (this.dtype) {
        DataType.ByteDataType -> {
            val dest = ByteArray(totalSize)
            System.arraycopy(this.data.getByteArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Byte
            D1Array(data = MemoryViewByteArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.ShortDataType -> {
            val dest = ShortArray(totalSize)
            System.arraycopy(this.data.getShortArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Short
            D1Array(data = MemoryViewShortArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.IntDataType -> {
            val dest = IntArray(totalSize)
            System.arraycopy(this.data.getIntArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Int
            D1Array(data = MemoryViewIntArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.LongDataType -> {
            val dest = LongArray(totalSize)
            System.arraycopy(this.data.getLongArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Long
            D1Array(data = MemoryViewLongArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.FloatDataType -> {
            val dest = FloatArray(totalSize)
            System.arraycopy(this.data.getFloatArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Float
            D1Array(data = MemoryViewFloatArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
        DataType.DoubleDataType -> {
            val dest = DoubleArray(totalSize)
            System.arraycopy(this.data.getDoubleArray(), 0, dest, 0, this.size)
            dest[this.size] = n as Double
            D1Array(data = MemoryViewDoubleArray(dest), dim = D1, shape = intArrayOf(totalSize), dtype = this.dtype)
        }
    }
    return appendedArray as D1Array<T>
}
