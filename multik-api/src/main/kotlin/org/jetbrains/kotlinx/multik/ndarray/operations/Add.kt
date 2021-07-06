/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.*

public fun <T : Number> D1Array<T>.append(n: D1Array<T>): D1Array<T>{
    val totalSize :Int = this.size+n.size
    when (this.dtype) {
        DataType.IntDataType -> {
            var dest = IntArray(totalSize)
            System.arraycopy(this.data.getIntArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getIntArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.FloatDataType -> {
            var dest = FloatArray(totalSize)
            System.arraycopy(this.data.getFloatArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getFloatArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.DoubleDataType -> {
            var dest = DoubleArray(totalSize)
            System.arraycopy(this.data.getDoubleArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getDoubleArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.LongDataType -> {
            var dest = LongArray(totalSize)
            System.arraycopy(this.data.getLongArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getLongArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.ShortDataType -> {
            var dest = ShortArray(totalSize)
            System.arraycopy(this.data.getShortArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getShortArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.ByteDataType -> {
            var dest = ByteArray(totalSize)
            System.arraycopy(this.data.getByteArray(),0,dest,0,this.size)
            System.arraycopy(n.data.getByteArray(), 0, dest,this.size, n.size)
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
    }

}

public fun <T : Number> D1Array<T>.append(n: T): D1Array<T>{
    val totalSize :Int = this.size+1
    when (this.dtype) {
        DataType.IntDataType -> {
            var dest = IntArray(totalSize)
            System.arraycopy(this.data.getIntArray(),0,dest,0,this.size)
            dest[this.size] = n as Int
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.FloatDataType -> {
            var dest = FloatArray(totalSize)
            System.arraycopy(this.data.getFloatArray(),0,dest,0,this.size)
            dest[this.size] = n as Float
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.DoubleDataType -> {
            var dest = DoubleArray(totalSize)
            System.arraycopy(this.data.getDoubleArray(),0,dest,0,this.size)
            dest[this.size] = n as Double
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.LongDataType -> {
            var dest = LongArray(totalSize)
            System.arraycopy(this.data.getLongArray(),0,dest,0,this.size)
            dest[this.size] = n as Long
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.ShortDataType -> {
            var dest = ShortArray(totalSize)
            System.arraycopy(this.data.getShortArray(),0,dest,0,this.size)
            dest[this.size] = n as Short
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
        DataType.ByteDataType -> {
            var dest = ByteArray(totalSize)
            System.arraycopy(this.data.getByteArray(),0,dest,0,this.size)
            dest[this.size] = n as Byte
            var appendedArr:D1Array<T> = D1Array(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
            return appendedArr
        }
    }
}
