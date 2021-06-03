/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.*

public inline fun <reified T : Number> D1Array<T>.append(n: D1Array<T>): D1Array<T>{
    val totalSize :Int = this.size+n.size
    var dest = arrayOfNulls<T>(totalSize)
    System.arraycopy(Array<T>(this.size) { this.data[it] }, 0,dest, 0, this.size)
    System.arraycopy(Array<T>(n.size){n.data[it]}, 0, dest,this.size, n.size)
    var appendedArr:D1Array<T> = D1Array<T>(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
    return appendedArr
}

public inline fun <reified T : Number> D1Array<T>.append(n: T): D1Array<T>{
    val totalSize :Int = this.size+1
    var dest = arrayOfNulls<T>(totalSize)
    System.arraycopy(Array<T>(this.size) { this.data[it] }, 0,dest, 0, this.size)
    dest[totalSize-1] = n
    var appendedArr:D1Array<T> = D1Array<T>(data = initMemoryView<T>(totalSize, this.dtype){dest[it] as T},dim=D1, shape = intArrayOf(totalSize), dtype = this.dtype)
    return appendedArr
}
