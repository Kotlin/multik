/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray3DTest {

    private val shape = intArrayOf(2, 5, 3)
    private val dim = 3

    @Test
    fun createByteArray3DTest() {
        val inputArray = ByteArray(30) { it.toByte() }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createShortArray3DTest() {
        val inputArray = ShortArray(30) { it.toShort() }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createIntArray3DTest() {
        val inputArray = IntArray(30) { it }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createLongArray3DTest() {
        val inputArray = LongArray(30) { it.toLong() }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createFloatArray3DTest() {
        val inputArray = FloatArray(30) { it.toFloat() }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDoubleArray3DTest() {
        val inputArray = DoubleArray(30) { it.toDouble() }
        val a = mk.ndarray(inputArray, 2, 5, 3)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(30) { it }
        val a = mk.d3array(2, 5, 3) { it }
        assertEquals(mk.ndarray(inputArr, 2, 5, 3), a)

        val b = mk.d3array(2, 5, 3) { it * it }
        assertEquals(mk.ndarray<Int, D3>(inputArr.map { it * it }, shape), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3, 3)
        val l = Array(18) { it }.toList()
        val size_ = HashSet(l)
        val a: NDArray<Int, D3> = mk.ndarray(size_, shapeCol)
        val b: NDArray<Int, D3> = mk.ndarray(l, shapeCol)
        assertEquals(a, b)
    }
}
