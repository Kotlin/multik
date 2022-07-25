/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.dnarray
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArrayNDTest {

    private val shape = intArrayOf(2, 5, 3, 2, 1)
    private val dim = 5

    @Test
    fun createByteArrayNDTest() {
        val inputArray = ByteArray(60) { it.toByte() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createShortArrayNDTest() {
        val inputArray = ShortArray(60) { it.toShort() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createIntArrayNDTest() {
        val inputArray = IntArray(60) { it }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createLongArrayNDTest() {
        val inputArray = LongArray(60) { it.toLong() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createFloatArrayNDTest() {
        val inputArray = FloatArray(60) { it.toFloat() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDoubleArrayNDTest() {
        val inputArray = DoubleArray(60) { it.toDouble() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createComplexFloatArrayNDTest() {
        val inputArray = ComplexFloatArray(60) { ComplexFloat(it.toFloat(), it.toFloat()) }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createComplexDoubleArrayNDTest() {
        val inputArray = ComplexDoubleArray(60) { ComplexDouble(it.toDouble(), it.toDouble()) }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2, 1)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(60) { it }
        val a = mk.dnarray(2, 5, 3, 2, 1) { it }
        val check = mk.ndarray(inputArr, 2, 5, 3, 2, 1)
        assertEquals(check, a)

        val b = mk.dnarray(2, 5, 3, 2, 1) { it * it }
        assertEquals(mk.ndarray(inputArr.map { it * it }, shape), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3, 3, 2, 1)
        val l = Array(36) { it }.toList()
        val size_ = HashSet(l)
        val a: NDArray<Int, DN> = mk.ndarray(size_, shapeCol)
        val b: NDArray<Int, DN> = mk.ndarray(l, shapeCol)
        assertEquals(a, b)
    }
}