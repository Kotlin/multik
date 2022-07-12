/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_kotlin.math

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.kotlin.math.KEMath
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class KEMathTest {

    @Test
    fun test_of_argMax_function_Double() {
        val ndarray1 = mk.ndarray(mk[0.008830892, 0.7638366187, -0.0401326368965, -0.269757419187])
        val ndarray2 = mk.ndarray(mk[0.0088308926050, 0.763836618743, Double.NaN, -0.2697574191872])

        assertEquals(1, mk.math.argMax(ndarray1))
        assertEquals(2, mk.math.argMax(ndarray2))
    }

    @Test
    fun test_of_argMax_function_with_axis() {
        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[0, 2], mk[2, 2]])
        val expectedWith1And2Axes = mk.ndarray(mk[mk[0, 1], mk[1, 1], mk[1, 1]])

        assertEquals(expectedWith0Axis, mk.math.argMaxD3(ndarray, axis = 0))
        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 1))
        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 2))
    }

    @Test
    fun test_of_argMin_function_with_axis() {
        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[1, 0], mk[0, 0]])
        val expectedWith1And2Axes = mk.ndarray(mk[mk[1, 0], mk[0, 0], mk[0, 0]])

        assertEquals(expectedWith0Axis, mk.math.argMinD3(ndarray, axis = 0))
        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 1))
        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 2))
    }

    @Test
    fun test_of_max_function_with_axis() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[7, 9], mk[10, 11]])
        val expectedWith1Axis = mk.ndarray(mk[mk[1, 4], mk[6, 8], mk[10, 11]])
        val expectedWith2Axis = mk.ndarray(mk[mk[3, 4], mk[5, 8], mk[9, 11]])

        assertEquals(expectedWith0Axis, mk.math.max(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.max(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.max(ndarray, axis = 2))
    }

    @Test
    fun test_of_min_function_with_axis() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
        val expectedWith1Axis = mk.ndarray(mk[mk[0, 3], mk[2, 5], mk[7, 9]])
        val expectedWith2Axis = mk.ndarray(mk[mk[0, 1], mk[2, 6], mk[7, 10]])

        assertEquals(expectedWith0Axis, mk.math.min(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.min(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.min(ndarray, axis = 2))
    }

    @Test
    fun test_of_sum_function_with_axis_on_flat_ndarray() {
        val ndarray = mk.ndarray(mk[0, 3, 1, 4])
        assertFailsWith<IllegalArgumentException> { KEMath.sum<Int, D1, D1>(ndarray, 0) }
    }

    @Test
    fun test_of_sum_function_with_axis_on_2d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])

        val expectedWith0Axis = mk.ndarray(mk[1, 7])
        assertEquals(expectedWith0Axis, KEMath.sumD2(ndarray, 0))

        val expectedWith1Axis = mk.ndarray(mk[3, 5])
        assertEquals(expectedWith1Axis, KEMath.sumD2(ndarray, 1))
    }

    @Test
    fun test_of_sum_function_with_axis_on_3d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[9, 17], mk[17, 23]])
        val expectedWith1Axis = mk.ndarray(mk[mk[1, 7], mk[8, 13], mk[17, 20]])
        val expectedWith2Axis = mk.ndarray(mk[mk[3, 5], mk[7, 14], mk[16, 21]])

        assertEquals(expectedWith0Axis, mk.math.sumD3(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.sumD3(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.sumD3(ndarray, axis = 2))
    }

    @Test
    fun test_of_sum_function_with_third_axis_on_2d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
        assertFailsWith<IllegalArgumentException> { KEMath.sumD2(ndarray, 2) }
    }

    @Test
    fun test_of_cumSum_function_with_axis_on_3d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis =
            mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 8], mk[7, 12]], mk[mk[9, 17], mk[17, 23]]])
        val expectedWith1Axis =
            mk.ndarray(mk[mk[mk[0, 3], mk[1, 7]], mk[mk[2, 5], mk[8, 13]], mk[mk[7, 9], mk[17, 20]]])
        val expectedWith2Axis =
            mk.ndarray(mk[mk[mk[0, 3], mk[1, 5]], mk[mk[2, 7], mk[6, 14]], mk[mk[7, 16], mk[10, 21]]])


        assertEquals(expectedWith0Axis, mk.math.cumSum(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.cumSum(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.cumSum(ndarray, axis = 2))
    }

    @Test
    fun test_multiplication_of_complex_2d_ndarray() {
        val result = ComplexFloat(-2) * mk.identity(3)
        assertEquals(ComplexFloat(-2), result[0, 0])
    }
}