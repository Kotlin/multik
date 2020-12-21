/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.math

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.JvmMath
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class JvmMathTest {

    @Test
    fun `test of argMax function with axis`() {
        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[0, 2], mk[2, 2]])
        val expectedWith1And2Axes = mk.ndarray(mk[mk[0, 1], mk[1, 1], mk[1, 1]])

        assertEquals(expectedWith0Axis, mk.math.argMaxD3(ndarray, axis = 0))
        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 1))
        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 2))
    }

    @Test
    fun `test of argMin function with axis`() {
        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[1, 0], mk[0, 0]])
        val expectedWith1And2Axes = mk.ndarray(mk[mk[1, 0], mk[0, 0], mk[0, 0]])

        assertEquals(expectedWith0Axis, mk.math.argMinD3(ndarray, axis = 0))
        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 1))
        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 2))
    }

    @Test
    fun `test of max function with axis`() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[7, 9], mk[10, 11]])
        val expectedWith1Axis = mk.ndarray(mk[mk[1, 4], mk[6, 8], mk[10, 11]])
        val expectedWith2Axis = mk.ndarray(mk[mk[3, 4], mk[5, 8], mk[9, 11]])

        assertEquals(expectedWith0Axis, mk.math.max(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.max(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.max(ndarray, axis = 2))
    }

    @Test
    fun `test of min function with axis`() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
        val expectedWith1Axis = mk.ndarray(mk[mk[0, 3], mk[2, 5], mk[7, 9]])
        val expectedWith2Axis = mk.ndarray(mk[mk[0, 1], mk[2, 6], mk[7, 10]])

        assertEquals(expectedWith0Axis, mk.math.min(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.min(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.min(ndarray, axis = 2))
    }

    @Test
    fun `test of sum function with axis on flat ndarray`() {
        val ndarray = mk.ndarray(mk[0, 3, 1, 4])
        assertFailsWith<IllegalArgumentException> { JvmMath.sum<Int, D1, D1>(ndarray, 0) }
    }

    @Test
    fun `test of sum function with axis on 2-d ndarray`() {
        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])

        val expectedWith0Axis = mk.ndarray(mk[1, 7])
        assertEquals(expectedWith0Axis, JvmMath.sumD2(ndarray, 0))

        val expectedWith1Axis = mk.ndarray(mk[3, 5])
        assertEquals(expectedWith1Axis, JvmMath.sumD2(ndarray, 1))
    }

    @Test
    fun `test of sum function with axis on 3-d ndarray`() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[9, 17], mk[17, 23]])
        val expectedWith1Axis = mk.ndarray(mk[mk[1, 7], mk[8, 13], mk[17, 20]])
        val expectedWith2Axis = mk.ndarray(mk[mk[3, 5], mk[7, 14], mk[16, 21]])

        assertEquals(expectedWith0Axis, mk.math.sumD3(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.math.sumD3(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.math.sumD3(ndarray, axis = 2))
    }

    @Test
    fun `test of sum function with third axis on 2-d ndarray`() {
        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
        assertFailsWith<IllegalArgumentException> { JvmMath.sumD2(ndarray, 2) }
    }

    @Test
    fun `test of cumSum function with axis on 3-d ndarray`() {
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
}