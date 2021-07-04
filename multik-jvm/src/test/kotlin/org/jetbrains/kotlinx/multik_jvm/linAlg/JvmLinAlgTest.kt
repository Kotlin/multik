/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.linAlg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmLinAlgTest {

    @Test
    fun `test of norm function with p=1`() {
        val d2arrayDouble1 = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val d2arrayDouble2 = mk.ndarray(mk[mk[-1.0, -2.0], mk[-3.0, -4.0]])

        assertEquals(10.0, mk.linalg.norm(d2arrayDouble1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayDouble2, 1))

        val d2arrayShort1 = mk.ndarray(mk[mk[1.toShort(), 2.toShort()], mk[3.toShort(), 4.toShort()]])
        val d2arrayShort2 = mk.ndarray(mk[mk[(-1).toShort(), (-2).toShort()], mk[(-3).toShort(), (-4).toShort()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayShort1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayShort2, 1))

        val d2arrayInt1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val d2arrayInt2 = mk.ndarray(mk[mk[-1, -2], mk[-3, -4]])

        assertEquals(10.0, mk.linalg.norm(d2arrayInt1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayInt2, 1))

        val d2arrayByte1 = mk.ndarray(mk[mk[1.toByte(), 2.toByte()], mk[3.toByte(), 4.toByte()]])
        val d2arrayByte2 = mk.ndarray(mk[mk[(-1).toByte(), (-2).toByte()], mk[(-3).toByte(), (-4).toByte()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayByte1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayByte2, 1))

        val d2arrayFloat1 = mk.ndarray(mk[mk[1.0.toFloat(), 2.0.toFloat()], mk[3.0.toFloat(), 4.0.toFloat()]])
        val d2arrayFloat2 = mk.ndarray(mk[mk[(-1.0).toFloat(), (-2.0).toFloat()], mk[(-3.0).toFloat(), (-4.0).toFloat()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayFloat1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayFloat2, 1))

        val d2arrayLong1 = mk.ndarray(mk[mk[1.toLong(), 2.toLong()], mk[3.toLong(), 4.toLong()]])
        val d2arrayLong2 = mk.ndarray(mk[mk[(-1).toLong(), (-2).toLong()], mk[(-3).toLong(), (-4).toLong()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayLong1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayLong2, 1))



    }

//    @Test
//    fun `test of argMax function Double`() {
//        val ndarray1 = mk.ndarray(mk[0.008830892, 0.7638366187, -0.0401326368965, -0.269757419187])
//        val ndarray2 = mk.ndarray(mk[0.0088308926050, 0.763836618743, Double.NaN, -0.2697574191872])
//
//        assertEquals(1, mk.math.argMax(ndarray1))
//        assertEquals(2, mk.math.argMax(ndarray2))
//    }
//
//    @Test
//    fun `test of argMax function with axis`() {
//        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis = mk.ndarray(mk[mk[0, 2], mk[2, 2]])
//        val expectedWith1And2Axes = mk.ndarray(mk[mk[0, 1], mk[1, 1], mk[1, 1]])
//
//        assertEquals(expectedWith0Axis, mk.math.argMaxD3(ndarray, axis = 0))
//        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 1))
//        assertEquals(expectedWith1And2Axes, mk.math.argMaxD3(ndarray, axis = 2))
//    }
//
//    @Test
//    fun `test of argMin function with axis`() {
//        val ndarray = mk.ndarray(mk[mk[mk[50, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis = mk.ndarray(mk[mk[1, 0], mk[0, 0]])
//        val expectedWith1And2Axes = mk.ndarray(mk[mk[1, 0], mk[0, 0], mk[0, 0]])
//
//        assertEquals(expectedWith0Axis, mk.math.argMinD3(ndarray, axis = 0))
//        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 1))
//        assertEquals(expectedWith1And2Axes, mk.math.argMinD3(ndarray, axis = 2))
//    }
//
//    @Test
//    fun `test of max function with axis`() {
//        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis = mk.ndarray(mk[mk[7, 9], mk[10, 11]])
//        val expectedWith1Axis = mk.ndarray(mk[mk[1, 4], mk[6, 8], mk[10, 11]])
//        val expectedWith2Axis = mk.ndarray(mk[mk[3, 4], mk[5, 8], mk[9, 11]])
//
//        assertEquals(expectedWith0Axis, mk.math.max(ndarray, axis = 0))
//        assertEquals(expectedWith1Axis, mk.math.max(ndarray, axis = 1))
//        assertEquals(expectedWith2Axis, mk.math.max(ndarray, axis = 2))
//    }
//
//    @Test
//    fun `test of min function with axis`() {
//        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
//        val expectedWith1Axis = mk.ndarray(mk[mk[0, 3], mk[2, 5], mk[7, 9]])
//        val expectedWith2Axis = mk.ndarray(mk[mk[0, 1], mk[2, 6], mk[7, 10]])
//
//        assertEquals(expectedWith0Axis, mk.math.min(ndarray, axis = 0))
//        assertEquals(expectedWith1Axis, mk.math.min(ndarray, axis = 1))
//        assertEquals(expectedWith2Axis, mk.math.min(ndarray, axis = 2))
//    }
//
//    @Test
//    fun `test of sum function with axis on flat ndarray`() {
//        val ndarray = mk.ndarray(mk[0, 3, 1, 4])
//        assertFailsWith<IllegalArgumentException> { JvmMath.sum<Int, D1, D1>(ndarray, 0) }
//    }
//
//    @Test
//    fun `test of sum function with axis on 2-d ndarray`() {
//        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
//
//        val expectedWith0Axis = mk.ndarray(mk[1, 7])
//        assertEquals(expectedWith0Axis, JvmMath.sumD2(ndarray, 0))
//
//        val expectedWith1Axis = mk.ndarray(mk[3, 5])
//        assertEquals(expectedWith1Axis, JvmMath.sumD2(ndarray, 1))
//    }
//
//    @Test
//    fun `test of sum function with axis on 3-d ndarray`() {
//        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis = mk.ndarray(mk[mk[9, 17], mk[17, 23]])
//        val expectedWith1Axis = mk.ndarray(mk[mk[1, 7], mk[8, 13], mk[17, 20]])
//        val expectedWith2Axis = mk.ndarray(mk[mk[3, 5], mk[7, 14], mk[16, 21]])
//
//        assertEquals(expectedWith0Axis, mk.math.sumD3(ndarray, axis = 0))
//        assertEquals(expectedWith1Axis, mk.math.sumD3(ndarray, axis = 1))
//        assertEquals(expectedWith2Axis, mk.math.sumD3(ndarray, axis = 2))
//    }
//
//    @Test
//    fun `test of sum function with third axis on 2-d ndarray`() {
//        val ndarray = mk.ndarray(mk[mk[0, 3], mk[1, 4]])
//        assertFailsWith<IllegalArgumentException> { JvmMath.sumD2(ndarray, 2) }
//    }
//
//    @Test
//    fun `test of cumSum function with axis on 3-d ndarray`() {
//        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])
//
//        val expectedWith0Axis =
//            mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 8], mk[7, 12]], mk[mk[9, 17], mk[17, 23]]])
//        val expectedWith1Axis =
//            mk.ndarray(mk[mk[mk[0, 3], mk[1, 7]], mk[mk[2, 5], mk[8, 13]], mk[mk[7, 9], mk[17, 20]]])
//        val expectedWith2Axis =
//            mk.ndarray(mk[mk[mk[0, 3], mk[1, 5]], mk[mk[2, 7], mk[6, 14]], mk[mk[7, 16], mk[10, 21]]])
//
//
//        assertEquals(expectedWith0Axis, mk.math.cumSum(ndarray, axis = 0))
//        assertEquals(expectedWith1Axis, mk.math.cumSum(ndarray, axis = 1))
//        assertEquals(expectedWith2Axis, mk.math.cumSum(ndarray, axis = 2))
//    }
}