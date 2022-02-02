/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.statistics

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.JvmStatistics
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmStatisticsTest {

    @Test
    fun test_median() {
        val a = mk.ndarray(mk[mk[10, 7, 4], mk[3, 2, 1]])
        println(mk.stat.median(a))
    }

    @Test
    fun test_simple_average() {
        val a = mk.arange<Long>(1, 11, 1)
        assertEquals(mk.stat.mean(a), mk.stat.average(a))
    }

    @Test
    fun test_average_with_weights() {
        val a = mk.arange<Long>(1, 11, 1)
        val weights = mk.arange<Long>(10, 0, -1)
        assertEquals(4.0, mk.stat.average(a, weights))
    }

    @Test
    fun test_of_mean_function_on_a_3d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[3.0, 5.666666666666667], mk[5.666666666666667, 7.666666666666667]])
        val expectedWith1Axis = mk.ndarray(mk[mk[0.5, 3.5], mk[4.0, 6.5], mk[8.5, 10.0]])
        val expectedWith2Axis = mk.ndarray(mk[mk[1.5, 2.5], mk[3.5, 7.0], mk[8.0, 10.5]])

        assertEquals(expectedWith0Axis, mk.stat.mean(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.stat.mean(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.stat.mean(ndarray, axis = 2))
    }

    @Test
    fun test_of_mean_function_on_a_flat_ndarray() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4])
        assertEquals(2.5, JvmStatistics.mean(ndarray))
    }

    @Test
    fun test_of_mean_function_on_a_2d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        assertEquals(2.5, JvmStatistics.mean(ndarray))
    }
}