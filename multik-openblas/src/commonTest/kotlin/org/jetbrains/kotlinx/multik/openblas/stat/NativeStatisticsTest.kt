package org.jetbrains.kotlinx.multik.openblas.stat

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.openblas.libLoader
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeStatisticsTest {

    @BeforeTest
    fun load() {
        libLoader("multik_jni").manualLoad()
    }

    @Test
    fun test_median() {
        val a = mk.ndarray(mk[mk[10, 7, 4], mk[3, 2, 1]])
        assertEquals(3.5, mk.stat.median(a))
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
    fun test_of_mean_function_on_a_flat_ndarray() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4])
        assertEquals(2.5, NativeStatistics.mean(ndarray))
    }

    @Test
    fun test_of_mean_function_on_a_2d_ndarray() {
        val ndarray = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        assertEquals(2.5, NativeStatistics.mean(ndarray))
    }
}