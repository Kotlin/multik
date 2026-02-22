package org.jetbrains.kotlinx.multik.openblas.stat

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.openblas.NativeTestBase
import org.jetbrains.kotlinx.multik.openblas.assertFloatingNumber
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeStatisticsTest : NativeTestBase() {

    // --- median ---

    @Test
    fun `median of even-sized array`() {
        val a = mk.ndarray(mk[mk[10, 7, 4], mk[3, 2, 1]])
        assertEquals(3.5, NativeStatistics.median(a))
    }

    @Test
    fun `median of odd-sized array`() {
        val a = mk.ndarray(mk[3, 1, 2, 5, 4])
        assertEquals(3.0, NativeStatistics.median(a))
    }

    @Test
    fun `median of single element`() {
        val a = mk.ndarray(mk[42])
        assertEquals(42.0, NativeStatistics.median(a))
    }

    // --- average ---

    @Test
    fun `average without weights equals mean`() {
        val a = mk.arange<Long>(1, 11, 1)
        assertEquals(NativeStatistics.mean(a), NativeStatistics.average(a))
    }

    @Test
    fun `average with weights`() {
        val a = mk.arange<Long>(1, 11, 1)
        val weights = mk.arange<Long>(10, 0, -1)
        assertEquals(4.0, NativeStatistics.average(a, weights))
    }

    @Test
    fun `average with equal weights equals mean`() {
        val a = mk.ndarray(mk[2.0, 4.0, 6.0, 8.0])
        val weights = mk.ndarray(mk[1.0, 1.0, 1.0, 1.0])
        assertFloatingNumber(NativeStatistics.mean(a), NativeStatistics.average(a, weights))
    }

    // --- mean ---

    @Test
    fun `mean of Int 1D array`() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4])
        assertEquals(2.5, NativeStatistics.mean(ndarray))
    }

    @Test
    fun `mean of Int 2D array`() {
        val ndarray = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        assertEquals(2.5, NativeStatistics.mean(ndarray))
    }

    @Test
    fun `mean of Float 2D array`() {
        val ndarray = data.getFloatM(3)
        // NativeStatistics.mean sums as Float then converts to Double, so use wider epsilon
        val expected = (0 until ndarray.size).sumOf { ndarray.data[it].toDouble() } / ndarray.size
        assertFloatingNumber(expected, NativeStatistics.mean(ndarray), epsilon = 1e-6)
    }

    @Test
    fun `mean of Double 2D array`() {
        val ndarray = data.getDoubleM(4)
        val expected = (0 until ndarray.size).sumOf { ndarray.data[it] } / ndarray.size
        assertFloatingNumber(expected, NativeStatistics.mean(ndarray))
    }
}
