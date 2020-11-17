package org.jetbrains.multik_jvm.statistics

import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.jvm.JvmStatistics
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmStatisticsTest {
    @Test
    fun `test of mean function on a 3-d ndarray`() {
        val ndarray = mk.ndarray(mk[mk[mk[0, 3], mk[1, 4]], mk[mk[2, 5], mk[6, 8]], mk[mk[7, 9], mk[10, 11]]])

        val expectedWith0Axis = mk.ndarray(mk[mk[3.0, 5.666666666666667], mk[5.666666666666667, 7.666666666666667]])
        val expectedWith1Axis = mk.ndarray(mk[mk[0.5, 3.5], mk[4.0, 6.5], mk[8.5, 10.0]])
        val expectedWith2Axis = mk.ndarray(mk[mk[1.5, 2.5], mk[3.5, 7.0], mk[8.0, 10.5]])

        assertEquals(expectedWith0Axis, mk.stat.mean(ndarray, axis = 0))
        assertEquals(expectedWith1Axis, mk.stat.mean(ndarray, axis = 1))
        assertEquals(expectedWith2Axis, mk.stat.mean(ndarray, axis = 2))
    }

    @Test
    fun `test of mean function on a flat ndarray`() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4])
        assertEquals(2.5, JvmStatistics.mean(ndarray))
    }

    @Test
    fun `test of mean function on a 2-d ndarray`() {
        val ndarray = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        assertEquals(2.5, JvmStatistics.mean(ndarray))
    }
}