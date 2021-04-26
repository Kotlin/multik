package org.jetbrains.kotlinx.multik.jni.math

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jni.Loader
import org.jetbrains.kotlinx.multik.jni.NativeMath
import org.jetbrains.kotlinx.multik.jni.roundDouble
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeMathTest {

    val ndarray: NDArray<Float, D2> = mk.ndarray(mk[mk[5.7f, 1.08f], mk[9.12f, 3.55f]])

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()
    }

    @Test
    fun argMaxTest() {
        assertEquals(2, NativeMath.argMax(ndarray))
    }

    @Test
    fun argMinTest() {
        assertEquals(1, NativeMath.argMin(ndarray))
    }

    @Test
    fun expTest() {
        val expected = mk.ndarray(mk[mk[298.87, 2.94], mk[9136.2, 34.81]])
        assertEquals(expected, roundDouble(NativeMath.exp(ndarray)))
    }

    @Test
    fun logTest() {
        val expected = mk.ndarray(mk[mk[1.74, 0.08], mk[2.21, 1.27]])
        assertEquals(expected, roundDouble(NativeMath.log(ndarray)))
    }

    @Test
    fun sinTest() {
        val expected = mk.ndarray(mk[mk[-0.55, 0.88], mk[0.3, -0.4]])
        assertEquals(expected, roundDouble(NativeMath.sin(ndarray)))
    }

    @Test
    fun cosTest() {
        val expected = mk.ndarray(mk[mk[0.83, 0.47], mk[-0.95, -0.92]])
        assertEquals(expected, roundDouble(NativeMath.cos(ndarray)))
    }

    @Test
    fun maxTest() {
        assertEquals(9.12f, NativeMath.max(ndarray))
    }

    @Test
    fun minTest() {
        assertEquals(1.08f, NativeMath.min(ndarray))
    }

    @Test
    fun sumTest() {
        assertEquals(19.449999f, NativeMath.sum(ndarray))
    }
}