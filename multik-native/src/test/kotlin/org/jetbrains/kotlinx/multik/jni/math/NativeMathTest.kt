package org.jetbrains.kotlinx.multik.jni.math

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jni.Loader
import org.jetbrains.kotlinx.multik.jni.NativeMath
import org.jetbrains.kotlinx.multik.jni.assertFloatingNDArray
import org.jetbrains.kotlinx.multik.jni.assertFloatingNumber
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeMathTest {

    private val ndarray: NDArray<Float, D2> = mk.ndarray(mk[mk[5.7f, 1.08f], mk[9.12f, 3.55f]])

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()
    }

    @Test
    fun argMaxTest() = assertEquals(2, NativeMath.argMax(ndarray))

    @Test
    fun argMinTest() = assertEquals(1, NativeMath.argMin(ndarray))

    @Test
    fun expTest() {
        val expected =
            mk.ndarray(mk[mk[298.8673439626328, 2.9446796774374633], mk[9136.200570869447, 34.813315827573724]])
        assertFloatingNDArray(expected, NativeMath.exp(ndarray))
    }

    @Test
    fun logTest() {
        val expected =
            mk.ndarray(mk[mk[1.7404661413782472, 0.07696108087255739], mk[2.2104697915378937, 1.2669475900552918]])
        assertFloatingNDArray(expected, NativeMath.log(ndarray))
    }

    @Test
    fun sinTest() {
        val expected = mk.ndarray(mk[mk[-0.5506857, 0.88195783], mk[0.30008149, -0.39714812]])
        assertFloatingNDArray(expected, NativeMath.sin(ndarray))
    }

    @Test
    fun cosTest() {
        val expected = mk.ndarray(mk[mk[0.83471268, 0.47132833], mk[-0.95391357, -0.91775452]])
        assertFloatingNDArray(expected, NativeMath.cos(ndarray))
    }

    @Test
    fun maxTest() = assertEquals(9.12f, NativeMath.max(ndarray))

    @Test
    fun minTest() = assertEquals(1.08f, NativeMath.min(ndarray))

    @Test
    fun sumTest() = assertFloatingNumber(19.449999f, NativeMath.sum(ndarray))
}