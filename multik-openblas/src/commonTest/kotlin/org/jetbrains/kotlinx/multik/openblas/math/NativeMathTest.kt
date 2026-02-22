package org.jetbrains.kotlinx.multik.openblas.math

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.openblas.*
import kotlin.experimental.ExperimentalNativeApi
import kotlin.math.*
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeMathTest : NativeTestBase() {

    private lateinit var floatNdarray: D2Array<Float>

    override fun extraSetUp() {
        floatNdarray = data.getFloatM(2)
    }

    // --- argMax / argMin ---

    @Test
    fun `argMax of Float 2D array`() {
        assertEquals(3, NativeMath.argMax(floatNdarray))
    }

    @Test
    fun `argMin of Float 2D array`() {
        assertEquals(0, NativeMath.argMin(floatNdarray))
    }

    @Test
    @OptIn(ExperimentalNativeApi::class)
    fun `argMax of Double 2D array`() {
        val ndarray = data.getDoubleM(3)
        val maxIdx = NativeMath.argMax(ndarray)
        // Verify the element at maxIdx is indeed the largest
        val maxVal = ndarray.data[maxIdx]
        for (i in 0 until ndarray.size) {
            assert(ndarray.data[i] <= maxVal) { "Element at $i (${ndarray.data[i]}) > element at $maxIdx ($maxVal)" }
        }
    }

    @Test
    @OptIn(ExperimentalNativeApi::class)
    fun `argMin of Double 2D array`() {
        val ndarray = data.getDoubleM(3)
        val minIdx = NativeMath.argMin(ndarray)
        val minVal = ndarray.data[minIdx]
        for (i in 0 until ndarray.size) {
            assert(ndarray.data[i] >= minVal) { "Element at $i (${ndarray.data[i]}) < element at $minIdx ($minVal)" }
        }
    }

    // --- max / min / sum ---

    @Test
    fun `max of Float 2D array`() {
        assertEquals(0.97374254f, NativeMath.max(floatNdarray))
    }

    @Test
    fun `min of Float 2D array`() {
        assertEquals(0.22631526f, NativeMath.min(floatNdarray))
    }

    @Test
    fun `sum of Float 2D array`() {
        assertFloatingNumber(2.5102746f, NativeMath.sum(floatNdarray))
    }

    @Test
    @OptIn(ExperimentalNativeApi::class)
    fun `max of Double 2D array`() {
        val ndarray = data.getDoubleM(3)
        val maxVal = NativeMath.max(ndarray)
        for (i in 0 until ndarray.size) {
            assert(ndarray.data[i] <= maxVal) { "Element at $i (${ndarray.data[i]}) > max ($maxVal)" }
        }
    }

    @Test
    @OptIn(ExperimentalNativeApi::class)
    fun `min of Double 2D array`() {
        val ndarray = data.getDoubleM(3)
        val minVal = NativeMath.min(ndarray)
        for (i in 0 until ndarray.size) {
            assert(ndarray.data[i] >= minVal) { "Element at $i (${ndarray.data[i]}) < min ($minVal)" }
        }
    }

    @Test
    fun `sum of Double 2D array`() {
        val ndarray = data.getDoubleM(2)
        val expected = (0 until ndarray.size).sumOf { ndarray.data[it] }
        assertFloatingNumber(expected, NativeMath.sum(ndarray))
    }

    // --- cumSum ---

    @Test
    fun `cumSum of Int 1D array`() {
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
        val result = NativeMath.cumSum(a)
        assertEquals(mk.ndarray(mk[1, 3, 6, 10, 15]), result)
    }

    @Test
    fun `cumSum of Double 2D array`() {
        val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val result = NativeMath.cumSum(a)
        assertFloatingNDArray(mk.ndarray(mk[1.0, 3.0, 6.0, 10.0]), result)
    }

    // --- exp ---

    @Test
    fun `exp promotes to Double`() {
        // Expected: exp applied element-wise to floatNdarray, promoted to Double
        val expected = mk.ndarray(
            mk[mk[1.2539709297612778, 1.499692498450485],
                mk[2.47182503414346, 2.647835581718662]]
        )
        assertFloatingNDArray(expected, NativeMathEx.exp(floatNdarray))
    }

    @Test
    fun `expF preserves Float`() {
        val result = NativeMathEx.expF(floatNdarray)
        // Verify against manual exp of known values
        for (i in 0 until floatNdarray.size) {
            assertFloatingNumber(exp(floatNdarray.data[i]), result.data[i], epsilon = 1e-5f)
        }
    }

    @Test
    fun `expCF with ComplexFloat`() {
        // exp(a+bi) = e^a * (cos(b) + i*sin(b))
        val a = mk.d2array(2, 2) { ComplexFloat(0.5f * it, 0.3f * it) }
        val result = NativeMathEx.expCF(a)
        for (i in 0 until a.size) {
            val re = a.data[i].re
            val im = a.data[i].im
            val expectedRe = exp(re) * cos(im)
            val expectedIm = exp(re) * sin(im)
            assertFloatingNumber(expectedRe, result.data[i].re, epsilon = 1e-5f)
            assertFloatingNumber(expectedIm, result.data[i].im, epsilon = 1e-5f)
        }
    }

    @Test
    fun `expCD with ComplexDouble`() {
        val a = mk.d2array(2, 2) { ComplexDouble(0.5 * it, 0.3 * it) }
        val result = NativeMathEx.expCD(a)
        for (i in 0 until a.size) {
            val re = a.data[i].re
            val im = a.data[i].im
            val expectedRe = exp(re) * cos(im)
            val expectedIm = exp(re) * sin(im)
            assertFloatingNumber(expectedRe, result.data[i].re, epsilon = 1e-8)
            assertFloatingNumber(expectedIm, result.data[i].im, epsilon = 1e-8)
        }
    }

    // --- log ---

    @Test
    fun `log promotes to Double`() {
        val expected = mk.ndarray(
            mk[mk[-1.4858262962985072, -0.9032262301885379],
                mk[-0.0998681176145879, -0.026608338154056003]]
        )
        assertFloatingNDArray(expected, NativeMathEx.log(floatNdarray))
    }

    @Test
    fun `logF preserves Float`() {
        val result = NativeMathEx.logF(floatNdarray)
        for (i in 0 until floatNdarray.size) {
            assertFloatingNumber(ln(floatNdarray.data[i]), result.data[i], epsilon = 1e-5f)
        }
    }

    @Test
    fun `logCF does not crash and returns correct size`() {
        val a = mk.d1array(3) { ComplexFloat(1.0f + it, 0.5f + it) }
        val result = NativeMathEx.logCF(a)
        assertEquals(a.size, result.size)
    }

    @Test
    fun `logCD does not crash and returns correct size`() {
        val a = mk.d1array(3) { ComplexDouble(1.0 + it, 0.5 + it) }
        val result = NativeMathEx.logCD(a)
        assertEquals(a.size, result.size)
    }

    // --- sin ---

    @Test
    fun `sin promotes to Double`() {
        val expected =
            mk.ndarray(mk[mk[0.22438827641771292, 0.39425779276225403], mk[0.7863984442713585, 0.826995590204087]])
        assertFloatingNDArray(expected, NativeMathEx.sin(floatNdarray))
    }

    @Test
    fun `sinF preserves Float`() {
        val result = NativeMathEx.sinF(floatNdarray)
        for (i in 0 until floatNdarray.size) {
            assertFloatingNumber(sin(floatNdarray.data[i]), result.data[i], epsilon = 1e-5f)
        }
    }

    @Test
    fun `sinCD with ComplexDouble`() {
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        val a = mk.d1array(3) { ComplexDouble(0.5 * it, 0.3 * it) }
        val result = NativeMathEx.sinCD(a)
        for (i in 0 until a.size) {
            val re = a.data[i].re
            val im = a.data[i].im
            val expectedRe = sin(re) * cosh(im)
            val expectedIm = cos(re) * sinh(im)
            assertFloatingNumber(expectedRe, result.data[i].re, epsilon = 1e-8)
            assertFloatingNumber(expectedIm, result.data[i].im, epsilon = 1e-8)
        }
    }

    // --- cos ---

    @Test
    fun `cos promotes to Double`() {
        val expected =
            mk.ndarray(mk[mk[0.9744998211422555, 0.9189998872939189], mk[0.6177195859349023, 0.5622084077839763]])
        assertFloatingNDArray(expected, NativeMathEx.cos(floatNdarray))
    }

    @Test
    fun `cosF preserves Float`() {
        val result = NativeMathEx.cosF(floatNdarray)
        for (i in 0 until floatNdarray.size) {
            assertFloatingNumber(cos(floatNdarray.data[i]), result.data[i], epsilon = 1e-5f)
        }
    }

    @Test
    fun `cosCD does not crash and returns correct size`() {
        val a = mk.d1array(3) { ComplexDouble(0.5 * (it + 1), 0.3 * (it + 1)) }
        val result = NativeMathEx.cosCD(a)
        assertEquals(a.size, result.size)
    }
}
