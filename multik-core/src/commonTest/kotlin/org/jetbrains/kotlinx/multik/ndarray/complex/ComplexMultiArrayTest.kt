package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ComplexMultiArrayTest {

    @Test
    fun re_and_im_properties_returns_real_and_imaginary_portion_of_float_complex_number_array() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
        val real = complex.re
        val im = complex.im
        val expectedReal = mk.d2arrayIndices(3, 3) { i, _ -> i.toFloat() }
        val expectedIm = mk.d2arrayIndices(3, 3) { _, j -> j.toFloat() }

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertEquals(expectedReal, real)
        assertEquals(expectedIm, im)
    }

    @Test
    fun re_and_im_properties_returns_real_and_imaginary_portion_of_double_complex_number_array() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, j) }
        val real = complex.re
        val im = complex.im
        val expectedReal = mk.d2arrayIndices(3, 3) { i, _ -> i.toDouble() }
        val expectedIm = mk.d2arrayIndices(3, 3) { _, j -> j.toDouble() }

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertEquals(expectedReal, real)
        assertEquals(expectedIm, im)
    }

    @Test
    fun re_and_im_properties_work_for_empty_complex_double_arrays() {
        val complex = mk.ndarray(emptyList<ComplexDouble>())
        val real = complex.re
        val im = complex.im

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertTrue(real.isEmpty())
        assertTrue(im.isEmpty())
    }

    @Test
    fun re_and_im_properties_work_for_empty_complex_float_arrays() {
        val complex = mk.ndarray(emptyList<ComplexFloat>())
        val real = complex.re
        val im = complex.im

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertTrue(real.isEmpty())
        assertTrue(im.isEmpty())
    }

    @Test
    fun conj_returns_array_of_conjugated_complex_doubles() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, j) }
        val expected = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, -j) }
        val actual = complex.conj()

        assertEquals(complex.shape, actual.shape)
        assertEquals(expected, actual)
    }

    @Test
    fun conj_returns_array_of_conjugated_complex_floats() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
        val expected = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, -j) }
        val actual = complex.conj()

        assertEquals(complex.shape, actual.shape)
        assertEquals(expected, actual)
    }

    @Test
    fun conj_for_empty_complex_float_arrays() {
        val complex = mk.ndarray(emptyList<ComplexFloat>())
        val conj = complex.conj()

        assertEquals(complex.shape, conj.shape)
        assertTrue(conj.isEmpty())
    }

    @Test
    fun conj_for_empty_complex_double_arrays() {
        val complex = mk.ndarray(emptyList<ComplexDouble>())
        val conj = complex.conj()

        assertEquals(complex.shape, conj.shape)
        assertTrue(conj.isEmpty())
    }
}
