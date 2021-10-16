package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ComplexMultiArrayTest {

    @Test
    fun `re and im properties returns real and imaginary portion of float complex number array`() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
        val real = complex.re
        val im = complex.im
        val expectedReal = mk.d2arrayIndices(3, 3) { i, j -> i.toFloat() }
        val expectedIm = mk.d2arrayIndices(3, 3) { i, j -> j.toFloat() }

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertEquals(expectedReal, real)
        assertEquals(expectedIm, im)
    }

    @Test
    fun `re and im properties returns real and imaginary portion of double complex number array`() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, j) }
        val real = complex.re
        val im = complex.im
        val expectedReal = mk.d2arrayIndices(3, 3) { i, j -> i.toDouble() }
        val expectedIm = mk.d2arrayIndices(3, 3) { i, j -> j.toDouble() }

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertEquals(expectedReal, real)
        assertEquals(expectedIm, im)
    }

    @Test
    fun `re and im properties work for empty complex double arrays`() {
        val complex = mk.ndarray(emptyList<ComplexDouble>())
        val real = complex.re
        val im = complex.im

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertTrue(real.isEmpty())
        assertTrue(im.isEmpty())
    }

    @Test
    fun `re and im properties work for empty complex float arrays`() {
        val complex = mk.ndarray(emptyList<ComplexFloat>())
        val real = complex.re
        val im = complex.im

        assertEquals(complex.shape, real.shape)
        assertEquals(complex.shape, im.shape)
        assertTrue(real.isEmpty())
        assertTrue(im.isEmpty())
    }

    @Test
    fun `conj returns array of conjugated complex doubles`() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, j) }
        val expected = mk.d2arrayIndices(3, 3) { i, j -> ComplexDouble(i, -j) }
        val actual = complex.conj()

        assertEquals(complex.shape, actual.shape)
        assertEquals(expected, actual)
    }

    @Test
    fun `conj returns array of conjugated complex floats`() {
        val complex = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
        val expected = mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, -j) }
        val actual = complex.conj()

        assertEquals(complex.shape, actual.shape)
        assertEquals(expected, actual)
    }

    @Test
    fun `conj for empty complex float arrays`() {
        val complex = mk.ndarray(emptyList<ComplexFloat>())
        val conj = complex.conj()

        assertEquals(complex.shape, conj.shape)
        assertTrue(conj.isEmpty())
    }

    @Test
    fun `conj for empty complex double arrays`() {
        val complex = mk.ndarray(emptyList<ComplexDouble>())
        val conj = complex.conj()

        assertEquals(complex.shape, conj.shape)
        assertTrue(conj.isEmpty())
    }


}
