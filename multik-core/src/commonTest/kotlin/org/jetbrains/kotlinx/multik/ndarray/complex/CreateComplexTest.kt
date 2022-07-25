package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.test.Test
import kotlin.test.assertEquals

class CreateComplexTest {

    @Test
    fun test_easy_complex_creation() {
        assertEquals(Complex.i(1.12f), 1.12f.i)
        assertEquals(Complex.i(3.33), 3.33.i)
        assertEquals(ComplexFloat(1, 1), 1 + 1f.i)
        assertEquals(ComplexDouble(3, 7), 3 + 7.0.i)
    }
}