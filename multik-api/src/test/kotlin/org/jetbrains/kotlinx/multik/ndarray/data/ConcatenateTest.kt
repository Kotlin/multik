package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.Slice
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.asSequence
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class ConcatenateTest {
    @Test
    fun `concatenate should concatenate for simple array`() {

        val arr1 = mk.zeros<Double>(1) + 1.0
        val arr2 = mk.zeros<Double>(2) + 2.0
        val arr3 = mk.zeros<Double>(3) + 3.0
        val arr4 = mk.zeros<Double>(4) + 4.0

        val result = arr1.catFix(listOf(arr2, arr3, arr4), 0)

        Assertions.assertArrayEquals(
            doubleArrayOf(
                1.0,
                2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 4.0
            ), result.data.getDoubleArray()
        )

    }

    @Test
    fun `concatenate should concatenate for non consistent array`() {

        val arr1 = mk.zeros<Double>(1) + 1.0
        val arr2 = mk.zeros<Double>(2) + 2.0
        val arr3 = mk.zeros<Double>(3) + 3.0
        val arr4 = mk.zeros<Double>(10) + 4.0
        val arr5 = arr4[Slice(2, 6, 1)]

        Assertions.assertFalse(arr5.consistent)
        val result = arr1.catFix(listOf(arr2, arr3, arr5), 0)

        Assertions.assertArrayEquals(
            doubleArrayOf(
                1.0,
                2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 4.0
            ), result.data.getDoubleArray()
        )

    }

    @Test
    fun `concatenate should concatenate for complex`() {

        val arr1 = mk.zeros<ComplexDouble>(1) + ComplexDouble(1.0, 0.0)
        val arr2 = mk.zeros<ComplexDouble>(2) + ComplexDouble(2.0, 0.0)
        val arr3 = mk.zeros<ComplexDouble>(3) + ComplexDouble(3.0, 0.0)
        val arr4 = mk.zeros<ComplexDouble>(4) + ComplexDouble(4.0, 0.0)


        val result = arr1.catFix(listOf(arr2, arr3, arr4), 0)

        var realResult = result.asSequence().map { it.re }.toList().toDoubleArray()

        Assertions.assertArrayEquals(
            doubleArrayOf(
                1.0,
                2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 4.0
            ),
            realResult
        )

    }


}
