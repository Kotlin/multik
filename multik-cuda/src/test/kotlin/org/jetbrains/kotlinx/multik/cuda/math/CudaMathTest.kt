package org.jetbrains.kotlinx.multik.cuda.math

import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.cuda.CudaEngine
import org.jetbrains.kotlinx.multik.cuda.CudaMath.add
import org.jetbrains.kotlinx.multik.cuda.CudaMath.subtract
import org.jetbrains.kotlinx.multik.cuda.roundDouble
import org.junit.Test
import kotlin.test.assertEquals

class CudaMathTest {
    @Test
    fun `combine test`() {
        val mat1 = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
            ]
        )

        val mat2 = mk.ndarray(
            mk[
                    mk[-1.0, -2.0, -3.0],
                    mk[100.0, 110.0, 120.0]]
        )

        val vec1 = mk.ndarray(mk[1, 2, 3]).asType<Double>()
        val vec2 = mk.ndarray(mk[4, 5, 2]).asType<Double>()

        val arrayD3_1 = mk.d3array<Double>(2, 2, 3) { it.toDouble() }
        val arrayD3_2 = mk.d3array<Double>(2, 2, 3) { 2 * it.toDouble() }
        val arrayD3_expected = mk.d3array<Double>(2, 2, 3) { 3 * it.toDouble() }

        val expected = mk.ndarray(
            mk[
                    mk[0.0, 0.0, 0.0],
                    mk[104.0, 115.0, 126.0]]
        )

        val expected2 = mk.ndarray(
            mk[
                    mk[-2.0, -4.0, -6.0],
                    mk[96.0, 105.0, 114.0]]
        )

        val vecExpected1 = mk.ndarray(mk[5.0, 7.0, 5.0])
        val vecExpected2 = mk.ndarray(mk[3.0, 3.0, -1.0])

        CudaEngine.runWithCuda {
            val res1 = mat1 add mat2

            val mat3 = mat2.transpose().deepCopy()

            val res2 = mat3.transpose() add mat1
            val res3 = mat1 add mat3.transpose()
            val res4 = (mat1.transpose() add mat2.transpose()).transpose()
            val res5 = mat2 subtract mat1

            assertEquals(expected, roundDouble(res1))
            assertEquals(expected, roundDouble(res2))
            assertEquals(expected, roundDouble(res3))
            assertEquals(expected, roundDouble(res4))
            assertEquals(expected2, roundDouble(res5))

            val vecRes1 = vec1 add vec2
            val vecRes2 = vec2 subtract vec1

            assertEquals(vecExpected1, roundDouble(vecRes1))
            assertEquals(vecExpected2, roundDouble(vecRes2))

            val arrayD3_res = arrayD3_1 add arrayD3_2
            assertEquals(arrayD3_expected, roundDouble(arrayD3_res))
        }
    }
}