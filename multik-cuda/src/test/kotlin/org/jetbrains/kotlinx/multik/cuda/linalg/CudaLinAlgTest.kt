package org.jetbrains.kotlinx.multik.cuda.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.cuda.CudaEngine
import org.jetbrains.kotlinx.multik.cuda.CudaLinAlg
import org.jetbrains.kotlinx.multik.cuda.roundDouble
import org.jetbrains.kotlinx.multik.cuda.roundFloat
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CudaLinAlgTest {
    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[mk[1.07, 0.62, 0.46, 0.48],
                    mk[0.82, 0.72, 0.79, 0.82],
                    mk[0.53, 0.48, 0.53, 0.51],
                    mk[1.04, 0.76, 0.71, 0.66]]
        )
        val matrix1 = mk.ndarray(
            mk[mk[0.22, 0.9, 0.27],
                    mk[0.97, 0.18, 0.59],
                    mk[0.29, 0.13, 0.59],
                    mk[0.08, 0.63, 0.8]]
        )
        val matrix2 = mk.ndarray(
            mk[mk[0.36, 0.31, 0.36, 0.44],
                    mk[0.95, 0.44, 0.22, 0.25],
                    mk[0.51, 0.57, 0.68, 0.59]]
        )

        CudaEngine.use {
            val actual = CudaLinAlg.dot(matrix1, matrix2)

            assertEquals(expected, roundDouble(actual))
        }
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val matrix1 = mk.ndarray(
            mk[
                    mk[0f, 4f],
                    mk[1f, 5f],
                    mk[2f, 6f],
                    mk[3f, 7f]]
        )

        val matrix2 = mk.ndarray(
            mk[
                    mk[2f, 4f, 6f],
                    mk[3f, 5f, 7f]]
        )

        val expected = mk.ndarray(
            mk[
                    mk[12f, 20f, 28f],
                    mk[17f, 29f, 41f],
                    mk[22f, 38f, 54f],
                    mk[27f, 47f, 67f]]
        )

        CudaEngine.use {
            val actual = CudaLinAlg.dot(matrix1, matrix2)

            assertEquals(expected, roundFloat(actual))
        }
    }

    @Test
    fun `matrix-vector dot test D`() {
        val expected = mk.ndarray(mk[0.80, 0.66, 0.58])

        val matrix = mk.ndarray(
            mk[mk[0.22, 0.9, 0.27],
                    mk[0.97, 0.18, 0.59],
                    mk[0.29, 0.13, 0.59]]
        )
        val vector = mk.ndarray(mk[0.08, 0.63, 0.8])

        CudaEngine.use {
            val actual = CudaLinAlg.dot(matrix, vector)

            assertEquals(expected, roundDouble(actual))
        }
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.80f, 0.66f])

        val matrix = mk.ndarray(
            mk[
                    mk[0.22f, 0.9f, 0.27f],
                    mk[0.97f, 0.18f, 0.59f]]
        )
        val vector = mk.ndarray(mk[0.08f, 0.63f, 0.8f])

        CudaEngine.use {
            val actual = CudaLinAlg.dot(matrix, vector)

            assertEquals(expected, roundFloat(actual))
        }
    }

    private val EPSILON = 1e-6

    @Test
    fun `vector-vector dot test F`() {
        val expected = 1.5325660405728998f

        val v1 = mk.ndarray(mk[0.875102f, 0.72775528f, 0.82218271f])
        val v2 = mk.ndarray(mk[0.49664201f, 0.98519225f, 0.46336994f])

        CudaEngine.use {
            val actual = CudaLinAlg.dot(v1, v2)

            val diff = kotlin.math.abs(actual - expected)

            assertTrue(diff < EPSILON, "Difference between expected and actual: $diff")
        }
    }

    @Test
    fun `vector-vector dot test D`() {
        val expected = 1.5325660405728998

        val v1 = mk.ndarray(mk[0.875102, 0.72775528, 0.82218271])
        val v2 = mk.ndarray(mk[0.49664201, 0.98519225, 0.46336994])

        CudaEngine.use {
            val actual = CudaLinAlg.dot(v1, v2)

            val diff = kotlin.math.abs(actual - expected)

            assertTrue(diff < EPSILON, "Difference between expected and actual: $diff")
        }
    }

}