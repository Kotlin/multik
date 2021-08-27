package org.jetbrains.kotlinx.multik.cuda.linalg

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.cuda.CudaEngine
import org.jetbrains.kotlinx.multik.cuda.assertFloatingNDArray
import org.jetbrains.kotlinx.multik.cuda.assertFloatingNumber
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.rangeTo
import org.junit.BeforeClass
import org.slf4j.simple.SimpleLogger
import kotlin.test.Test

class CudaLinAlgTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun before() {
            System.setProperty(SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "TRACE")
        }
    }

    @Test
    fun `slice caching test`() {
        val mat1 = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
                    mk[7.0, 8.0, 9.0],
            ]
        )

        val slice1 = mat1[0..2]
        val slice2 = mat1[1..3].transpose()

        CudaEngine.runWithCuda {
            repeat(50) {
                CudaLinAlg.dot(slice1, slice2)
            }
        }
    }

    @Test
    fun `vector-vector dot inconsistent D`() {
        val vec1 = mk.ndarray(mk[1.0, 2.0, 3.0])[0..3..2]
        val vec2 = mk.ndarray(mk[4.0, 5.0, 6.0, 7.0, 8.0])[1..5..3]

        val expected = 29.0

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(vec1, vec2)

            assertFloatingNumber(expected, actual)
        }
    }

    @Test
    fun `matrix-vector dot transposed consistent D`() {
        val mat1 = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
            ]
        ).transpose() // transposed consistent

        val vec1 = mk.ndarray(mk[1.0, 2.0, 3.0])[0..3..2] //inconsistent

        val expected = mk.ndarray(mk[13.0, 17.0, 21.0])

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(mat1, vec1)

            assertFloatingNDArray(expected, actual)
        }
    }

    @Test
    fun `matrix-matrix dot transposed consistent D`() {
        val mat1 = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
                    mk[7.0, 8.0, 9.0],
                    mk[10.0, 11.0, 12.0]
            ]
        )

        val mat2 = mat1.transpose()

        val expected1 = mk.ndarray(
            mk[
                    mk[14.0, 32.0, 50.0, 68.0],
                    mk[32.0, 77.0, 122.0, 167.0],
                    mk[50.0, 122.0, 194.0, 266.0],
                    mk[68.0, 167.0, 266.0, 365.0]
            ]

        )

        val expected2 = mk.ndarray(
            mk[
                    mk[166.0, 188.0, 210.0],
                    mk[188.0, 214.0, 240.0],
                    mk[210.0, 240.0, 270.0]
            ]

        )

        CudaEngine.runWithCuda {
            val actual1 = CudaLinAlg.dot(mat1, mat2)
            assertFloatingNDArray(expected1, actual1)

            val actual2 = CudaLinAlg.dot(mat2, mat1)
            assertFloatingNDArray(expected2, actual2)
        }
    }

    @Test
    fun `matrix-matrix dot inconsistent D`() {
        val matrix = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
                    mk[7.0, 8.0, 9.0],
            ]
        )


        val mat1 = matrix[0..matrix.shape[0]..2]
        val mat2 = matrix[0..matrix.shape[0], 0..matrix.shape[1]..2]

        val expected = mk.ndarray(
            mk[
                    mk[30.0, 42.0],
                    mk[102.0, 150.0]]
        )

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(mat1, mat2)

            assertFloatingNDArray(expected, actual)
        }
    }


    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[
                    mk[1.0719, 0.6181, 0.4608, 0.4811],
                    mk[0.8211, 0.7162, 0.79  , 0.8199],
                    mk[0.5288, 0.4834, 0.5342, 0.5082],
                    mk[1.0353, 0.758 , 0.7114, 0.6647]]
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

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(matrix1, matrix2)

            assertFloatingNDArray(expected, actual)
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

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(matrix1, matrix2)

            assertFloatingNDArray(expected, actual)
        }
    }

    @Test
    fun `matrix-vector dot test D`() {
        val expected = mk.ndarray(mk[0.8006, 0.663, 0.5771])

        val matrix = mk.ndarray(
            mk[mk[0.22, 0.9, 0.27],
                    mk[0.97, 0.18, 0.59],
                    mk[0.29, 0.13, 0.59]]
        )
        val vector = mk.ndarray(mk[0.08, 0.63, 0.8])

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(matrix, vector)

            assertFloatingNDArray(expected, actual)
        }
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.8006f, 0.663f])

        val matrix = mk.ndarray(
            mk[
                    mk[0.22f, 0.9f, 0.27f],
                    mk[0.97f, 0.18f, 0.59f]]
        )
        val vector = mk.ndarray(mk[0.08f, 0.63f, 0.8f])

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(matrix, vector)

            assertFloatingNDArray(expected, actual)
        }
    }

    private val EPSILON = 1e-6

    @Test
    fun `vector-vector dot test F`() {
        val expected = 1.5325660405728998f

        val v1 = mk.ndarray(mk[0.875102f, 0.72775528f, 0.82218271f])
        val v2 = mk.ndarray(mk[0.49664201f, 0.98519225f, 0.46336994f])

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(v1, v2)
            
            assertFloatingNumber(expected, actual)
        }
    }

    @Test
    fun `vector-vector dot test D`() {
        val expected = 1.5325660405728998

        val v1 = mk.ndarray(mk[0.875102, 0.72775528, 0.82218271])
        val v2 = mk.ndarray(mk[0.49664201, 0.98519225, 0.46336994])

        CudaEngine.runWithCuda {
            val actual = CudaLinAlg.dot(v1, v2)

            assertFloatingNumber(expected, actual)
        }
    }

    @Test
    fun `memory management vector test`() {
        val vec1 = mk.ndarray(mk[1.0, 2.0, 3.0])
        val vec2 = mk.ndarray(mk[4.0, 5.0, 6.0])
        val vec3 = mk.ndarray(mk[7.0, 8.0, 9.0])

        val expected1 = 32.0
        val expected2 = 50.0
        val expected3 = 14.0

        CudaEngine.runWithCuda {
            val actual1 = CudaLinAlg.dot(vec1, vec2)
            val actual2 = CudaLinAlg.dot(vec1, vec3)
            val actual3 = CudaLinAlg.dot(vec1, vec1)

            assertFloatingNumber(expected1, actual1)
            assertFloatingNumber(expected2, actual2)
            assertFloatingNumber(expected3, actual3)
        }
    }

    @Test
    fun `memory management matrix test`() {
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
                    mk[272f, 332f],
                    mk[396f, 483f],
                    mk[520f, 634f],
                    mk[644f, 785f]
            ]
        )

        CudaEngine.runWithCuda {
            val matrix3 = CudaLinAlg.dot(matrix1, matrix2)
            val actual = CudaLinAlg.dot(matrix3, matrix2.transpose())

            assertFloatingNDArray(expected, actual)
        }
    }
}