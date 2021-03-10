package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.jni.Loader
import org.jetbrains.kotlinx.multik.jni.NativeLinAlg
import org.jetbrains.kotlinx.multik.jni.roundDouble
import org.jetbrains.kotlinx.multik.jni.roundFloat
import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeLinAlgTest {

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()
    }

    @Test
    fun `solve linear system F`() {
        val expected = mk.ndarray(
            mk[mk[-0.8f, -0.39f, 0.96f],
                    mk[-0.7f, -0.55f, 0.22f],
                    mk[0.59f, 0.84f, 1.9f],
                    mk[1.32f, -0.10f, 5.36f],
                    mk[0.57f, 0.11f, 4.04f]]
        )

        val a = mk.ndarray(
            mk[mk[6.80f, -6.05f, -0.45f, 8.32f, -9.67f],
                    mk[-2.11f, -3.30f, 2.58f, 2.71f, -5.14f],
                    mk[5.66f, 5.36f, -2.70f, 4.35f, -7.26f],
                    mk[5.97f, -4.44f, 0.27f, -7.17f, 6.08f],
                    mk[8.23f, 1.08f, 9.04f, 2.14f, -6.87f]]
        )

        val b = mk.ndarray(
            mk[mk[4.02f, -1.56f, 9.81f],
                    mk[6.19f, 4.00f, -4.09f],
                    mk[-8.22f, -8.67f, -4.57f],
                    mk[-7.57f, 1.75f, -8.61f],
                    mk[-3.03f, 2.86f, 8.99f]]
        )

        assertEquals(expected, roundFloat(NativeLinAlg.solve(a, b)))
    }

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

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertEquals(expected, roundDouble(actual))
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val expected = mk.ndarray(
            mk[mk[1.07f, 0.62f, 0.46f, 0.48f],
                    mk[0.82f, 0.72f, 0.79f, 0.82f],
                    mk[0.53f, 0.48f, 0.53f, 0.51f],
                    mk[1.04f, 0.76f, 0.71f, 0.66f]]
        )
        val matrix1 = mk.ndarray(
            mk[mk[0.22f, 0.9f, 0.27f],
                    mk[0.97f, 0.18f, 0.59f],
                    mk[0.29f, 0.13f, 0.59f],
                    mk[0.08f, 0.63f, 0.8f]]
        )
        val matrix2 = mk.ndarray(
            mk[mk[0.36f, 0.31f, 0.36f, 0.44f],
                    mk[0.95f, 0.44f, 0.22f, 0.25f],
                    mk[0.51f, 0.57f, 0.68f, 0.59f]]
        )

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertEquals(expected, roundFloat(actual))
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

        val actual = NativeLinAlg.dot(matrix, vector)
        assertEquals(expected, roundDouble(actual))
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.80f, 0.66f, 0.58f])

        val matrix = mk.ndarray(
            mk[mk[0.22f, 0.9f, 0.27f],
                    mk[0.97f, 0.18f, 0.59f],
                    mk[0.29f, 0.13f, 0.59f]]
        )
        val vector = mk.ndarray(mk[0.08f, 0.63f, 0.8f])

        val actual = NativeLinAlg.dot(matrix, vector)
        assertEquals(expected, roundFloat(actual))
    }

    @Test
    fun `vector-vector dot test F`() {
        val vector1 = mk.ndarrayOf(0.22f, 0.9f, 0.27f, 0.97f, 0.18f, 0.59f, 0.29f, 0.13f, 0.59f)
        val vector2 = mk.ndarrayOf(0.36f, 0.31f, 0.36f, 0.44f, 0.95f, 0.44f, 0.22f, 0.25f, 0.51f)

        val actual =
            BigDecimal(NativeLinAlg.dot(vector1, vector2).toDouble()).setScale(2, RoundingMode.HALF_UP).toFloat()
        assertEquals(1.71f, actual)
    }

    @Test
    fun `vector-vector dot test D`() {
        val vector1 = mk.ndarrayOf(0.22, 0.9, 0.27, 0.97, 0.18, 0.59, 0.29, 0.13, 0.59)
        val vector2 = mk.ndarrayOf(0.36, 0.31, 0.36, 0.44, 0.95, 0.44, 0.22, 0.25, 0.51)

        val actual =
            BigDecimal(NativeLinAlg.dot(vector1, vector2).toDouble()).setScale(2, RoundingMode.HALF_UP).toDouble()
        assertEquals(1.71, actual)
    }
}