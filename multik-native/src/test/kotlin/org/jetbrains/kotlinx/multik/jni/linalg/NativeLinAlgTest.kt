package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.jni.Loader
import org.jetbrains.kotlinx.multik.jni.NativeLinAlg
import org.jetbrains.kotlinx.multik.jni.assertFloatingNDArray
import org.jetbrains.kotlinx.multik.jni.assertFloatingNumber
import kotlin.test.BeforeTest
import kotlin.test.Test

class NativeLinAlgTest {

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()
    }

    @Test
    fun `solve linear system F`() {
        val expected = mk.ndarray(
            mk[mk[-0.800714f, -0.3896214f, 0.95546514f],
                mk[-0.6952434f, -0.55442715f, 0.22065955f],
                mk[0.5939149f, 0.84222734f, 1.9006364f],
                mk[1.3217257f, -0.10380188f, 5.3576617f],
                mk[0.5657562f, 0.10571092f, 4.0406027f]]
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

        assertFloatingNDArray(expected, NativeLinAlg.solve(a, b))
    }

    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[mk[1.0719, 0.6181, 0.4608, 0.4811],
                mk[0.8211, 0.7162, 0.79, 0.8199],
                mk[0.5288, 0.4834, 0.5342, 0.5082],
                mk[1.0353, 0.758, 0.7114, 0.6647]]
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
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val expected = mk.ndarray(
            mk[mk[1.0719f, 0.6181f, 0.4608f, 0.4811f],
                mk[0.8211f, 0.7162f, 0.79f, 0.8199f],
                mk[0.5288f, 0.4834f, 0.5342f, 0.5082f],
                mk[1.0353f, 0.758f, 0.7114f, 0.6647f]]
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
        assertFloatingNDArray(expected, actual)
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

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.8006f, 0.663f, 0.5771f])

        val matrix = mk.ndarray(
            mk[mk[0.22f, 0.9f, 0.27f],
                mk[0.97f, 0.18f, 0.59f],
                mk[0.29f, 0.13f, 0.59f]]
        )
        val vector = mk.ndarray(mk[0.08f, 0.63f, 0.8f])

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `vector-vector dot test F`() {
        val vector1 = mk.ndarrayOf(0.22f, 0.9f, 0.27f, 0.97f, 0.18f, 0.59f, 0.29f, 0.13f, 0.59f)
        val vector2 = mk.ndarrayOf(0.36f, 0.31f, 0.36f, 0.44f, 0.95f, 0.44f, 0.22f, 0.25f, 0.51f)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(1.71f, actual)
    }

    @Test
    fun `vector-vector dot test D`() {
        val vector1 = mk.ndarrayOf(0.22, 0.9, 0.27, 0.97, 0.18, 0.59, 0.29, 0.13, 0.59)
        val vector2 = mk.ndarrayOf(0.36, 0.31, 0.36, 0.44, 0.95, 0.44, 0.22, 0.25, 0.51)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(1.71, actual)
    }

    @Test
    fun `compute inverse matrix of float`() {
        val a = mk.ndarray(mk[mk[1f, 2f], mk[3f, 4f]])
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(2), NativeLinAlg.dot(a, ainv))
    }

    @Test
    fun `compute inverse matrix of double`() {
        val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(2), NativeLinAlg.dot(a, ainv))
    }
}