package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jni.Loader
import org.jetbrains.kotlinx.multik.jni.NativeLinAlg
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeLinAlgTest {

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()
    }

    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[mk[1.0718999999999999, 0.6181, 0.46080000000000004, 0.48109999999999997],
                    mk[0.8210999999999999, 0.7162, 0.79, 0.8199],
                    mk[0.5287999999999999, 0.48339999999999994, 0.5342, 0.5082],
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
        assertEquals(expected, actual)
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val expected = mk.ndarray(
            mk[mk[1.0719f, 0.6181f, 0.4608f, 0.4811f],
                    mk[0.8211f, 0.7162f, 0.79f, 0.8199f],
                    mk[0.52879995f, 0.48339996f, 0.5342f, 0.5082f],
                    mk[1.0353f, 0.758f, 0.71140003f, 0.6647f]]
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
        assertEquals(expected, actual)
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
        assertEquals(expected, actual)
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
        assertEquals(expected, actual)
    }
}