package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jni.*
import kotlin.test.BeforeTest
import kotlin.test.Test

class NativeLinAlgTest {

    private lateinit var data: DataStructure

    @BeforeTest
    fun load() {
        Loader("multik_jni").manualLoad()

        data = DataStructure(42)
    }

    @Test
    fun `solve linear system F`() {
        val expected = mk.ndarray(
            mk[mk[4.1391945f, 1.2361444f, 4.4088345f],
                mk[-3.0071893f, 0.13484901f, -3.9121897f],
                mk[3.2885208f, -0.04077824f, 4.3054614f],
                mk[0.7955365f, 0.57545465f, 0.42709854f],
                mk[-11.024394f, -1.9956491f, -11.173507f]]
        )

        val (a, b) = data.getFloatMM(5, 5, 5, 3)

        assertFloatingNDArray(expected, NativeLinAlg.solve(a, b), epsilon = 1e5f)
    }

    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[mk[1.0853811780469889, 0.6321441231331913, 0.46677507285707914, 0.4892609866360924],
                mk[0.833116624067087, 0.7287671731075991, 0.7973174517659147, 0.8294695934205714],
                mk[0.5426264067593305, 0.4939259489941979, 0.5413707808847182, 0.5183069608507607],
                mk[1.048984456958365, 0.7710348889066437, 0.7189440755132327, 0.6763964597209662]]
        )

        val (matrix1, matrix2) = data.getDoubleMM(4, 3, 3, 4)

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val expected = mk.ndarray(
            mk[mk[0.8819745f, 0.64614516f, 0.7936589f, 0.5490592f],
                mk[0.543343f, 0.8133113f, 0.2793616f, 1.0130367f],
                mk[0.98215795f, 0.90664136f, 0.3652947f, 1.1545719f],
                mk[0.79763675f, 0.43727058f, 0.60035574f, 0.36558864f]]
        )

        val (matrix1, matrix2) = data.getFloatMM(4, 3, 3, 4)

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test D`() {
        val expected = mk.ndarray(mk[0.8120680956454793, 0.676196362161166, 0.5898845530863276])

        val (matrix, vector) = data.getDoubleMV(3)

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.86327714f, 0.3244831f, 0.76492393f])

        val (matrix, vector) = data.getFloatMV(3)

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `vector-vector dot test F`() {
        val (vector1, vector2) = data.getFloatVV(9)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(2.883776f, actual)
    }

    @Test
    fun `vector-vector dot test D`() {
        val (vector1, vector2) = data.getDoubleVV(9)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(1.9696041133566367, actual)
    }

    @Test
    fun `compute inverse matrix of float`() {
        val a = data.getFloatM(2)
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(2), NativeLinAlg.dot(a, ainv))
    }

    @Test
    fun `compute inverse matrix of double`() {
        val a = data.getDoubleM(2)
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(2), NativeLinAlg.dot(a, ainv))
    }
}