package org.jetbrains.kotlinx.multik.cuda

import org.jetbrains.kotlinx.multik.api.abs
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.abs
import kotlin.test.assertTrue


fun <D : Dimension> assertFloatingNDArray(
    expected: NDArray<Float, D>, actual: NDArray<Float, D>,
    epsilon: Float = 1e-6f, message: String? = null
) {
    val diff = abs(expected.deepCopy() - actual.deepCopy())
    assertTrue("${if (message == null) "" else "$message.\n"} " +
            "Expected \n<$expected>\nActual\n<$actual>.\nDiff\n<$diff>\n") { diff.all { it < epsilon } }
}

fun <D : Dimension> assertFloatingNDArray(
    expected: NDArray<Double, D>, actual: NDArray<Double, D>,
    epsilon: Double = 1e-8, message: String? = null
) {
    val diff = abs(expected.deepCopy() - actual.deepCopy())
    assertTrue("${if (message == null) "" else "$message.\n"} " +
            "Expected \n<$expected>\nActual\n<$actual>.\nDiff\n<$diff>\n") { diff.all { it < epsilon } }
}

fun assertFloatingNumber(expected: Float, actual: Float, epsilon: Float = 1e-6f, message: String? = null) {
    val diff = abs(expected - actual)
    assertTrue("${if (message == null) "" else "$message. "} Expected <$expected>, actual <$actual>.") { diff < epsilon }
}

fun assertFloatingNumber(expected: Double, actual: Double, epsilon: Double = 1e-8, message: String? = null) {
    val diff = abs(expected - actual)
    assertTrue("${if (message == null) "" else "$message. "} Expected <$expected>, actual <$actual>.") { diff < epsilon }
}
