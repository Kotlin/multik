package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

internal fun requireSquare(shape: IntArray) {
    require(shape[0] == shape[1]) { "Square matrix expected, shape=(${shape[0]}, ${shape[1]}) given" }
}

internal fun requireDotShape(aShape: IntArray, bShape: IntArray) = require(aShape[1] == bShape[0]) {
    "Shapes mismatch: shapes " +
        "${aShape.joinToString(prefix = "(", postfix = ")")} and " +
        "${bShape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
        "${aShape[1]} (dim 1) != ${bShape[0]} (dim 0)"
}

/**
 * swap lines for plu decomposition
 * @param swap `a[i, j] = a[i + rowPerm[ i ], j].also { a[i + rowPerm[ i ], j] = a[i, j] }`
 */
internal inline fun swapLines(
    rowPerm: MultiArray<Int, D1>,
    from1: Int = 0, to1: Int = rowPerm.size, from2: Int = 0, to2: Int, swap: (Int, Int) -> Unit
) {
    for (i in from1 until to1) {
        if (rowPerm[i] != 0) {
            for (j in from2 until to2) {
                swap(i, j)
            }
        }
    }
}

