package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.D4Array
import org.jetbrains.kotlinx.multik.ndarray.data.toPrimitiveType
import kotlin.jvm.JvmName

/**
 * Creates a 2D array from ragged lists by padding shorter rows to the maximum row length.
 *
 * Each inner list becomes a row. Rows shorter than the longest are right-padded with [filling].
 * The resulting shape is `(data.size, maxRowLength)`.
 *
 * ```
 * mk.createAlignedNDArray(listOf(listOf(1, 2, 3), listOf(4, 5)))
 * // [[1, 2, 3],
 * //  [4, 5, 0]]
 * ```
 *
 * @param data ragged list of rows.
 * @param filling value used to pad shorter rows, converted to [T].
 * @return a new [D2Array] with shape `(data.size, maxRowLength)`.
 * @throws IllegalArgumentException if [data] is empty.
 */
@ExperimentalMultikApi
@JvmName("createAligned2DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<T>>, filling: Double = 0.0
): D2Array<T> {
    require(data.isNotEmpty())
    val maxLength = data.maxOf { it.size }
    val paddingIdx: T = filling.toPrimitiveType()
    return mk.d2array(data.size, maxLength) { idx ->
        val sequenceIdx = idx / maxLength
        val elementIdx = idx % maxLength

        if (elementIdx < data[sequenceIdx].size)
            data[sequenceIdx][elementIdx]
        else
            paddingIdx
    }
}

/**
 * Array overload of [createAlignedNDArray] for 2D. See the `List` variant for details.
 */
@ExperimentalMultikApi
@JvmName("createAligned2DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<T>>, filling: Double = 0.0
): D2Array<T> = this.createAlignedNDArray(data.map { it.asList() }, filling)

/**
 * Creates a 3D array from ragged nested lists by padding shorter sequences to the maximum length at each depth.
 *
 * The resulting shape is `(data.size, maxDim2, maxDim3)` where each `maxDimN` is the maximum
 * sequence length at that nesting level. Missing elements are filled with [filling].
 *
 * @param data ragged nested lists with 3 levels of nesting.
 * @param filling value used to pad shorter sequences, converted to [T].
 * @return a new [D3Array] with shape `(data.size, maxDim2, maxDim3)`.
 * @throws IllegalArgumentException if [data] is empty.
 */
@ExperimentalMultikApi
@JvmName("createAligned3DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<List<T>>>, filling: Double = 0.0
): D3Array<T> {
    require(data.isNotEmpty())
    val maxLength2Dim = data.maxOf { it.size }
    val maxLength3Dim = data.maxOf { seq -> seq.maxOf { it.size } }
    val paddingIdx: T = filling.toPrimitiveType()

    return mk.d3array(data.size, maxLength2Dim, maxLength3Dim) { idx ->
        val dim1 = idx / (maxLength2Dim * maxLength3Dim)
        val dim2 = (idx / maxLength3Dim) % maxLength2Dim
        val dim3 = idx % maxLength3Dim

        if (dim2 < data[dim1].size && dim3 < data[dim1][dim2].size) {
            data[dim1][dim2][dim3]
        } else {
            paddingIdx
        }
    }
}

/**
 * Array overload of [createAlignedNDArray] for 3D. See the `List` variant for details.
 */
@ExperimentalMultikApi
@JvmName("createAligned3DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<Array<T>>>, filling: Double = 0.0
): D3Array<T> = this.createAlignedNDArray(data.map { it2d -> it2d.map { it3d -> it3d.asList() } }, filling)

/**
 * Creates a 4D array from ragged nested lists by padding shorter sequences to the maximum length at each depth.
 *
 * The resulting shape is `(data.size, maxDim2, maxDim3, maxDim4)` where each `maxDimN` is the maximum
 * sequence length at that nesting level. Missing elements are filled with [filling].
 *
 * @param data ragged nested lists with 4 levels of nesting.
 * @param filling value used to pad shorter sequences, converted to [T].
 * @return a new [D4Array] with shape `(data.size, maxDim2, maxDim3, maxDim4)`.
 * @throws IllegalArgumentException if [data] is empty.
 */
@ExperimentalMultikApi
@JvmName("createAligned4DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<List<List<T>>>>, filling: Double = 0.0
): D4Array<T> {
    require(data.isNotEmpty())
    val maxLength2Dim = data.maxOf { it2d -> it2d.size }
    val maxLength3Dim = data.maxOf { it2d -> it2d.maxOf { it3d -> it3d.size } }
    val maxLength4Dim = data.maxOf { it2d -> it2d.maxOf { it3d -> it3d.maxOf { it4d -> it4d.size } } }
    val paddingIdx: T = filling.toPrimitiveType()

    return this.d4array(data.size, maxLength2Dim, maxLength3Dim, maxLength4Dim) { idx ->
        val dim1 = idx / (maxLength2Dim * maxLength3Dim * maxLength4Dim)
        val dim2 = (idx / (maxLength3Dim * maxLength4Dim)) % maxLength2Dim
        val dim3 = (idx / maxLength4Dim) % maxLength3Dim
        val dim4 = idx % maxLength4Dim

        if (dim2 < data[dim1].size && dim3 < data[dim1][dim2].size && dim4 < data[dim1][dim2][dim3].size) {
            data[dim1][dim2][dim3][dim4]
        } else {
            paddingIdx
        }
    }
}

/**
 * Array overload of [createAlignedNDArray] for 4D. See the `List` variant for details.
 */
@ExperimentalMultikApi
@JvmName("createAligned4DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<Array<Array<T>>>>, filling: Double = 0.0
): D4Array<T> =
    this.createAlignedNDArray(data.map { it2d -> it2d.map { it3d -> it3d.map { it4d -> it4d.asList() } } }, filling)
