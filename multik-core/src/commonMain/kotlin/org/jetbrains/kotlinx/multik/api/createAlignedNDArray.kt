package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName

/**
 * Creates [NDArray] of 2nd dims filled with values from [data].
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
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
 * Creates [NDArray] of 2nd dims filled with values from [data].
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
 */
@ExperimentalMultikApi
@JvmName("createAligned2DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<T>>, filling: Double = 0.0
): D2Array<T> = this.createAlignedNDArray(data.map { it.asList() }, filling)

/**
 * Creates [NDArray] of 3rd dims filled with values from `data`.
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
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
 * Creates [NDArray] of 3rd dims filled with values from `data`.
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
 */
@ExperimentalMultikApi
@JvmName("createAligned3DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<Array<T>>>, filling: Double = 0.0
): D3Array<T> = this.createAlignedNDArray(data.map { it2d -> it2d.map { it3d -> it3d.asList() } }, filling)

/**
 * Creates [NDArray] of 3rd dims filled with values from `data`.
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
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
 * Creates [NDArray] of 3rd dims filled with values from `data`.
 * Sequences in the batch can have a different number of elements the maximum length will be chosen for each dimension.
 * Smaller sequences will be filled with [filling] to the maximum length.
 */
@ExperimentalMultikApi
@JvmName("createAligned4DArray")
public inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: Array<Array<Array<Array<T>>>>, filling: Double = 0.0
): D4Array<T> =
    this.createAlignedNDArray(data.map { it2d -> it2d.map { it3d -> it3d.map { it4d -> it4d.asList() } } }, filling)
