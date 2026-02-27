package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Creates a [D2Array] from an array of [ByteArray] rows.
 *
 * All inner arrays must have the same size.
 *
 * @param args the rows of the 2D array.
 * @return a new [D2Array] with shape `(args.size, args[0].size)`.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<ByteArray>): D2Array<Byte> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = ByteArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewByteArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/**
 * Creates a [D2Array] from an array of [ShortArray] rows.
 *
 * @param args the rows of the 2D array; all must have the same size.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<ShortArray>): D2Array<Short> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = ShortArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewShortArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/**
 * Creates a [D2Array] from an array of [IntArray] rows.
 *
 * @param args the rows of the 2D array; all must have the same size.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<IntArray>): D2Array<Int> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = IntArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewIntArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/**
 * Creates a [D2Array] from an array of [LongArray] rows.
 *
 * @param args the rows of the 2D array; all must have the same size.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<LongArray>): D2Array<Long> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = LongArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewLongArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/**
 * Creates a [D2Array] from an array of [FloatArray] rows.
 *
 * @param args the rows of the 2D array; all must have the same size.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<FloatArray>): D2Array<Float> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = FloatArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewFloatArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/**
 * Creates a [D2Array] from an array of [DoubleArray] rows.
 *
 * @param args the rows of the 2D array; all must have the same size.
 * @throws IllegalArgumentException if inner arrays have different sizes.
 */
public fun Multik.ndarray(args: Array<DoubleArray>): D2Array<Double> {
    val dim0 = args.size
    val dim1 = args[0].size
    require(args.all { dim1 == it.size }) { "Arrays must be the same size." }

    val array = DoubleArray(dim0 * dim1)
    var index = 0
    for (i in 0 until dim0) {
        for (j in 0 until dim1) {
            array[index++] = args[i][j]
        }
    }
    val data = MemoryViewDoubleArray(array)
    return D2Array(data, shape = intArrayOf(dim0, dim1), dim = D2)
}

/** Converts this array of [ByteArray] rows to a [D2Array]. */
public fun Array<ByteArray>.toNDArray(): D2Array<Byte> = Multik.ndarray(this)

/** Converts this array of [ShortArray] rows to a [D2Array]. */
public fun Array<ShortArray>.toNDArray(): D2Array<Short> = Multik.ndarray(this)

/** Converts this array of [IntArray] rows to a [D2Array]. */
public fun Array<IntArray>.toNDArray(): D2Array<Int> = Multik.ndarray(this)

/** Converts this array of [LongArray] rows to a [D2Array]. */
public fun Array<LongArray>.toNDArray(): D2Array<Long> = Multik.ndarray(this)

/** Converts this array of [FloatArray] rows to a [D2Array]. */
public fun Array<FloatArray>.toNDArray(): D2Array<Float> = Multik.ndarray(this)

/** Converts this array of [DoubleArray] rows to a [D2Array]. */
public fun Array<DoubleArray>.toNDArray(): D2Array<Double> = Multik.ndarray(this)


