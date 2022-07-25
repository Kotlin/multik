/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName
import kotlin.random.Random

/**
 * Returns a vector of the specified size filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number> Multik.rand(dim0: Int): D1Array<T> {
    require(dim0 > 0) { "Dimension must be positive." }
    val dtype = DataType.ofKClass(T::class)
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(dim0, dtype) { frand() }
    return D1Array(data, shape = intArrayOf(dim0), dim = D1)
}

/**
 * Returns a matrix of the specified shape filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number> Multik.rand(dim0: Int, dim1: Int): D2Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(dim0, dim1)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(dim0 * dim1, dtype) { frand() }
    return D2Array(data, shape = shape, dim = D2)
}

/**
 * Returns an NDArray of the specified shape filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number> Multik.rand(dim0: Int, dim1: Int, dim2: Int): D3Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(dim0, dim1, dim2)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(dim0 * dim1 * dim2, dtype) { frand() }
    return D3Array(data, shape = shape, dim = D3)
}

/**
 * Returns an NDArray of the specified shape filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number> Multik.rand(dim0: Int, dim1: Int, dim2: Int, dim3: Int): D4Array<T> {
    val dtype = DataType.ofKClass(T::class)
    val shape = intArrayOf(dim0, dim1, dim2, dim3)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(dim0 * dim1 * dim2 * dim3, dtype) { frand() }
    return D4Array(data, shape = shape, dim = D4)
}

/**
 * Returns an NDArray of the specified shape filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number> Multik.rand(
    dim0: Int, dim1: Int, dim2: Int, dim3: Int, vararg dims: Int
): NDArray<T, DN> {
    return rand(intArrayOf(dim0, dim1, dim2, dim3, *dims))
}

/**
 * Returns an NDArray of the specified shape filled with random numbers uniformly distributed for:
 * Int - [Int.MIN_VALUE, Int.MAX_VALUE)
 * Long - [Long.MIN_VALUE, Long.MAX_VALUE)
 * Float - [0f, 1f)
 * Double - [0.0, 1.0)
 */
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(shape: IntArray): NDArray<T, D> {
    val dtype = DataType.ofKClass(T::class)
    val dim = dimensionClassOf<D>(shape.size)
    requireDimension(dim, shape.size)
    for (i in shape.indices) {
        require(shape[i] > 0) { "Dimension $i must be positive." }
    }
    val size = shape.fold(1, Int::times)
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(size, dtype) { frand() }
    return NDArray(data, shape = shape, dim = dim)
}


/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 */
@JvmName("randWithVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    from: T, until: T, vararg dims: Int
): NDArray<T, D> =
    Multik.rand(from, until, dims)

/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 *
 * Note: Float generation is inefficient.
 */
@JvmName("randWithShape")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    from: T, until: T, dims: IntArray
): NDArray<T, D> {
    val dtype = DataType.ofKClass(T::class)
    val dim = dimensionClassOf<D>(dims.size)
    requireDimension(dim, dims.size)
    for (i in dims.indices) {
        require(dims[i] > 0) { "Dimension $i must be positive." }
    }
    val size = dims.fold(1, Int::times)
    val data = randData(from, until, size, dtype)
    return NDArray(data, shape = dims, dim = dim)
}

/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 * with the specified [seed].
 *
 * Note: Float generation is inefficient.
 */
@JvmName("randSeedVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, vararg dims: Int
): NDArray<T, D> = Multik.rand(Random(seed), from, until, dims)

/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 * with the specified [seed].
 *
 * Note: Float generation is inefficient.
 */
@JvmName("randSeedShape")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, dims: IntArray
): NDArray<T, D> = Multik.rand(Random(seed), from, until, dims)

/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 * with the specified [gen].
 *
 * Note: Float generation is inefficient.
 */
@JvmName("randGenVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    gen: Random, from: T, until: T, vararg dims: Int
): NDArray<T, D> = Multik.rand(gen, from, until, dims)

/**
 * Returns an NDArray of the specified shape filled with number uniformly distributed between [[from], [until])
 * with the specified [gen].
 *
 * Note: Float generation is inefficient.
 */
@JvmName("randGenShape")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    gen: Random, from: T, until: T, dims: IntArray
): NDArray<T, D> {
    val dtype = DataType.ofKClass(T::class)
    val dim = dimensionClassOf<D>(dims.size)
    requireDimension(dim, dims.size)
    for (i in dims.indices) {
        require(dims[i] > 0) { "Dimension $i must be positive." }
    }
    val size = dims.fold(1, Int::times)
    val data = randData(from, until, size, dtype, gen)
    return NDArray(data, shape = dims, dim = dim)
}

@PublishedApi
@Suppress("UNCHECKED_CAST")
internal inline fun <T : Number> fRand(dtype: DataType): () -> T {
    return when (dtype) {
        DataType.IntDataType -> { { Random.nextInt() } }
        DataType.LongDataType -> { { Random.nextLong() } }
        DataType.FloatDataType -> { { Random.nextFloat() } }
        DataType.DoubleDataType -> { { Random.nextDouble() } }
        else -> throw UnsupportedOperationException("Other types are not currently supported")
    } as () -> T
}

@PublishedApi
@Suppress("UNCHECKED_CAST")
internal inline fun <T : Number> randData(
    from: T, until: T, size: Int, dtype: DataType, gen: Random? = null
): MemoryView<T> {
    var f = 0.0
    var u = 0.0
    val random = gen ?: Random.Default
    if (from is Float && until is Float) {
        f = from.toDouble()
        u = until.toDouble()
    }
    return when {
        from is Int && until is Int -> initMemoryView(size, dtype) { random.nextInt(from, until) }
        from is Long && until is Long -> initMemoryView(size, dtype) { random.nextLong(from, until) }
        from is Float && until is Float -> initMemoryView(size, dtype) { random.nextDouble(f, u).toFloat() }
        from is Double && until is Double -> initMemoryView(size, dtype) { random.nextDouble(from, until) }
        else -> throw UnsupportedOperationException()
    } as MemoryView<T>
}