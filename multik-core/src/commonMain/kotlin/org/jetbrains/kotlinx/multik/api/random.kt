package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName
import kotlin.random.Random

/**
 * Creates a 1D array filled with uniformly distributed random numbers.
 *
 * Default ranges by type: `Int` — [[Int.MIN_VALUE], [Int.MAX_VALUE]),
 * `Long` — [[Long.MIN_VALUE], [Long.MAX_VALUE]), `Float` — [0f, 1f), `Double` — [0.0, 1.0).
 *
 * ```
 * mk.rand<Double>(5) // e.g. [0.123, 0.456, 0.789, 0.012, 0.345]
 * ```
 *
 * @param dim0 the array size (must be positive).
 * @return a new [D1Array] of random values.
 */
public inline fun <reified T : Number> Multik.rand(dim0: Int): D1Array<T> {
    require(dim0 > 0) { "Dimension must be positive." }
    val dtype = DataType.ofKClass(T::class)
    val frand: () -> T = fRand(dtype)
    val data = initMemoryView(dim0, dtype) { frand() }
    return D1Array(data, shape = intArrayOf(dim0), dim = D1)
}

/**
 * Creates a 2D array of shape ([dim0], [dim1]) filled with uniformly distributed random numbers.
 *
 * @see [Multik.rand] (1D) for default ranges by type.
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
 * Creates a 3D array of shape ([dim0], [dim1], [dim2]) filled with uniformly distributed random numbers.
 *
 * @see [Multik.rand] (1D) for default ranges by type.
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
 * Creates a 4D array of shape ([dim0], [dim1], [dim2], [dim3]) filled with uniformly distributed random numbers.
 *
 * @see [Multik.rand] (1D) for default ranges by type.
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
 * Creates an N-dimensional array filled with uniformly distributed random numbers.
 *
 * @see [Multik.rand] (1D) for default ranges by type.
 */
public inline fun <reified T : Number> Multik.rand(
    dim0: Int, dim1: Int, dim2: Int, dim3: Int, vararg dims: Int
): NDArray<T, DN> {
    return rand(intArrayOf(dim0, dim1, dim2, dim3, *dims))
}

/**
 * Creates an array of the given [shape] filled with uniformly distributed random numbers.
 *
 * @param shape the array shape; all dimensions must be positive.
 * @return a new [NDArray] of random values.
 * @see [Multik.rand] (1D) for default ranges by type.
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
 * Creates an array filled with random numbers uniformly distributed in [[from], [until]).
 *
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape as vararg.
 */
@JvmName("randWithVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    from: T, until: T, vararg dims: Int
): NDArray<T, D> =
    Multik.rand(from, until, dims)

/**
 * Creates an array filled with random numbers uniformly distributed in [[from], [until]).
 *
 * Note: `Float` generation is inefficient (converts through `Double` internally).
 *
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape.
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
 * Creates an array filled with random numbers in [[from], [until]) using the given [seed] for reproducibility.
 *
 * @param seed the random seed.
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape as vararg.
 */
@JvmName("randSeedVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, vararg dims: Int
): NDArray<T, D> = Multik.rand(Random(seed), from, until, dims)

/**
 * Creates an array filled with random numbers in [[from], [until]) using the given [seed] for reproducibility.
 *
 * @param seed the random seed.
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape.
 */
@JvmName("randSeedShape")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, dims: IntArray
): NDArray<T, D> = Multik.rand(Random(seed), from, until, dims)

/**
 * Creates an array filled with random numbers in [[from], [until]) using the given [gen] random generator.
 *
 * @param gen the [Random] generator to use.
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape as vararg.
 */
@JvmName("randGenVarArg")
public inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    gen: Random, from: T, until: T, vararg dims: Int
): NDArray<T, D> = Multik.rand(gen, from, until, dims)

/**
 * Creates an array filled with random numbers in [[from], [until]) using the given [gen] random generator.
 *
 * Note: `Float` generation is inefficient (converts through `Double` internally).
 *
 * @param gen the [Random] generator to use.
 * @param from inclusive lower bound.
 * @param until exclusive upper bound.
 * @param dims the array shape.
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

/** Returns a random element generator lambda for the given [dtype]. Supports Int, Long, Float, Double. */
@PublishedApi
@Suppress("unchecked_cast", "nothing_to_inline")
internal inline fun <T : Number> fRand(dtype: DataType): () -> T {
    return when (dtype) {
        DataType.IntDataType -> { { Random.nextInt() } }
        DataType.LongDataType -> { { Random.nextLong() } }
        DataType.FloatDataType -> { { Random.nextFloat() } }
        DataType.DoubleDataType -> { { Random.nextDouble() } }
        else -> throw UnsupportedOperationException("Other types are not currently supported")
    } as () -> T
}

/** Generates a [MemoryView] of [size] random elements in [[from], [until]) for the given [dtype]. */
@PublishedApi
@Suppress( "unchecked_cast", "nothing_to_inline")
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
