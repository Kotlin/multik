package org.jetbrains.multik.api

import org.jetbrains.multik.core.*
import kotlin.math.ceil

/**
 * Return a new array with the specified shape.
 */
public inline fun <reified T : Number, reified D : DN> Multik.empty(vararg dims: Int): Ndarray<T, D> {
    val dim = DN.of<D>()
    requireDimension(dim, dims.size)
    val dtype = DataType.of(T::class)
    val size = dims.reduce { acc, el -> acc * el }
    val data = initMemoryView<T>(size, dtype)
    return initNdarray<T, D>(data, shape = dims, dtype = dtype, dim = dim)
}

/**
 *
 */
public inline fun <reified T : Number> Multik.identity(n: Int): D2Array<T> {
    val dtype = DataType.of(T::class)
    return identity(n, dtype)
}

/**
 *
 */
public fun <T : Number> Multik.identity(n: Int, dtype: DataType): D2Array<T> {
    val shape = intArrayOf(n, n)
    val ret = D2Array<T>(initMemoryView<T>(n * n, dtype), shape = shape, dtype = dtype)
    val one: Number = when (dtype.nativeCode) {
        1 -> 1.toByte()
        2 -> 1.toShort()
        3 -> 1
        4 -> 1L
        5 -> 1f
        6 -> 1.0
        else -> throw Exception("Type not defined.")
    }
    for (i in 0 until n) {
        @Suppress("UNCHECKED_CAST")
        ret[i, i] = one as T
    }
    return ret
}

/**
 *
 */
@JvmName("ndarray1D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<T>): D1Array<T> {
    val dtype = DataType.of(T::class)
    val data = arg.toViewPrimitiveArray(dtype)
    return D1Array(data, 0, intArrayOf(arg.size), dtype = dtype)
}

/**
 *
 */
@JvmName("ndarray2D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<T>>): D2Array<T> {
    val dtype = DataType.of(T::class)
    val size = IntArray(2)
    size[0] = arg.size
    size[1] = arg.first().size
    val res = ArrayList<T>()
    for (ax0 in arg) {
        check(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        res.addAll(ax0)
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D2Array(data, 0, size, dtype = dtype)
}

/**
 *
 */
@JvmName("ndarray3D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<List<T>>>): D3Array<T> {
    val dtype = DataType.of(T::class)
    val size = IntArray(3)
    size[0] = arg.size
    size[1] = arg.first().size
    size[2] = arg.first().first().size
    val res = ArrayList<T>()
    for (ax0 in arg) {
        check(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        for (ax1 in ax0) {
            check(size[2] == ax1.size) { "The size of the incoming array $ax1 does not match the rest" }
            res.addAll(ax1)
        }
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D3Array(data, 0, size, dtype = dtype)
}

/**
 *
 */
@JvmName("ndarray4D")
public inline fun <reified T : Number> Multik.ndarray(arg: List<List<List<List<T>>>>): D4Array<T> {
    val dtype = DataType.of(T::class)
    val size = IntArray(4)
    size[0] = arg.size
    size[1] = arg.first().size
    size[2] = arg.first().first().size
    size[3] = arg.first().first().first().size
    val res = ArrayList<T>()
    for (ax0 in arg) {
        check(size[1] == ax0.size) { "The size of the incoming array $ax0 does not match the rest" }
        for (ax1 in ax0) {
            check(size[2] == ax1.size) { "The size of the incoming array $ax1 does not match the rest" }
            for (ax2 in ax1) {
                check(size[3] == ax2.size) { "The size of the incoming array $ax2 does not match the rest" }
                res.addAll(ax2)
            }
        }
    }
    val data = res.toViewPrimitiveArray(dtype)
    return D4Array(data, 0, size, dtype = dtype)
}


/**
 * Return a new array given shape from collection.
 */
public inline fun <T : Number, reified D : DN> Multik.ndarray(elements: Collection<T>, shape: IntArray): Ndarray<T, D> {
    requireShapeEmpty(shape)
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    return ndarray(elements, shape, dim)
}

/**
 * Return a new array given shape and dimension from collection.
 */
public fun <T : Number, D : DN> Multik.ndarray(elements: Collection<T>, shape: IntArray, dim: D): Ndarray<T, D> {
    requireShapeEmpty(shape)
    requireDimension(dim, shape.size)
    val size = shape.reduce { acc, el -> acc * el }
    val dtype = DataType.of(elements.first())
    val data = initMemoryView<T>(size, dtype).apply {
        var count = 0
        for (el in elements)
            this[count++] = el
    }
    return initNdarray<T, D>(data, shape = shape, dtype = dtype, dim = DN.of(shape.size))
}

/**
 * Return a new array from [ByteArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: ByteArray, vararg shape: Int): Ndarray<Byte, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewByteArray(args)
    return initNdarray<Byte, D>(data, shape = shape, dtype = DataType.ByteDataType, dim = dim)
}

/**
 * Return a new array from [ShortArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: ShortArray, vararg shape: Int): Ndarray<Short, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewShortArray(args)
    return initNdarray<Short, D>(data, shape = shape, dtype = DataType.ShortDataType, dim = dim)
}

/**
 * Return a new array from [IntArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: IntArray, vararg shape: Int): Ndarray<Int, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewIntArray(args)
    return initNdarray<Int, D>(data, shape = shape, dtype = DataType.IntDataType, dim = dim)
}

/**
 * Return a new array from [LongArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: LongArray, vararg shape: Int): Ndarray<Long, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewLongArray(args)
    return initNdarray<Long, D>(data, shape = shape, dtype = DataType.LongDataType, dim = dim)
}

/**
 * Return a new array from [FloatArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: FloatArray, vararg shape: Int): Ndarray<Float, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewFloatArray(args)
    return initNdarray<Float, D>(data, shape = shape, dtype = DataType.FloatDataType, dim = dim)
}

/**
 * Return a new array from [DoubleArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: DoubleArray, vararg shape: Int): Ndarray<Double, D> {
    val dim = DN.of<D>()
    requireDimension(dim, shape.size)
    val data = MemoryViewDoubleArray(args)
    return initNdarray<Double, D>(data, shape = shape, dtype = DataType.DoubleDataType, dim = dim)
}

/**
 * Return a 1-dimension array.
 */
public inline fun <reified T : Number> Multik.d1array(sizeD1: Int, noinline init: (Int) -> T): D1Array<T> {
    val dtype = DataType.of(T::class)
    val shape = intArrayOf(sizeD1)
    val data = initMemoryView<T>(sizeD1, dtype, init)
    return D1Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * Return a 2-dimensions array.
 */
public inline fun <reified T : Number> Multik.d2array(sizeD1: Int, sizeD2: Int, noinline init: (Int) -> T): D2Array<T> {
    val dtype = DataType.of(T::class)
    val shape = intArrayOf(sizeD1, sizeD2)
    val data = initMemoryView<T>(sizeD1 * sizeD2, dtype, init)
    return D2Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * Return a 3-dimensions array.
 */
public inline fun <reified T : Number> Multik.d3array(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    noinline init: (Int) -> T
): D3Array<T> {
    val dtype = DataType.of(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3)
    val data = initMemoryView<T>(sizeD1 * sizeD2 * sizeD3, dtype, init)
    return D3Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * Return a 4-dimensions array.
 */
public inline fun <reified T : Number> Multik.d4array(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    sizeD4: Int,
    noinline init: (Int) -> T
): D4Array<T> {
    val dtype = DataType.of(T::class)
    val shape = intArrayOf(sizeD1, sizeD2, sizeD3, sizeD4)
    val data = initMemoryView<T>(sizeD1 * sizeD2 * sizeD3 * sizeD4, dtype, init)
    return D4Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * Return a new array with the specified [shape], where each element is calculated by calling the specified
 * [init] function.
 */
public inline fun <reified T : Number> Multik.ndarray(vararg shape: Int, noinline init: (Int) -> T): Ndarray<T, DN> {
    val dtype = DataType.of(T::class)
    val size = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<T>(size, dtype, init)
    return initNdarray(data, shape = shape, dtype = dtype, dim = DN.of(shape.size))
}

/**
 * Return a new 1-dimension array from [items].
 */
public fun <T : Number> Multik.ndarrayOf(vararg items: T): D1Array<T> {
    val dtype = DataType.of(items.first())
    val shape = intArrayOf(items.size)
    val data = initMemoryView<T>(items.size, dtype) { items[it] }
    return D1Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * Return evenly spaced values within a given interval, where [step] is Integer.
 */
public inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Int = 1): D1Array<T> {
    return arange(start, stop, step.toDouble())
}

/**
 * Return evenly spaced values within a given interval, where [step] is Double.
 */
public inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Double): D1Array<T> {
    if (start < stop) require(step > 0) { "Step must be positive." }
    else if (start > stop) require(step < 0) { "Step must be negative." }

    val size = ceil((stop.toDouble() - start) / step).toInt()
    val dtype = DataType.of(T::class)
    val shape = intArrayOf(size)
    val data = initMemoryView<T>(size, dtype).apply {
        var tmp = start.toDouble()
        for (i in indices) {
            this[i] = tmp.toPrimitiveType()
            tmp += step
        }
    }
    return D1Array<T>(data, shape = shape, dtype = dtype)
}

/**
 * see [Multik.arange]
 */
public inline fun <reified T : Number> Multik.arange(stop: Int, step: Int = 1): D1Array<T> = arange(0, stop, step)

/**
 * see [Multik.arange]
 */
public inline fun <reified T : Number> Multik.arange(stop: Int, step: Double): D1Array<T> = arange(0, stop, step)


/**
 * Return evenly spaced values within a given interval, more precisely than [arange].
 */
public inline fun <reified T : Number> Multik.linspace(start: Int, stop: Int, num: Int = 50): D1Array<T> {
    return linspace(start.toDouble(), stop.toDouble(), num)
}

public inline fun <reified T : Number> Multik.linspace(start: Double, stop: Double, num: Int = 50): D1Array<T> {
    require(num > 0) { "The number of elements cannot be less than zero or equal to zero." }
    val div = num - 1.0
    val delta = stop - start
    val ret = arange<Double>(0, stop = num)
    if (num > 1) {
        val step = delta / div
        ret *= step
    }

    ret += start
    ret[ret.size - 1] = stop
    return ret.asType<T>().toD1Array()
}

public fun <T : Number> Iterable<T>.toNdarray(): D1Array<T> {
    if (this is Collection<T>)
        return Multik.ndarray<T, D1>(this, intArrayOf(this.size), D1) as D1Array<T>

    val tmp = ArrayList<T>()
    for (item in this) {
        tmp.add(item)
    }
    return Multik.ndarray<T, D1>(tmp, intArrayOf(tmp.size), D1) as D1Array<T>
}
