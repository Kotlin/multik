package org.jetbrains.multik.api

import org.jetbrains.multik.core.*
import org.jetbrains.multik.jni.Basic
import kotlin.math.ceil

/**
 * Return a new array with the specified shape.
 */
public inline fun <reified T : Number, D : DN> Multik.empty(vararg dims: Int): Ndarray<T, D> {
    val dtype = DataType.of(T::class)
    val size = dims.reduce { acc, el -> acc * el }
    val data = initMemoryView<T>(size, dtype)
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D>(handle, data, dims, size, dtype, DN.of(dims.size))
}


/**
 * Return a new array given shape from collection.
 */
public inline fun <T : Number, reified D : DN> Multik.ndarray(elements: Collection<T>, shape: IntArray): Ndarray<T, D> {
    assert(shape.isNotEmpty())
    assertDimension<D>(shape.size)
    return ndarray(elements, shape, DN.of(shape.size))
}

/**
 * Return a new array given shape and dimension from collection.
 */
public fun <T : Number, D : DN> Multik.ndarray(elements: Collection<T>, shape: IntArray, dim: DN): Ndarray<T, D> {
    assert(shape.isNotEmpty())
    assert(dim.d == shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val dtype = DataType.of(elements.first())
    val data = initMemoryView<T>(numElements, dtype).apply {
        for (el in elements)
            put(el)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D>(handle, data, shape, numElements, dtype, DN.of(shape.size))
}

/**
 * Return a new array from [ByteArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: ByteArray, shape: IntArray): Ndarray<Byte, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Byte>(numElements, DataType.ByteDataType)
        .apply {
            put(args)
            rewind()
        }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Byte, D>(handle, data, shape, numElements, DataType.ByteDataType, DN.of(shape.size))
}

/**
 * Return a new array from [ShortArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: ShortArray, shape: IntArray): Ndarray<Short, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Short>(
        numElements,
        DataType.ShortDataType
    ).apply {
        put(args)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Short, D>(handle, data, shape, numElements, DataType.ShortDataType, DN.of(shape.size))
}

/**
 * Return a new array from [IntArray].
 */
public inline fun <reified D : DN> Multik.ndarray(args: IntArray, shape: IntArray): Ndarray<Int, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Int>(numElements, DataType.IntDataType).apply {
        put(args)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Int, D>(handle, data, shape, numElements, DataType.IntDataType, DN.of(shape.size))
}

/**
 * Return a new array from [LongArray].
 */
inline fun <reified D : DN> Multik.ndarray(args: LongArray, shape: IntArray): Ndarray<Long, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Long>(numElements, DataType.LongDataType)
        .apply {
            put(args)
            rewind()
        }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Long, D>(handle, data, shape, numElements, DataType.LongDataType, DN.of(shape.size))
}

/**
 * Return a new array from [FloatArray].
 */
inline fun <reified D : DN> Multik.ndarray(args: FloatArray, shape: IntArray): Ndarray<Float, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Float>(numElements, DataType.FloatDataType).apply {
        put(args)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Float, D>(handle, data, shape, numElements, DataType.FloatDataType, DN.of(shape.size))
}

/**
 * Return a new array from [DoubleArray].
 */
inline fun <reified D : DN> Multik.ndarray(args: DoubleArray, shape: IntArray): Ndarray<Double, D> {
    assertDimension<D>(shape.size)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<Double>(numElements, DataType.DoubleDataType).apply {
        put(args)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<Double, D>(handle, data, shape, numElements, DataType.DoubleDataType, DN.of(shape.size))
}

/**
 * Return a 1-dimension array.
 */
inline fun <reified T : Number> Multik.d1array(sizeD1: Int, init: (Int) -> T): Ndarray<T, D1> {
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(sizeD1, dtype).apply {
        for (i in 0 until sizeD1)
            put(init(i))
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D1>(handle, data, intArrayOf(sizeD1), sizeD1, dtype, D1)
}

/**
 * Return a 2-dimensions array.
 */
inline fun <reified T : Number> Multik.d2array(sizeD1: Int, sizeD2: Int, init: (Int) -> T): Ndarray<T, D2> {
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(sizeD1 * sizeD2, dtype).apply {
        for (i in 0 until sizeD1 * sizeD2)
            put(init(i))
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D2>(handle, data, intArrayOf(sizeD1, sizeD2), sizeD1 * sizeD2, dtype, D2)
}

/**
 * Return a 3-dimensions array.
 */
inline fun <reified T : Number> Multik.d3array(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    init: (Int) -> T
): Ndarray<T, D3> {
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(sizeD1 * sizeD2 * sizeD3, dtype).apply {
        for (i in 0 until sizeD1 * sizeD2 * sizeD3)
            put(init(i))
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D3>(handle, data, shape = intArrayOf(sizeD1, sizeD2, sizeD3), dtype = dtype, dim = D3)
}

/**
 * Return a 4-dimensions array.
 */
inline fun <reified T : Number> Multik.d4array(sizeD1: Int, sizeD2: Int, sizeD3: Int, sizeD4: Int, init: (Int) -> T)
        : Ndarray<T, D4> {
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(
        sizeD1 * sizeD2 * sizeD3 * sizeD4,
        dtype
    ).apply {
        for (i in 0 until sizeD1 * sizeD2 * sizeD3 * sizeD4)
            put(init(i))
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray<T, D4>(handle, data, shape = intArrayOf(sizeD1, sizeD2, sizeD3, sizeD4), dtype = dtype, dim = D4)
}

/**
 * Return a new array with the specified [size], where each element is calculated by calling the specified
 * [init] function.
 */
@Suppress("UNCHECKED_CAST")
inline fun <reified T : Number> Multik.ndarray(size: Int, init: (Int) -> T): Ndarray<T, DN> =
    d1array(size, init) as Ndarray<T, DN>


/**
 * Return a new array with the specified [shape], where each element is calculated by calling the specified
 * [init] function.
 */
inline fun <reified T : Number> Multik.ndarray(vararg shape: Int, init: (Int) -> T): Ndarray<T, DN> {
    val dtype = DataType.of(T::class)
    val numElements = shape.reduce { acc, el -> acc * el }
    val data = initMemoryView<T>(numElements, dtype).apply {
        for (i in 0 until numElements)
            put(init(i))
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, shape, numElements, dtype, DN.of(shape.size))
}

/**
 * Return a new 1-dimension array from [items].
 */
fun <T : Number> Multik.ndarrayOf(vararg items: T): Ndarray<T, D1> {
    val dtype = DataType.of(items.first())
    val data = initMemoryView<T>(items.size, dtype).apply {
        put(items)
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, intArrayOf(items.size), items.size, dtype, D1)
}

/**
 * Return evenly spaced values within a given interval, where [step] is Integer.
 */
inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Int = 1): Ndarray<T, D1> {
    val size = ceil((stop.toDouble() - start) / step).toInt()
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(size, dtype).apply {
        var tmp = start
        for (i in 1..size) {
            put(tmp)
            tmp += step
        }
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, intArrayOf(size), size, dtype, D1)
}

/**
 * Return evenly spaced values within a given interval, where [step] is Double.
 */
inline fun <reified T : Number> Multik.arange(start: Int, stop: Int, step: Double): Ndarray<T, D1> {
    val size = ceil((stop.toDouble() - start) / step).toInt()
    val dtype = DataType.of(T::class)
    val data = initMemoryView<T>(size, dtype).apply {
        var tmp = start.toDouble()
        for (i in 1..size) {
            put(tmp)
            tmp += step
        }
        rewind()
    }
    val handle = Basic.allocate(data.getData())
    return Ndarray(handle, data, intArrayOf(size), size, dtype, D1)
}

/**
 * see [Multik.arange]
 */
inline fun <reified T : Number> Multik.arange(stop: Int, step: Int = 1): Ndarray<T, D1> = arange(0, stop, step)

/**
 * see [Multik.arange]
 */
inline fun <reified T : Number> Multik.arange(stop: Int, step: Double): Ndarray<T, D1> = arange(0, stop, step)

/**
 * Return evenly spaced values within a given interval, more precisely than [arange].
 */
fun Multik.linspace(start: Int, stop: Int, num: Int = 50): Ndarray<Double, D1> {
    if (num < 0)
        throw IllegalArgumentException("")
    val div = num - 1
    val delta = stop - start
    val ret = arange<Double>(0, stop = num)
    if (num > 1) {
        val step = delta.toDouble() / div.toDouble()
        ret *= step
    }

    ret += start.toDouble()
    ret[ret.size - 1] = stop.toDouble()
    return ret
}

//todo
fun <T : Number> Iterable<T>.toNdarray(): Ndarray<T, D1> {
    if (this is Collection<T>)
        return Multik.ndarray(this, intArrayOf(this.size))

    val tmp = ArrayList<T>()
    for (item in this) {
        tmp.add(item)
    }
    return Multik.ndarray(tmp, intArrayOf(tmp.size))
}

inline fun <reified D : DN> assertDimension(shapeSize: Int) {
    when (D::class) {
        D1::class -> assert(shapeSize == 1)
        D2::class -> assert(shapeSize == 2)
        D3::class -> assert(shapeSize == 3)
        D4::class -> assert(shapeSize == 4)
        DN::class -> assert(shapeSize >= 5)
    }
}
