package org.jetbrains.multik.api

import org.jetbrains.multik.ndarray.data.*

/**
 *
 */
public interface Statistics {

    /**
     *
     */
    public fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double?

    /**
     *
     */
    public fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>? = null): Double

    /**
     *
     */
    public fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double

    /**
     *
     */
    public fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): Ndarray<Double, O>

    /**
     *
     */
    public fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): Ndarray<Double, D1>

    /**
     *
     */
    public fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): Ndarray<Double, D2>

    /**
     *
     */
    public fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): Ndarray<Double, D3>

    /**
     *
     */
    public fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): Ndarray<Double, D4>
}

@JvmName("absByte")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Byte, D>): Ndarray<Byte, D> {
    val ret = initMemoryView<Byte>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absByte(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absShort")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Short, D>): Ndarray<Short, D> {
    val ret = initMemoryView<Short>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = absShort(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absInt")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Int, D>): Ndarray<Int, D> {
    val ret = initMemoryView<Int>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absLong")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Long, D>): Ndarray<Long, D> {
    val ret = initMemoryView<Long>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absFloat")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Float, D>): Ndarray<Float, D> {
    val ret = initMemoryView<Float>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

@JvmName("absDouble")
public fun <D : Dimension> Statistics.abs(a: MultiArray<Double, D>): Ndarray<Double, D> {
    val ret = initMemoryView<Double>(a.size, a.dtype)
    var index = 0
    for (element in a) {
        ret[index++] = kotlin.math.abs(element)
    }
    return Ndarray(ret, 0, a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
}

private inline fun absByte(a: Byte): Byte = if (a < 0) (-a).toByte() else a

private inline fun absShort(a: Short): Short = if (a < 0) (-a).toShort() else a