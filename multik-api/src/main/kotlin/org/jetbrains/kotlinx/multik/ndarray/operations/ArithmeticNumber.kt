package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public operator fun <D : Dimension> Byte.plus(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Short.plus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Int.plus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Long.plus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Float.plus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Double.plus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> ComplexFloat.plus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> ComplexDouble.plus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret += other
    return other
}

public operator fun <D : Dimension> Byte.minus(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getByteArray()
    for (i in ret.indices) {
        data[i] = (this - data[i]).toByte()
    }
    return other
}

public operator fun <D : Dimension> Short.minus(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getShortArray()
    for (i in ret.indices) {
        data[i] = (this - data[i]).toShort()
    }
    return other
}

public operator fun <D : Dimension> Int.minus(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getIntArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> Long.minus(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getLongArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> Float.minus(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getFloatArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> Double.minus(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getDoubleArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> ComplexFloat.minus(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getComplexFloatArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> ComplexDouble.minus(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getComplexDoubleArray()
    for (i in ret.indices) {
        data[i] = this - data[i]
    }
    return other
}

public operator fun <D : Dimension> Byte.times(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Short.times(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Int.times(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Long.times(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Float.times(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Double.times(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> ComplexFloat.times(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> ComplexDouble.times(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    ret *= other
    return other
}

public operator fun <D : Dimension> Byte.div(other: MultiArray<Byte, D>): NDArray<Byte, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getByteArray()
    for (i in ret.indices) {
        data[i] = (this / data[i]).toByte()
    }
    return other
}

public operator fun <D : Dimension> Short.div(other: MultiArray<Short, D>): NDArray<Short, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getShortArray()
    for (i in ret.indices) {
        data[i] = (this / data[i]).toShort()
    }
    return other
}

public operator fun <D : Dimension> Int.div(other: MultiArray<Int, D>): NDArray<Int, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getIntArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}

public operator fun <D : Dimension> Long.div(other: MultiArray<Long, D>): NDArray<Long, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getLongArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}

public operator fun <D : Dimension> Float.div(other: MultiArray<Float, D>): NDArray<Float, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getFloatArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}

public operator fun <D : Dimension> Double.div(other: MultiArray<Double, D>): NDArray<Double, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getDoubleArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}

public operator fun <D : Dimension> ComplexFloat.div(other: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getComplexFloatArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}

public operator fun <D : Dimension> ComplexDouble.div(other: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> {
    val ret = if (other.consistent) (other as NDArray).clone() else (other as NDArray).deepCopy()
    val data = other.data.getComplexDoubleArray()
    for (i in ret.indices) {
        data[i] = this / data[i]
    }
    return other
}
