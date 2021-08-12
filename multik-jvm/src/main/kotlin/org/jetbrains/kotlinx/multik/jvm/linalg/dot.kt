package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.lang.UnsupportedOperationException


//-------------------dotMM--------------------------
internal fun <T : Number> dotMatrix(a: MultiArray<T, D2>, b: MultiArray<T, D2>): D2Array<T> {
    val newShape = intArrayOf(a.shape[0], b.shape[1])
    val size = newShape.reduce(Int::times)
    return when (a.dtype) {
        DataType.FloatDataType -> {
            val ret = D2Array(MemoryViewFloatArray(FloatArray(size)), 0, newShape, dtype = DataType.FloatDataType, dim = D2)
            dotMatrix(a.data.getFloatArray(), a.offset, a.strides, b.data.getFloatArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getFloatArray(), ret.strides[0])
            ret
        }
        DataType.IntDataType -> {
            val ret = D2Array(MemoryViewIntArray(IntArray(size)), 0, newShape, dtype = DataType.IntDataType, dim = D2)
            dotMatrix(a.data.getIntArray(), a.offset, a.strides, b.data.getIntArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getIntArray(), ret.strides[0])
            ret
        }
        DataType.DoubleDataType -> {
            val ret = D2Array(MemoryViewDoubleArray(DoubleArray(size)), 0, newShape, dtype = DataType.DoubleDataType, dim = D2)
            dotMatrix(a.data.getDoubleArray(), a.offset, a.strides, b.data.getDoubleArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getDoubleArray(), ret.strides[0])
            ret
        }
        DataType.LongDataType -> {
            val ret = D2Array(MemoryViewLongArray(LongArray(size)), 0, newShape, dtype = DataType.LongDataType, dim = D2)
            dotMatrix(a.data.getLongArray(), a.offset, a.strides, b.data.getLongArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getLongArray(), ret.strides[0])
            ret
        }
        DataType.ShortDataType -> {
            val ret = D2Array(MemoryViewShortArray(ShortArray(size)), 0, newShape, dtype = DataType.ShortDataType, dim = D2)
            dotMatrix(a.data.getShortArray(), a.offset, a.strides, b.data.getShortArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getShortArray(), ret.strides[0])
            ret
        }
        DataType.ByteDataType -> {
            val ret = D2Array(MemoryViewByteArray(ByteArray(size)), 0, newShape, dtype = DataType.ByteDataType, dim = D2)
            dotMatrix(a.data.getByteArray(), a.offset, a.strides, b.data.getByteArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getByteArray(), ret.strides[0])
            ret
        }
        else -> throw UnsupportedOperationException("expected matrix type inherit Number")
    } as D2Array<T>
}

private fun dotMatrix(
    left: FloatArray, leftOffset: Int, leftStrides: IntArray,
    right: FloatArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: FloatArray, dStrides: Int
): FloatArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}

private fun dotMatrix(
    left: ByteArray, leftOffset: Int, leftStrides: IntArray,
    right: ByteArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: ByteArray, dStrides: Int
): ByteArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] = (destination[dInd + j] + ceil * right[rInd + j * rightStride_1]).toByte()
            }
        }
    }
    return destination
}

private fun dotMatrix(
    left: ShortArray, leftOffset: Int, leftStrides: IntArray,
    right: ShortArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: ShortArray, dStrides: Int
): ShortArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] = (destination[dInd + j] + ceil * right[rInd + j * rightStride_1]).toShort()
            }
        }
    }
    return destination
}

private fun dotMatrix(
    left: IntArray, leftOffset: Int, leftStrides: IntArray,
    right: IntArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: IntArray, dStrides: Int
): IntArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}

private fun dotMatrix(
    left: LongArray, leftOffset: Int, leftStrides: IntArray,
    right: LongArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: LongArray, dStrides: Int
): LongArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}

private fun dotMatrix(
    left: DoubleArray, leftOffset: Int, leftStrides: IntArray,
    right: DoubleArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: DoubleArray, dStrides: Int
): DoubleArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}
//------------------dotMMComplex------------------
internal fun <T : Complex> dotMatrixComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): D2Array<T> {
    val newShape = intArrayOf(a.shape[0], b.shape[1])
    val size = newShape.reduce(Int::times)
    return when (a.dtype) {
        DataType.ComplexDoubleDataType -> {
            val ret = D2Array(MemoryViewComplexDoubleArray(ComplexDoubleArray(size)), 0, newShape, dtype = DataType.ComplexDoubleDataType, dim = D2)
            dotMatrixComplex(a.data.getComplexDoubleArray(), a.offset, a.strides, b.data.getComplexDoubleArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getComplexDoubleArray(), ret.strides[0])
            ret
        }
        DataType.ComplexFloatDataType -> {
            val ret = D2Array(MemoryViewComplexFloatArray(ComplexFloatArray(size)), 0, newShape, dtype = DataType.ComplexFloatDataType, dim = D2)
            dotMatrixComplex(a.data.getComplexFloatArray(), a.offset, a.strides, b.data.getComplexFloatArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getComplexFloatArray(), ret.strides[0])
            ret
        }
        else -> throw UnsupportedOperationException("expected complex matrix")
    } as D2Array<T>
}

private fun dotMatrixComplex(
    left: ComplexDoubleArray, leftOffset: Int, leftStrides: IntArray,
    right: ComplexDoubleArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: ComplexDoubleArray, dStrides: Int
): ComplexDoubleArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}

private fun dotMatrixComplex(
    left: ComplexFloatArray, leftOffset: Int, leftStrides: IntArray,
    right: ComplexFloatArray, rightOffset: Int, rightStrides: IntArray,
    n: Int, m: Int, t: Int, destination: ComplexFloatArray, dStrides: Int
): ComplexFloatArray {
    val (leftStride_0, leftStride_1) = leftStrides
    val (rightStride_0, rightStride_1) = rightStrides

    for (i in 0 until n) {
        val dInd = i * dStrides
        val lInd = i * leftStride_0 + leftOffset
        for (k in 0 until t) {
            val ceil = left[lInd + k * leftStride_1]
            val rInd = k * rightStride_0 + rightOffset
            for (j in 0 until m) {
                destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
            }
        }
    }
    return destination
}



//-------------------dotMV------------------------
internal fun <T : Number> dotMatrixToVector(a: MultiArray<T, D2>, b: MultiArray<T, D1>): D1Array<T> {
    val newShape = intArrayOf(a.shape[0])

    return when (a.dtype) {
        DataType.FloatDataType -> {
            val ret = D1Array(MemoryViewFloatArray(FloatArray(newShape[0])), 0, newShape, dtype = DataType.FloatDataType, dim = D1)
            dotVector(a.data.getFloatArray(), a.offset, a.strides, b.data.getFloatArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getFloatArray())
            ret
        }
        DataType.IntDataType -> {
            val ret = D1Array(MemoryViewIntArray(IntArray(newShape[0])), 0, newShape, dtype = DataType.IntDataType, dim = D1)
            dotVector(a.data.getIntArray(), a.offset, a.strides, b.data.getIntArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getIntArray())
            ret
        }
        DataType.DoubleDataType -> {
            val ret = D1Array(MemoryViewDoubleArray(DoubleArray(newShape[0])), 0, newShape, dtype = DataType.DoubleDataType, dim = D1)
            dotVector(a.data.getDoubleArray(), a.offset, a.strides, b.data.getDoubleArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getDoubleArray())
            ret
        }
        DataType.LongDataType -> {
            val ret = D1Array(MemoryViewLongArray(LongArray(newShape[0])), 0, newShape, dtype = DataType.LongDataType, dim = D1)
            dotVector(a.data.getLongArray(), a.offset, a.strides, b.data.getLongArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getLongArray())
            ret
        }
        DataType.ShortDataType -> {
            val ret = D1Array(MemoryViewShortArray(ShortArray(newShape[0])), 0, newShape, dtype = DataType.ShortDataType, dim = D1)
            dotVector(a.data.getShortArray(), a.offset, a.strides, b.data.getShortArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getShortArray())
            ret
        }
        DataType.ByteDataType -> {
            val ret = D1Array(MemoryViewByteArray(ByteArray(newShape[0])), 0, newShape, dtype = DataType.ByteDataType, dim = D1)
            dotVector(a.data.getByteArray(), a.offset, a.strides, b.data.getByteArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getByteArray())
            ret
        }
        else -> throw UnsupportedOperationException("expected matrix type inherit Number")
    } as D1Array<T>
}

private fun dotVector(
    left: FloatArray, leftOffset: Int, leftStrides: IntArray,
    right: FloatArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: FloatArray
): FloatArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}

private fun dotVector(
    left: IntArray, leftOffset: Int, leftStrides: IntArray,
    right: IntArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: IntArray
): IntArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}

private fun dotVector(
    left: DoubleArray, leftOffset: Int, leftStrides: IntArray,
    right: DoubleArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: DoubleArray
): DoubleArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}

private fun dotVector(
    left: LongArray, leftOffset: Int, leftStrides: IntArray,
    right: LongArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: LongArray
): LongArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}

private fun dotVector(
    left: ShortArray, leftOffset: Int, leftStrides: IntArray,
    right: ShortArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: ShortArray
): ShortArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] =
                (destination[i] + left[lInd + j * lStride_1] * right[j * rStride + rightOffset]).toShort()
        }
    }
    return destination
}

private fun dotVector(
    left: ByteArray, leftOffset: Int, leftStrides: IntArray,
    right: ByteArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: ByteArray
): ByteArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] =
                (destination[i] + left[lInd + j * lStride_1] * right[j * rStride + rightOffset]).toByte()
        }
    }
    return destination
}


//-------------------dotMVComplex-----------------

internal fun <T : Complex> dotMatrixToVectorComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): D1Array<T> {
    val newShape = intArrayOf(a.shape[0])

    return when (a.dtype) {
        DataType.ComplexDoubleDataType -> {
            val ret = D1Array(MemoryViewComplexDoubleArray(ComplexDoubleArray(newShape[0])), 0, newShape, dtype = DataType.ComplexDoubleDataType, dim = D1)
            dotVectorComplex(a.data.getComplexDoubleArray(), a.offset, a.strides, b.data.getComplexDoubleArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getComplexDoubleArray())
            ret
        }
        DataType.ComplexFloatDataType -> {
            val ret = D1Array(MemoryViewComplexFloatArray(ComplexFloatArray(newShape[0])), 0, newShape, dtype = DataType.ComplexFloatDataType, dim = D1)
            dotVectorComplex(a.data.getComplexFloatArray(), a.offset, a.strides, b.data.getComplexFloatArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getComplexFloatArray())
            ret
        }
        else -> throw UnsupportedOperationException("expected Complex matrix")
    } as D1Array<T>
}

private fun dotVectorComplex(
    left: ComplexDoubleArray, leftOffset: Int, leftStrides: IntArray,
    right: ComplexDoubleArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: ComplexDoubleArray
): ComplexDoubleArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}

private fun dotVectorComplex(
    left: ComplexFloatArray, leftOffset: Int, leftStrides: IntArray,
    right: ComplexFloatArray, rightOffset: Int, rStride: Int, n: Int, m: Int, destination: ComplexFloatArray
): ComplexFloatArray {
    val (lStride_0, lStride_1) = leftStrides
    for (i in 0 until n) {
        val lInd = i * lStride_0 + leftOffset
        for (j in 0 until m) {
            destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
        }
    }
    return destination
}


//-------------VecToVec-----------

internal fun <T : Number> dotVecToVec(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = when (a.dtype) {
        DataType.FloatDataType -> dotVecToVec(a.data.getFloatArray(), a.offset, a.strides[0], b.data.getFloatArray(), b.offset, b.strides[0], a.size)
        DataType.IntDataType -> dotVecToVec(a.data.getIntArray(), a.offset, a.strides[0], b.data.getIntArray(), b.offset, b.strides[0], a.size)
        DataType.DoubleDataType -> dotVecToVec(a.data.getDoubleArray(), a.offset, a.strides[0], b.data.getDoubleArray(), b.offset, b.strides[0], a.size)
        DataType.LongDataType -> dotVecToVec(a.data.getLongArray(), a.offset, a.strides[0], b.data.getLongArray(), b.offset, b.strides[0], a.size)
        DataType.ShortDataType -> dotVecToVec(a.data.getShortArray(), a.offset, a.strides[0], b.data.getShortArray(), b.offset, b.strides[0], a.size)
        DataType.ByteDataType -> dotVecToVec(a.data.getByteArray(), a.offset, a.strides[0], b.data.getByteArray(), b.offset, b.strides[0], a.size)
        else -> TODO("Complex numbers")
    } as T

private fun dotVecToVec(
    left: FloatArray, leftOffset: Int, lStride: Int, right: FloatArray, rightOffset: Int, rStride: Int, n: Int
): Float {
    var ret = 0f
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}

private fun dotVecToVec(
    left: IntArray, leftOffset: Int, lStride: Int, right: IntArray, rightOffset: Int, rStride: Int, n: Int
): Int {
    var ret = 0
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}

private fun dotVecToVec(
    left: DoubleArray, leftOffset: Int, lStride: Int, right: DoubleArray, rightOffset: Int, rStride: Int, n: Int
): Double {
    var ret = 0.0
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}

private fun dotVecToVec(
    left: LongArray, leftOffset: Int, lStride: Int, right: LongArray, rightOffset: Int, rStride: Int, n: Int
): Long {
    var ret = 0L
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}

private fun dotVecToVec(
    left: ShortArray, leftOffset: Int, lStride: Int, right: ShortArray, rightOffset: Int, rStride: Int, n: Int
): Short {
    var ret = 0
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret.toShort()
}

private fun dotVecToVec(
    left: ByteArray, leftOffset: Int, lStride: Int, right: ByteArray, rightOffset: Int, rStride: Int, n: Int
): Byte {
    var ret = 0
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret.toByte()
}

//------------VecToVecComplex
//-------------VecToVec-----------

internal fun <T : Complex> dotVecToVecComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = when (a.dtype) {
    DataType.ComplexFloatDataType -> dotVecToVecComplex(a.data.getComplexFloatArray(), a.offset, a.strides[0], b.data.getComplexFloatArray(), b.offset, b.strides[0], a.size)
    DataType.ComplexDoubleDataType -> dotVecToVecComplex(a.data.getComplexDoubleArray(), a.offset, a.strides[0], b.data.getComplexDoubleArray(), b.offset, b.strides[0], a.size)
    else -> throw UnsupportedOperationException("expected Complex vector")
} as T

private fun dotVecToVecComplex(
    left: ComplexFloatArray, leftOffset: Int, lStride: Int, right: ComplexFloatArray, rightOffset: Int, rStride: Int, n: Int
): ComplexFloat {
    var ret = ComplexFloat.zero
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}

private fun dotVecToVecComplex(
    left: ComplexDoubleArray, leftOffset: Int, lStride: Int, right: ComplexDoubleArray, rightOffset: Int, rStride: Int, n: Int
): ComplexDouble {
    var ret = ComplexDouble.zero
    for (i in 0 until n) {
        ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
    }
    return ret
}
