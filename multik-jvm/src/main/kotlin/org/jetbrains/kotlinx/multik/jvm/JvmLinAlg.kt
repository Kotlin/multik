/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import kotlin.math.abs

public object JvmLinAlg : LinAlg {

    override val linAlgEx: LinAlgEx
        get() = JvmLinAlgEx


//    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<T, D2> {
//        TODO("Not yet implemented")
//    }

    fun <T : Number> invDouble(a: MultiArray<T, D2>): D2Array<Double> {
        requireSquare(a)
        return solveDouble(a.map { it.toDouble() }, mk.identity(a.shape[0]))
    }

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        if (n == 0) return mk.identity(mat.shape[0], mat.dtype)

        return if (n % 2 == 0) {
            val tmp = pow(mat, n / 2)
            dot(tmp, tmp)
        } else {
            dot(mat, pow(mat, n - 1))
        }

    }



    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {

        require(p > 0) { "Power $p must be positive" }

        return when (mat.dtype) {
            DataType.DoubleDataType -> {
                norm(mat.data.getDoubleArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
            DataType.FloatDataType -> {
                norm(mat.data.getFloatArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
            else -> {
                norm(mat.data, mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
        }
    }



//    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
//        if(a.dtype != DataType.DoubleDataType) {
//            throw UnsupportedOperationException()
//        }
//
//        val aDouble = a.map { it.toDouble() }
//        val bDouble: MultiArray<Double, D2> = if (b.dim.d == 2) { b } else { b.reshape(b.shape[0], 1) } as MultiArray<Double, D2>
//
//        val ans = solveDouble(aDouble, bDouble)
//        return if (b.dim.d == 2) { ans } else { ans.reshape(ans.shape[0]) } as NDArray<T, D>
//    }

    private fun solveDouble(a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, singularityErrorLevel: Double = 1e-7): D2Array<Double> {
        requireSquare(a)
        require( a.shape[1] == b.shape[0])
            { "Shapes of arguments are incompatible: expected a.shape[1] = ${a.shape[1]} to be equal to the b.shape[0] = ${b.shape[0]}" }
        val (P, L, U) = pluCompressed(a)
        val _b: D2Array<Double> = (b as D2Array<Double>).deepCopy()

        for (i in P.indices) {
            if(P[i] != 0) {
                _b[i] = _b[i + P[i]].deepCopy().also { _b[i + P[i]] = _b[i].deepCopy() }
            }
        }
        for (i in 0 until U.shape[0]) {
            if (abs(U[i, i]) < singularityErrorLevel) {
                throw ArithmeticException("Matrix a is singular or almost singular")
            }
        }

        return solveTriangle(U, solveTriangle(L, _b), false)
    }

    private fun <T : Number> requireSquare(a: MultiArray<T, D2>) {
        require(a.shape[0] == a.shape[1]) { "Square matrix expected, shape=(${a.shape[0]}, ${a.shape[1]}) given" }
    }

    /**
     * solves a*x = b where a lower or upper triangle square matrix
     */
    private fun solveTriangle(a: MultiArray<Double, D2>, b: MultiArray<Double, D2>, isLowerTriangle: Boolean = true): D2Array<Double> {
        require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
        requireSquare(a)
        val x = b.map { it }
        for (i in 0 until x.shape[0]) {
            for (j in 0 until x.shape[1]) {
                x[i, j] /= a[i, i]
            }
        }
        if (isLowerTriangle) {
            for (i in 0 until x.shape[0]) {
                for (k in i + 1 until x.shape[0]) {
                    for (j in 0 until x.shape[1]) {
                        x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                    }
                }
            }
        } else {
            for (i in x.shape[0] - 1 downTo 0) {
                for (k in i - 1 downTo 0) {
                    for (j in 0 until x.shape[1]) {
                        x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                    }
                }
            }
        }
        return x
    }




    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[1] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
        }

        return if (b.dim.d == 2) {
            dotMatrix(a, b as D2Array<T>) as NDArray<T, D>
        } else {
            dotVector(a, b as D1Array<T>) as NDArray<T, D>
        }
    }

    private fun <T : Number> dotMatrix(a: MultiArray<T, D2>, b: MultiArray<T, D2>): D2Array<T> {
        val newShape = intArrayOf(a.shape[0], b.shape[1])
        return when (a.dtype) {
            DataType.FloatDataType -> {
                val ret = D2Array(MemoryViewFloatArray(FloatArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.FloatDataType, dim = D2)
                dotMatrix(a.data.getFloatArray(), a.offset, a.strides, b.data.getFloatArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getFloatArray(), ret.strides[0])
                ret
            }
            DataType.IntDataType -> {
                val ret = D2Array(MemoryViewIntArray(IntArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.IntDataType, dim = D2)
                dotMatrix(a.data.getIntArray(), a.offset, a.strides, b.data.getIntArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getIntArray(), ret.strides[0])
                ret
            }
            DataType.DoubleDataType -> {
                val ret = D2Array(MemoryViewDoubleArray(DoubleArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.DoubleDataType, dim = D2)
                dotMatrix(a.data.getDoubleArray(), a.offset, a.strides, b.data.getDoubleArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getDoubleArray(), ret.strides[0])
                ret
            }
            DataType.LongDataType -> {
                val ret = D2Array(MemoryViewLongArray(LongArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.LongDataType, dim = D2)
                dotMatrix(a.data.getLongArray(), a.offset, a.strides, b.data.getLongArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getLongArray(), ret.strides[0])
                ret
            }
            DataType.ShortDataType -> {
                val ret = D2Array(MemoryViewShortArray(ShortArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.ShortDataType, dim = D2)
                dotMatrix(a.data.getShortArray(), a.offset, a.strides, b.data.getShortArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getShortArray(), ret.strides[0])
                ret
            }
            DataType.ByteDataType -> {
                val ret = D2Array(MemoryViewByteArray(ByteArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.ByteDataType, dim = D2)
                dotMatrix(a.data.getByteArray(), a.offset, a.strides, b.data.getByteArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getByteArray(), ret.strides[0])
                ret
            }
            else -> TODO("Complex numbers")
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

    private fun <T : Number> dotVector(a: MultiArray<T, D2>, b: MultiArray<T, D1>): D1Array<T> {
        val newShape = intArrayOf(a.shape[0])

        return when (a.dtype) {
            DataType.FloatDataType -> {
                val ret = D1Array(
                    MemoryViewFloatArray(FloatArray(newShape[0])),
                    0,
                    newShape,
                    dtype = DataType.FloatDataType,
                    dim = D1
                )
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
            else -> TODO("Complex numbers")
        } as D1Array<T>
    }

    private fun dotVector(
        left: FloatArray, leftOffset: Int, leftStrides: IntArray,
        right: FloatArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: FloatArray
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
        right: IntArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: IntArray): IntArray {
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
        right: DoubleArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: DoubleArray): DoubleArray {
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
        right: LongArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: LongArray): LongArray {
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
        right: ShortArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: ShortArray
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
        right: ByteArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: ByteArray
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

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.size == b.size) { "Sizes a and b don't match: a.size(${a.size}) != b.size(${b.size})" }
        return when (a.dtype) {
            DataType.FloatDataType -> {
                dotVecToVec(a.data.getFloatArray(), a.offset, a.strides[0], b.data.getFloatArray(), b.offset, b.strides[0], a.size)
            }
            DataType.IntDataType -> {
                dotVecToVec(a.data.getIntArray(), a.offset, a.strides[0], b.data.getIntArray(), b.offset, b.strides[0], a.size)
            }
            DataType.DoubleDataType -> {
                dotVecToVec(a.data.getDoubleArray(), a.offset, a.strides[0], b.data.getDoubleArray(), b.offset, b.strides[0], a.size)
            }
            DataType.LongDataType -> {
                dotVecToVec(a.data.getLongArray(), a.offset, a.strides[0], b.data.getLongArray(), b.offset, b.strides[0], a.size)
            }
            DataType.ShortDataType -> {
                dotVecToVec(a.data.getShortArray(), a.offset, a.strides[0], b.data.getShortArray(), b.offset, b.strides[0], a.size)
            }
            DataType.ByteDataType -> {
                dotVecToVec(a.data.getByteArray(), a.offset, a.strides[0], b.data.getByteArray(), b.offset, b.strides[0], a.size)
            }
            else -> TODO("Complex numbers")
        } as T
    }

    private fun dotVecToVec(
        left: FloatArray, leftOffset: Int, lStride: Int,
        right: FloatArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Float {
        var ret = 0f
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: IntArray, leftOffset: Int, lStride: Int,
        right: IntArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Int {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: DoubleArray, leftOffset: Int, lStride: Int,
        right: DoubleArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Double {
        var ret = 0.0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: LongArray, leftOffset: Int, lStride: Int,
        right: LongArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Long {
        var ret = 0L
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: ShortArray, leftOffset: Int, lStride: Int,
        right: ShortArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Short {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret.toShort()
    }

    private fun dotVecToVec(
        left: ByteArray, leftOffset: Int, lStride: Int,
        right: ByteArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Byte {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret.toByte()
    }
}

public object JvmLinAlgEx: LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        TODO("Not yet implemented")
    }

}
