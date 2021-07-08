/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas
import org.jetbrains.kotlinx.multik.api.LinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object CudaLinAlg : LinAlg {
    init {
        CudaEngine
    }

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
    }

    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D: Dim2> isTransposedConsistent(x: MultiArray<T, D>): Boolean {
        return x.transpose().consistent
    }

    private fun <T : Number, D: Dim2> getConsistentOrTransposedConsistent(x: MultiArray<T, D>): Pair<MultiArray<T, D>, Boolean> {
        return when {
            x.consistent -> x to false
            x.dim.d == 2 && isTransposedConsistent(x) -> x to true
            else -> x.deepCopy() to false
        }
    }

    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[1] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
        }

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val matrixMatrix = b.dim.d == 2

        val elemSize = when (a.dtype) {
            DataType.DoubleDataType -> {
                Double.SIZE_BYTES
            }
            DataType.FloatDataType -> {
                Float.SIZE_BYTES
            }
            else -> throw UnsupportedOperationException()
        }

        val shape = if (matrixMatrix)
            intArrayOf(a.shape[0], b.shape[1])
        else
            intArrayOf(a.shape[0])

        val dA = Pointer()
        val dB = Pointer()
        val dC = Pointer()

        val cSize = shape.reduce(Int::times)
        val phA: Pointer
        val phB: Pointer
        val phC: Pointer

        val hA: Any
        val hB: Any
        val hC: Any

        if (a.dtype == DataType.FloatDataType) {
            hA = a.data.getFloatArray()
            hB = b.data.getFloatArray()
            hC = FloatArray(cSize)

            phA = Pointer.to(hA)
            phB = Pointer.to(hB)
            phC = Pointer.to(hC)
        } else {
            hA = a.data.getDoubleArray()
            hB = b.data.getDoubleArray()
            hC = DoubleArray(cSize)

            phA = Pointer.to(hA)
            phB = Pointer.to(hB)
            phC = Pointer.to(hC)
        }

        JCublas.cublasAlloc(a.size, elemSize, dA)
        JCublas.cublasAlloc(b.size, elemSize, dB)
        JCublas.cublasAlloc(cSize, elemSize, dC)

        JCublas.cublasSetVector(a.size, elemSize, phA, 1, dA, 1)
        JCublas.cublasSetVector(b.size, elemSize, phB, 1, dB, 1)

        if (matrixMatrix) {
            val m = a.shape[0]
            val n = b.shape[1]
            val k = a.shape[1]

            val transA = if (transposedA) 't' else 'n'
            val transB = if (transposedB) 't' else 'n'

            val lda = if (transposedA) m else k
            val ldb = if (transposedB) k else n

            // multiplication order is swapped because cublas uses column-major storage
            if (a.dtype == DataType.FloatDataType)
                JCublas.cublasSgemm(
                    'n', 'n',
                    n, m, k,
                    1f, dB, n, dA, k, 0f, dC, n
                )
            else
                JCublas.cublasDgemm(
                    'n', 'n',
                    n, m, k,
                    1.0, dB, n, dA, k, 0.0, dC, n
                )
        } else {
            val transA = if (transposedA) 'n' else 't'

            var (m, n) = a.shape
            if (!transposedA)
                m = n.also { n = m }

            if (a.dtype == DataType.FloatDataType)
                JCublas.cublasSgemv(
                    't', a.shape[1], a.shape[0], 1f, dA, a.shape[1], dB, 1, 0f, dC, 1
                )
            else
                JCublas.cublasDgemv(
                    't', a.shape[1], a.shape[0], 1.0, dA, a.shape[1], dB, 1, 0.0, dC, 1
                )
        }

        JCublas.cublasGetVector(cSize, elemSize, dC, 1, phC, 1)

        JCublas.cublasFree(dA)
        JCublas.cublasFree(dB)
        JCublas.cublasFree(dC)

        return NDArray(hC, shape = shape, dtype = a.dtype, dim = b.dim)
    }

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.shape[0] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[0]} (dim 0) != ${b.shape[0]} (dim 0)"
        }

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val elemSize = a.dtype.itemSize

        val dA = Pointer()
        val dB = Pointer()
        val dC = Pointer()

        val phA: Pointer
        val phB: Pointer

        val (consistentA, _) = getConsistentOrTransposedConsistent(a)
        val (consistentB, _) = getConsistentOrTransposedConsistent(b)

        if (a.dtype == DataType.FloatDataType) {
            phA = Pointer.to(consistentA.data.getFloatArray())
            phB = Pointer.to(consistentB.data.getFloatArray())
        } else {
            phA = Pointer.to(consistentA.data.getDoubleArray())
            phB = Pointer.to(consistentB.data.getDoubleArray())
        }

        JCublas.cublasAlloc(a.size, elemSize, dA)
        JCublas.cublasAlloc(b.size, elemSize, dB)

        JCublas.cublasSetVector(a.size, elemSize, phA, 1, dA, 1)
        JCublas.cublasSetVector(b.size, elemSize, phB, 1, dB, 1)

        val result = if (a.dtype == DataType.FloatDataType)
            JCublas.cublasSdot(a.shape[0], dA, 1, dB, 1)
        else
            JCublas.cublasDdot(a.shape[0], dA, 1, dB, 1)

        JCublas.cublasFree(dA)
        JCublas.cublasFree(dB)
        JCublas.cublasFree(dC)

        return result as T
    }

}
