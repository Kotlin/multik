/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT
import jcuda.jcublas.cublasOperation
import jcuda.runtime.JCuda
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.cuda.CudaEngine.contextHandle
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object CudaLinAlg : LinAlg {

    override val linAlgEx: LinAlgEx
        get() = CudaLinAlgEx

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
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

        val elemSize = a.dtype.itemSize

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

        val hC = initMemoryView<T>(cSize, a.dtype)

        val (consistentA, transposedA) = getConsistentOrTransposedConsistent(a)
        val (consistentB, transposedB) = getConsistentOrTransposedConsistent(b)

        if (a.dtype == DataType.FloatDataType) {
            phA = Pointer.to(consistentA.data.getFloatArray())
            phB = Pointer.to(consistentB.data.getFloatArray())
            phC = Pointer.to(hC.getFloatArray())
        } else {
            phA = Pointer.to(consistentA.data.getDoubleArray())
            phB = Pointer.to(consistentB.data.getDoubleArray())
            phC = Pointer.to(hC.getDoubleArray())
        }

        JCuda.cudaMalloc(dA, elemSize.toLong() * a.size)
        JCuda.cudaMalloc(dB, elemSize.toLong() * b.size)
        JCuda.cudaMalloc(dC, elemSize.toLong() * cSize)

        JCublas2.cublasSetVector(a.size, elemSize, phA, 1, dA, 1)
        JCublas2.cublasSetVector(b.size, elemSize, phB, 1, dB, 1)

        val zeroPtr = a.dtype.getZeroPointer()
        val onePtr = a.dtype.getOnePointer()

        if (matrixMatrix) {
            val m = a.shape[0]
            val n = b.shape[1]
            val k = a.shape[1]

            val transA = if (transposedA) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
            val transB = if (transposedB) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N

            val lda = if (transposedA) m else k
            val ldb = if (transposedB) k else n

            val type = a.dtype.getCudaType()
            val computeType = a.dtype.getDefaultComputeType()

            // multiplication order is swapped because cublas uses column-major storage
            JCublas2.cublasGemmEx_new(
                contextHandle, transB, transA, n, m, k,
                onePtr, dB, type, ldb, dA, type, lda, zeroPtr, dC, type, n,
                computeType, CUBLAS_GEMM_DEFAULT
            )
        } else {
            val transA = if (transposedA) cublasOperation.CUBLAS_OP_N else cublasOperation.CUBLAS_OP_T

            var (m, n) = a.shape
            if (!transposedA)
                m = n.also { n = m }

            if (a.dtype == DataType.FloatDataType)
                JCublas2.cublasSgemv(contextHandle, transA, m, n, onePtr, dA, m, dB, 1, zeroPtr, dC, 1)
            else
                JCublas2.cublasDgemv(contextHandle, transA, m, n, onePtr, dA, m, dB, 1, zeroPtr, dC, 1)
        }

        JCublas2.cublasGetVector(cSize, elemSize, dC, 1, phC, 1)

        JCuda.cudaFree(dA)
        JCuda.cudaFree(dB)
        JCuda.cudaFree(dC)

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
        val resultPtr: Pointer

        val (consistentA, _) = getConsistentOrTransposedConsistent(a)
        val (consistentB, _) = getConsistentOrTransposedConsistent(b)

        val result = initMemoryView<T>(1, a.dtype)

        if (a.dtype == DataType.FloatDataType) {
            phA = Pointer.to(consistentA.data.getFloatArray())
            phB = Pointer.to(consistentB.data.getFloatArray())
            resultPtr = Pointer.to(result.getFloatArray())
        } else {
            phA = Pointer.to(consistentA.data.getDoubleArray())
            phB = Pointer.to(consistentB.data.getDoubleArray())
            resultPtr = Pointer.to(result.getDoubleArray())
        }

        JCuda.cudaMalloc(dA, elemSize.toLong() * a.size)
        JCuda.cudaMalloc(dB, elemSize.toLong() * b.size)

        JCublas2.cublasSetVector(a.size, elemSize, phA, 1, dA, 1)
        JCublas2.cublasSetVector(b.size, elemSize, phB, 1, dB, 1)

        val type = a.dtype.getCudaType()

        JCublas2.cublasDotEx(contextHandle, a.shape[0], dA, type, 1, dB, type, 1, resultPtr, type, type)

        JCuda.cudaFree(dA)
        JCuda.cudaFree(dB)
        JCuda.cudaFree(dC)

        return result[0]
    }

    // Note: NDArray.transpose() only creates a lightweight view
    private fun <T : Number, D : Dim2> isTransposedConsistent(x: MultiArray<T, D>): Boolean =
        x.transpose().consistent


    /**
     * Helper function used to get consistent data from [MultiArray]
     *
     * First value in returned pair - [MultiArray] that is consistent or
     * consistent when transposed
     *
     * Second value in returned pair - transposition flag - indicates whether
     * returned [MultiArray] should be transposed in order to be consistent
     *
     * @return pair of [MultiArray] and transposition flag
     */
    private fun <T : Number, D : Dim2> getConsistentOrTransposedConsistent(x: MultiArray<T, D>): Pair<MultiArray<T, D>, Boolean> =
        when {
            x.consistent -> x to false
            x.dim.d == 2 && isTransposedConsistent(x) -> x to true
            else -> x.deepCopy() to false
        }
}
