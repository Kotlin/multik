/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT
import jcuda.jcublas.cublasOperation
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

        val shape = if (matrixMatrix)
            intArrayOf(a.shape[0], b.shape[1])
        else
            intArrayOf(a.shape[0])

        val cSize = shape.reduce(Int::times)

        val (consistentA, transposedA) = getConsistentOrTransposedConsistent(a)
        val (consistentB, transposedB) = getConsistentOrTransposedConsistent(b)

//        val hC = initMemoryView<T>(cSize, a.dtype)

        val gA = GpuArray.getOrAlloc(consistentA)
        val gB = GpuArray.getOrAlloc(consistentB)

        val result = NDArray(initMemoryView<T>(cSize, a.dtype), shape = shape, dtype = a.dtype, dim = b.dim)
        val gC = GpuArray.getOrAlloc(result, setMemory = false)

        //TODO check if all arrays are loaded

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
                onePtr, gB.deviceDataPtr, type, ldb, gA.deviceDataPtr, type, lda, zeroPtr, gC.deviceDataPtr, type, n,
                computeType, CUBLAS_GEMM_DEFAULT
            )
        } else {
            val transA = if (transposedA) cublasOperation.CUBLAS_OP_N else cublasOperation.CUBLAS_OP_T

            var (m, n) = a.shape
            if (!transposedA)
                m = n.also { n = m }

            if (a.dtype == DataType.FloatDataType)
                JCublas2.cublasSgemv(contextHandle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1)
            else
                JCublas2.cublasDgemv(contextHandle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1)
        }

        JCublas2.cublasGetVector(cSize, a.dtype.itemSize, gC.deviceDataPtr, 1, gC.hostDataPtr, 1)

        return result
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

        val (consistentA, _) = getConsistentOrTransposedConsistent(a)
        val (consistentB, _) = getConsistentOrTransposedConsistent(b)

        val gA = GpuArray.getOrAlloc(consistentA)
        val gB = GpuArray.getOrAlloc(consistentB)

        //TODO check if all arrays are loaded

        val result = initMemoryView<T>(1, a.dtype)
        val resultPtr = a.dtype.getDataPointer(result)
        val type = a.dtype.getCudaType()

        JCublas2.cublasDotEx(contextHandle, a.shape[0], gA.deviceDataPtr, type, 1, gB.deviceDataPtr, type, 1, resultPtr, type, type)

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
