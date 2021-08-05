/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT
import jcuda.jcublas.cublasOperation
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
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

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(consistentA)
        val gB = context.cache.getOrAlloc(consistentB)

        val result = NDArray(initMemoryView<T>(cSize, a.dtype), shape = shape, dtype = a.dtype, dim = b.dim)
        val gC = context.cache.getOrAlloc(result, setMemory = false)

        context.cache.assertAllLoaded(gA, gB, gC)

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
                context.handle, transB, transA, n, m, k,
                onePtr, gB.deviceDataPtr, type, ldb, gA.deviceDataPtr, type, lda, zeroPtr, gC.deviceDataPtr, type, n,
                computeType, CUBLAS_GEMM_DEFAULT
            )
        } else {
            val transA = if (transposedA) cublasOperation.CUBLAS_OP_N else cublasOperation.CUBLAS_OP_T

            var (m, n) = a.shape
            if (!transposedA)
                m = n.also { n = m }

            if (a.dtype == DataType.FloatDataType)
                JCublas2.cublasSgemv(context.handle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1)
            else
                JCublas2.cublasDgemv(context.handle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1)
        }

        gC.copyFromGpu()

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

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(consistentA)
        val gB = context.cache.getOrAlloc(consistentB)

        context.cache.assertAllLoaded(gA, gB)

        val result = initMemoryView<T>(1, a.dtype)
        val resultPtr = a.dtype.getDataPointer(result)
        val type = a.dtype.getCudaType()

        JCublas2.cublasDotEx(context.handle, a.shape[0], gA.deviceDataPtr, type, 1, gB.deviceDataPtr, type, 1, resultPtr, type, type)

        return result[0]
    }

    fun <T : Number, D : Dim2> add(a: MultiArray<T, D>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape contentEquals b.shape)

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val context = CudaEngine.getContext()

        val aIsConsistent = a.consistent
        val bIsConsistent = b.consistent

        val (x: MultiArray<T, D>, y: MultiArray<T, D>) =
            if (aIsConsistent && !bIsConsistent)
                a to b.deepCopy()
            else if (!aIsConsistent && bIsConsistent)
                b to a.deepCopy()
            else if (!aIsConsistent && !bIsConsistent)
                a.deepCopy() to b.deepCopy()
            else
                a to b.deepCopy()

        val gX = context.cache.getOrAlloc(x)
        val gY = context.cache.getOrAlloc(y)

        val type = a.dtype.getCudaType()

        JCublas2.cublasAxpyEx(context.handle, a.size, a.dtype.getOnePointer(), type,
            gX.deviceDataPtr, type, 1,
            gY.deviceDataPtr, type, 1, type)

        gY.copyFromGpu()

        return y as NDArray<T, D>
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
