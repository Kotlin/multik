/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT
import jcuda.jcublas.cublasOperation
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind
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

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        val (result, gC) = context.cache.alloc<T, D>(cSize, a.dtype, shape, b.dim)

        context.cache.assertAllLoaded(gA, gB, gC)

        val zeroPtr = a.dtype.getZeroPointer()
        val onePtr = a.dtype.getOnePointer()

        if (matrixMatrix) {
            val m = a.shape[0]
            val n = b.shape[1]
            val k = a.shape[1]

            val transA = if (gA.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
            val transB = if (gB.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N

            val lda = if (gA.transposed) m else k
            val ldb = if (gB.transposed) k else n

            val type = a.dtype.getCudaType()
            val computeType = a.dtype.getDefaultComputeType()

            // multiplication order is swapped because cublas uses column-major storage
            checkResult(JCublas2.cublasGemmEx_new(
                context.handle, transB, transA, n, m, k,
                onePtr, gB.deviceDataPtr, type, ldb, gA.deviceDataPtr, type, lda, zeroPtr, gC.deviceDataPtr, type, n,
                computeType, CUBLAS_GEMM_DEFAULT
            ))
        } else {
            val transA = if (gA.transposed) cublasOperation.CUBLAS_OP_N else cublasOperation.CUBLAS_OP_T

            var (m, n) = a.shape
            if (!gA.transposed)
                m = n.also { n = m }

            val func = if (a.dtype == DataType.DoubleDataType) JCublas2::cublasDgemv else JCublas2::cublasSgemv

            checkResult(func(context.handle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1))
        }

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

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        context.cache.assertAllLoaded(gA, gB)

        val result = initMemoryView<T>(1, a.dtype)
        val resultPtr = a.dtype.getDataPointer(result)
        val type = a.dtype.getCudaType()

        checkResult(JCublas2.cublasDotEx(context.handle, a.shape[0], gA.deviceDataPtr, type, 1, gB.deviceDataPtr, type, 1, resultPtr, type, type))

        return result[0]
    }

    fun <T : Number, D : Dim2> add(a: MultiArray<T, D>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape contentEquals b.shape)

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        val (result, gC) = context.cache.alloc<T, D>(a.size, a.dtype, a.shape, a.dim)

        context.cache.assertAllLoaded(gA, gB, gC)

        if (a.dim.d == 2) {
            val transA = if (gA.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
            val transB = if (gB.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N

            val one = a.dtype.getOnePointer()

            val ldA = if (gA.transposed) a.shape[0] else a.shape[1]
            val ldB = if (gB.transposed) b.shape[0] else b.shape[1]

            val func = if (a.dtype == DataType.DoubleDataType) JCublas2::cublasDgeam else JCublas2::cublasSgeam

            checkResult(
                func(
                    context.handle,
                    transA, transB, a.shape[1], a.shape[0],
                    one, gA.deviceDataPtr, ldA,
                    one, gB.deviceDataPtr, ldB,
                    gC.deviceDataPtr, a.shape[1]
                )
            )
        } else {
            val type = a.dtype.getCudaType()

            checkResult(JCuda.cudaMemcpy(gC.deviceDataPtr, gB.deviceDataPtr, gB.byteSize, cudaMemcpyKind.cudaMemcpyDeviceToDevice))

            checkResult(JCublas2.cublasAxpyEx(
                context.handle, a.size, a.dtype.getOnePointer(), type,
                gA.deviceDataPtr, type, 1,
                gC.deviceDataPtr, type, 1, type
            ))
        }

        return result
    }
}
