/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda.linalg

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasOperation
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.cuda.*
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

    private enum class CombineType { ADD, SUBTRACT }

    private fun <T : Number, D : Dimension> combine(a: MultiArray<T, D>, b: MultiArray<T, D>, combineType: CombineType): NDArray<T, D> {
        require(a.shape contentEquals b.shape)

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        val (result, gC) = context.cache.alloc<T, D>(a.size, a.dtype, a.shape, a.dim)

        context.cache.assertAllLoaded(gA, gB, gC)

        val beta = if (combineType == CombineType.ADD)
            a.dtype.getOnePointer()
        else
            a.dtype.singleValuePointer(-1)

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
                    beta, gB.deviceDataPtr, ldB,
                    gC.deviceDataPtr, a.shape[1]
                )
            )
        } else {
            val type = a.dtype.getCudaType()

            checkResult(JCuda.cudaMemcpy(gC.deviceDataPtr, gA.deviceDataPtr, gA.byteSize, cudaMemcpyKind.cudaMemcpyDeviceToDevice))

            checkResult(JCublas2.cublasAxpyEx(
                context.handle, a.size, beta, type,
                gB.deviceDataPtr, type, 1,
                gC.deviceDataPtr, type, 1, type
            ))
        }

        return result
    }

    fun <T : Number, D : Dimension> add(a: MultiArray<T, D>, b: MultiArray<T, D>): NDArray<T, D> = combine(a, b, CombineType.ADD)

    fun <T : Number, D : Dimension> subtract(a: MultiArray<T, D>, b: MultiArray<T, D>): NDArray<T, D> = combine(a, b, CombineType.SUBTRACT)

}
