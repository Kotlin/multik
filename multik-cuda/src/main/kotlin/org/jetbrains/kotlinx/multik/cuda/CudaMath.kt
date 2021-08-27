/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasOperation
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind
import org.jetbrains.kotlinx.multik.api.Math
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object CudaMath : Math {
    override fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMaxDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD2(a: MultiArray<T, D2>, axis: Int): NDArray<Int, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD3(a: MultiArray<T, D3>, axis: Int): NDArray<Int, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinD4(a: MultiArray<T, D4>, axis: Int): NDArray<Int, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> argMinDN(a: MultiArray<T, DN>, axis: Int): NDArray<Int, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> max(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> maxDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> min(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> minDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD2(a: MultiArray<T, D2>, axis: Int): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD3(a: MultiArray<T, D3>, axis: Int): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumD4(a: MultiArray<T, D4>, axis: Int): NDArray<T, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> sumDN(a: MultiArray<T, DN>, axis: Int): NDArray<T, D4> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D> {
        TODO("Not yet implemented")
    }

    private fun <T : Number, D : Dimension> combine(a: MultiArray<T, D>, b: MultiArray<T, D>, bSign: Int): NDArray<T, D> {
        require(a.shape contentEquals b.shape)

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        val (result, gC) = context.cache.alloc<T, D>(a.size, a.dtype, a.shape, a.dim)

        context.cache.assertAllLoaded(gA, gB, gC)

        val beta = a.dtype.singleValuePointer(bSign)

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

            checkResult(
                JCublas2.cublasAxpyEx(
                context.handle, a.size, beta, type,
                gB.deviceDataPtr, type, 1,
                gC.deviceDataPtr, type, 1, type
            ))
        }

        return result
    }

    infix fun <T : Number, D : Dimension> MultiArray<T, D>.add(other: MultiArray<T, D>): NDArray<T, D> = combine(this, other, 1)

    infix fun <T : Number, D : Dimension> MultiArray<T, D>.subtract(other: MultiArray<T, D>): NDArray<T, D> = combine(this, other, -1)

}