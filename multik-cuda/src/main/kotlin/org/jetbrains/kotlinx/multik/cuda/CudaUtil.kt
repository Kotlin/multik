package org.jetbrains.kotlinx.multik.cuda

import jcuda.Pointer
import jcuda.cudaDataType
import jcuda.jcublas.cublasComputeType
import org.jetbrains.kotlinx.multik.ndarray.data.DataType

internal val DZeroPointer = Pointer.to(doubleArrayOf(0.0))
internal val DOnePointer = Pointer.to(doubleArrayOf(1.0))

internal val FZeroPointer = Pointer.to(floatArrayOf(0F))
internal val FOnePointer = Pointer.to(floatArrayOf(1F))

fun DataType.getDefaultComputeType(): Int =
    when (this) {
        DataType.FloatDataType -> cublasComputeType.CUBLAS_COMPUTE_32F
        DataType.DoubleDataType -> cublasComputeType.CUBLAS_COMPUTE_64F
        else -> throw UnsupportedOperationException()
    }

fun DataType.getCudaType(): Int =
    when (this) {
        DataType.FloatDataType -> cudaDataType.CUDA_R_32F
        DataType.DoubleDataType -> cudaDataType.CUDA_R_64F
        else -> throw UnsupportedOperationException()
    }

fun DataType.getZeroPointer(): Pointer =
    when (this) {
        DataType.FloatDataType -> FZeroPointer
        DataType.DoubleDataType -> DZeroPointer
        else -> throw UnsupportedOperationException()
    }

fun DataType.getOnePointer(): Pointer =
    when (this) {
        DataType.FloatDataType -> FOnePointer
        DataType.DoubleDataType -> DOnePointer
        else -> throw UnsupportedOperationException()
    }
