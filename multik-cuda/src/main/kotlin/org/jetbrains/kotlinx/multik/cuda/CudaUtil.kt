package org.jetbrains.kotlinx.multik.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.cudaDataType
import jcuda.jcublas.cublasComputeType
import jcuda.runtime.JCuda
import jcuda.runtime.cudaError
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.ImmutableMemoryView
import kotlin.math.pow

internal val DZeroPointer = Pointer.to(doubleArrayOf(0.0))
internal val DOnePointer = Pointer.to(doubleArrayOf(1.0))

internal val FZeroPointer = Pointer.to(floatArrayOf(0F))
internal val FOnePointer = Pointer.to(floatArrayOf(1F))

fun getGpuMemInfo(): Pair<Long, Long> {
    val free = LongArray(1)
    val total = LongArray(1)

    checkResult(JCuda.cudaMemGetInfo(free, total))

    return free[0] to total[0]
}

fun byteSizeToString(bytes: Long): String {
    val unit = 1024
    if (bytes < unit) return "$bytes B"
    val exp = (Math.log(bytes.toDouble()) / Math.log(unit.toDouble())).toInt()
    val pre = "KMGTPE"[exp - 1].toString() + "i"
    return String.format("%.1f %sB", bytes / unit.toDouble().pow(exp.toDouble()), pre)
}

internal fun DataType.getDefaultComputeType(): Int =
    when (this) {
        DataType.FloatDataType -> cublasComputeType.CUBLAS_COMPUTE_32F
        DataType.DoubleDataType -> cublasComputeType.CUBLAS_COMPUTE_64F
        else -> throw UnsupportedOperationException()
    }

internal fun DataType.getCudaType(): Int =
    when (this) {
        DataType.FloatDataType -> cudaDataType.CUDA_R_32F
        DataType.DoubleDataType -> cudaDataType.CUDA_R_64F
        else -> throw UnsupportedOperationException()
    }

internal fun DataType.getZeroPointer(): Pointer =
    when (this) {
        DataType.FloatDataType -> FZeroPointer
        DataType.DoubleDataType -> DZeroPointer
        else -> throw UnsupportedOperationException()
    }

internal fun DataType.getOnePointer(): Pointer =
    when (this) {
        DataType.FloatDataType -> FOnePointer
        DataType.DoubleDataType -> DOnePointer
        else -> throw UnsupportedOperationException()
    }

internal fun <T> DataType.getDataPointer(data: ImmutableMemoryView<T>): Pointer =
    when (this) {
        DataType.FloatDataType -> Pointer.to(data.getFloatArray())
        DataType.DoubleDataType -> Pointer.to(data.getDoubleArray())
        else -> throw UnsupportedOperationException("Unsupported data type: $this")
    }

internal fun checkResult(result: Int) {
    if (result != cudaError.cudaSuccess)
        throw CudaException(cudaError.stringFor(result))
}
