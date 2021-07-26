package org.jetbrains.kotlinx.multik.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.jcublas.JCublas2
import jcuda.runtime.JCuda
import jcuda.runtime.cudaError
import jcuda.runtime.cudaError.cudaErrorMemoryAllocation
import jcuda.runtime.cudaError.cudaSuccess
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import java.util.concurrent.LinkedBlockingQueue
import kotlin.math.pow


private val logger = KotlinLogging.logger {}

internal class GpuArray private constructor(
    val hostDataPtr: Pointer,
    val deviceDataPtr: Pointer
) {
    companion object {
        private const val CACHE_INITIAL_CAPACITY = 16
        private const val CACHE_LOAD_FACTOR = 0.75f

        private val cache = LinkedHashMap<Any, GpuArray>(CACHE_INITIAL_CAPACITY, CACHE_LOAD_FACTOR, true)
        private val deleteQueue = LinkedBlockingQueue<Any>()

        private fun checkResult(result: Int) {
            if (result != cudaSuccess)
                throw CudaException(cudaError.stringFor(result))
        }

        private fun byteSizeToString(bytes: Long): String {
            val unit = 1024
            if (bytes < unit) return "$bytes B"
            val exp = (Math.log(bytes.toDouble()) / Math.log(unit.toDouble())).toInt()
            val pre = "KMGTPE"[exp - 1].toString() + "i"
            return String.format("%.1f %sB", bytes / unit.toDouble().pow(exp.toDouble()), pre)
        }

        private fun getGpuMemInfo(): String {
            val free = LongArray(1)
            val total = LongArray(1)

            checkResult(JCuda.cudaMemGetInfo(free, total))
            return "free: ${byteSizeToString(free[0])}, total: ${byteSizeToString(total[0])} MB"
        }

        private fun freeMemory(data: Any) {
            val gpuArray = cache.remove(data)

            if (gpuArray != null) {
                logger.debug { "Freeing GPU memory. Data: $data" }
                checkResult(JCuda.cudaFree(gpuArray.deviceDataPtr))
            }
        }

        private fun <T : Number, D : Dimension> allocMemory(array: MultiArray<T, D>, setMemory: Boolean): GpuArray {
            val elemSize = array.dtype.itemSize

            val deviceDataPtr = Pointer()
            val hostDataPtr = array.dtype.getDataPointer(array.data)

            logger.debug { "Allocating array on GPU. Shape: ${array.shape.contentToString()}, " +
                    "dtype: ${array.dtype.name}, data: ${array.data.data}, " +
                    "size: ${byteSizeToString(elemSize.toLong() * array.size)}" }

            while (true) {
                val result = JCuda.cudaMalloc(deviceDataPtr, elemSize.toLong() * array.size)

                if (result == cudaErrorMemoryAllocation) {
                    logger.trace { "Not enough GPU memory for allocation. Trying to free stale memory. Gpu Mem: {${getGpuMemInfo()}}" }

                    val iterator = cache.iterator()
                    if (iterator.hasNext()) {
                        val entry = iterator.next()
                        freeMemory(entry.key)
                    } else {
                        throw CudaException("Insufficient GPU memory for the array")
                    }
                } else
                    break
            }

            if (setMemory)
                checkResult(JCublas2.cublasSetVector(array.size, array.dtype.itemSize, hostDataPtr, 1, deviceDataPtr, 1))

            array.data.__onFinalizeFunc = { deleteQueue.add(it.data) }

            return GpuArray(hostDataPtr, deviceDataPtr)
        }

        private fun cleanup() {
            var firstRun = true
            while (deleteQueue.isNotEmpty()) {
                if (firstRun) {
                    logger.trace { "Delete queue is not empty. Cleaning" }
                    firstRun = false
                }

                val data = deleteQueue.poll()
                freeMemory(data)
            }
        }

        fun <T : Number, D : Dimension> getOrAlloc(array: MultiArray<T, D>, setMemory: Boolean = true): GpuArray {
            cleanup()
            return cache.getOrPut(array.data.data) { allocMemory(array, setMemory) }
        }
    }
}