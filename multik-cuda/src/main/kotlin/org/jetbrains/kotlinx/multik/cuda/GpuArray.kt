package org.jetbrains.kotlinx.multik.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.runtime.JCuda
import jcuda.runtime.cudaError.cudaErrorMemoryAllocation
import jcuda.runtime.cudaMemcpyKind
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import java.lang.ref.Cleaner
import java.util.*
import java.util.concurrent.LinkedBlockingQueue


private val logger = KotlinLogging.logger {}

internal class GpuArray constructor(
    private val rawData : Any,
    val hostDataPtr : Pointer,
    val deviceDataPtr : Pointer,
    val byteSize : Long
) {
    var isLoaded = true
        private set

    fun copyFromGpu() {
        checkResult(JCuda.cudaMemcpy(hostDataPtr, deviceDataPtr, byteSize, cudaMemcpyKind.cudaMemcpyDeviceToHost))
    }

    fun free() {
        if (!isLoaded)
            throw IllegalStateException("Trying to free memory that it is already freed")

        logger.debug { "Freeing GPU memory. Data: $rawData, size: ${byteSizeToString(byteSize)}" }
        checkResult(JCuda.cudaFree(deviceDataPtr))

        isLoaded = false
    }
}

internal class GpuCache {
    fun assertAllLoaded(vararg arrays: GpuArray) {
        if (!arrays.all { it.isLoaded })
            throw OutOfMemoryError("Not all arrays have are loaded in the GPU memory")
    }

    fun fullCleanup() {
        cache.forEach { it.value.free() }
        cache.clear()

        deleteQueue.clear()
    }

    fun <T : Number, D : Dimension> getOrAlloc(array: MultiArray<T, D>, setMemory: Boolean = true): GpuArray {
        cleanup()
        return cache.getOrPut(array.data.data) { allocMemory(array, setMemory) }
    }

    companion object {
        private const val CACHE_INITIAL_CAPACITY = 16
        private const val CACHE_LOAD_FACTOR = 0.75f
    }

    private val cache = LinkedHashMap<Any, GpuArray>(CACHE_INITIAL_CAPACITY, CACHE_LOAD_FACTOR, true)

    private val deleteQueue = LinkedBlockingQueue<Any>()
    private val cleaner = Cleaner.create()

    private fun getGpuMemInfo(): String {
        val free = LongArray(1)
        val total = LongArray(1)

        checkResult(JCuda.cudaMemGetInfo(free, total))
        return "free: ${byteSizeToString(free[0])}, total: ${byteSizeToString(total[0])} MB"
    }

    private fun <T : Number, D : Dimension> allocMemory(array: MultiArray<T, D>, setMemory: Boolean): GpuArray {
        val elemSize = array.dtype.itemSize
        val rawData = array.data.data

        val deviceDataPtr = Pointer()
        val hostDataPtr = array.dtype.getDataPointer(array.data)
        val byteSize = elemSize.toLong() * array.size

        logger.debug { "Allocating array on GPU. Data: ${rawData}, size: ${byteSizeToString(byteSize)}" }

        while (true) {
            val result = JCuda.cudaMalloc(deviceDataPtr, byteSize)

            if (result == cudaErrorMemoryAllocation) {
                logger.trace { "Not enough GPU memory for allocation. Trying to free stale memory. Gpu Mem: {${getGpuMemInfo()}}" }

                val iterator = cache.iterator()
                if (iterator.hasNext()) {
                    iterator.next().value.free()
                    iterator.remove()
                } else {
                    throw CudaException("Insufficient GPU memory for the array")
                }
            } else
                break
        }

        if (setMemory)
            checkResult(JCuda.cudaMemcpy(deviceDataPtr, hostDataPtr, byteSize, cudaMemcpyKind.cudaMemcpyHostToDevice))

        cleaner.register(array.data) { deleteQueue.add(rawData) }

        return GpuArray(rawData, hostDataPtr, deviceDataPtr, byteSize)
    }

    private fun cleanup() {
        var firstRun = true
        while (deleteQueue.isNotEmpty()) {
            if (firstRun) {
                logger.trace { "Delete queue is not empty. Cleaning" }
                firstRun = false
            }

            val data = deleteQueue.poll()
            cache.remove(data)?.free()
        }
    }
}