package org.jetbrains.kotlinx.multik.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.ImmutableMemoryView
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryView

private val logger = KotlinLogging.logger {}

/*fun main() {
    val list = LinkedCache<Int>()

    println(list.toString())

    list.add(1)
    val second = list.add(2)
    list.add(3)
    list.add(4)

    println(list.toString())

    list.placeLast(second)
    println(list.toString())
    list.popFirst()
    list.popFirst()
    list.popFirst()
    list.popFirst()
//    list.placeLast(second)

    println(list.toString())
}*/

private val GpuMemoryCache = LinkedCache<MemoryLocation>()

internal class MemoryLocation(
    val hostDataPtr: Pointer,
    val size: Int
) {
    val deviceDataPtr = Pointer()
    var cacheTag : LinkedCache.Tag<MemoryLocation>? = null

    val isLoaded: Boolean = cacheTag != null

    fun alloc(set: Boolean = true) {
        if (isLoaded) {
            logger.warn { "Trying to allocate memory that is already allocated" }
            return
        }

        JCuda.cudaMalloc(deviceDataPtr, size.toLong())
        if (set) {
            checkResult(JCuda.cudaMemcpy(deviceDataPtr, hostDataPtr, size.toLong(), cudaMemcpyHostToDevice))
        }

        cacheTag = GpuMemoryCache.add(this)
    }

    fun copyFromGpu() {
        checkResult(JCuda.cudaMemcpy(hostDataPtr, deviceDataPtr, size.toLong(), cudaMemcpyDeviceToHost))
    }

    fun free() {
        if (!isLoaded) {
            logger.warn { "Trying to free memory that is not loaded" }
            return
        }

        checkResult(JCuda.cudaFree(deviceDataPtr))

        if (cacheTag!!.parent != null) {
            GpuMemoryCache.remove(cacheTag!!)
        }
    }
}

class CudaMemoryView<T>(private val baseView: MemoryView<T>) : MemoryView<T>() {
    init {
        require(baseView.dtype == DataType.FloatDataType || baseView.dtype == DataType.DoubleDataType)
    }

    override val data = baseView.data
    override val dtype = baseView.dtype

    override var size: Int = baseView.size
    override var indices: IntRange = baseView.indices
    override var lastIndex: Int = baseView.lastIndex

    override fun get(index: Int): T = baseView[index]

    override fun iterator(): Iterator<T> = baseView.iterator()

    override fun set(index: Int, value: T) {
        throw UnsupportedOperationException()
    }

    override fun copyOf(): MemoryView<T> {
        return CudaMemoryView(baseView.copyOf())
    }

    internal val memoryLocation = MemoryLocation(dtype.getDataPointer(this), size)

    /*override fun finalize() {
        memoryLocation.free()
    }*/

    override fun copyInto(
        destination: ImmutableMemoryView<T>,
        destinationOffset: Int,
        startIndex: Int,
        endIndex: Int
    ): MemoryView<T> {
        return baseView.copyInto(destination, destinationOffset, startIndex, endIndex)
    }
}
