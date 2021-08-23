package org.jetbrains.kotlinx.multik.cuda

import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.ndarray.data.*

private val logger = KotlinLogging.logger {}

object DeferredData

internal class CudaMemoryView<T>(override var size: Int, override val dtype: DataType, private val gpuArray: GpuArray) : MemoryView<T>() {

    //Data is loaded to heap lazily
    override var data: Any = DeferredData
        private set

    override var lastIndex: Int = size - 1
    override var indices: IntRange = IntRange(0, lastIndex)

    @Suppress("UNCHECKED_CAST")
    override fun get(index: Int): T {
        loadToHost()

        return when (dtype) {
            DataType.FloatDataType -> (data as FloatArray)[index]
            DataType.DoubleDataType -> (data as DoubleArray)[index]
            else -> throw IllegalArgumentException("Unsupported datatype")
        } as T
    }

    @Suppress("UNCHECKED_CAST")
    override fun iterator(): Iterator<T> {
        loadToHost()

        return when (dtype) {
            DataType.FloatDataType -> (data as FloatArray).iterator()
            DataType.DoubleDataType -> (data as DoubleArray).iterator()
            else -> throw IllegalArgumentException("Unsupported datatype")
        } as Iterator<T>
    }

    override fun set(index: Int, value: T) {
        throw UnsupportedOperationException()
    }

    @Suppress("UNCHECKED_CAST")
    override fun copyOf(): MemoryView<T> {
        loadToHost()

        return when (dtype) {
            DataType.FloatDataType -> MemoryViewFloatArray((data as FloatArray).copyOf())
            DataType.DoubleDataType -> MemoryViewDoubleArray((data as DoubleArray).copyOf())
            else -> throw IllegalArgumentException("Unsupported datatype")
        } as MemoryView<T>
    }

    override fun getFloatArray(): FloatArray {
        if (dtype != DataType.FloatDataType)
            throw UnsupportedOperationException()

        loadToHost()

        return data as FloatArray
    }

    override fun getDoubleArray(): DoubleArray {
        if (dtype != DataType.DoubleDataType)
            throw UnsupportedOperationException()

        loadToHost()

        return data as DoubleArray
    }

    internal fun loadToHost() {
        if (data == DeferredData) {
            logger.trace { "Loading deferred data to the host" }

            data = when (dtype) {
                DataType.FloatDataType -> FloatArray(size)
                DataType.DoubleDataType -> DoubleArray(size)
                else -> throw IllegalArgumentException("Unsupported datatype")
            }

            gpuArray.transferFromGpu(this.getDataPointer())
        }
    }
}