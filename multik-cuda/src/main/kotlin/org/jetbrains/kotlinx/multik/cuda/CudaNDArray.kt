package org.jetbrains.kotlinx.multik.cuda

import jcuda.Pointer
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.MemoryView
import java.util.*

private val logger = KotlinLogging.logger {}

private class MyLinkedList<T> : Iterable<T> {
    sealed class Tag<T> {
        abstract val parent: MyLinkedList<T>?
    }

    val isEmpty
        get() = size == 0

    val first
        get() = head!!.value

    fun add(value: T): Tag<T> {
        return addNode(Node(value))
    }

    fun popFirst(): T {
        return removeNode(head!!).value
    }

    fun placeLast(tag: Tag<T>) {
        @Suppress("UNCHECKED_CAST")
        addNode(removeNode(tag as Node<T>))
    }

    override fun iterator(): Iterator<T> {
        return object : Iterator<T> {
            var current = head

            override fun hasNext(): Boolean = current != null

            override fun next(): T {
                val result = current!!.value
                current = current!!.next

                return result
            }
        }
    }

    override fun toString(): String {
        val builder = StringJoiner(", ", "[", "]")

        for (x in this)
            builder.add(x.toString())

        return builder.toString()
    }

    private class Node<T>(var value: T) : Tag<T>() {
        var next: Node<T>? = null
        var prev: Node<T>? = null

        override var parent: MyLinkedList<T>? = null
    }

    private var head: Node<T>? = null
    private var tail: Node<T>? = null

    private var size: Int = 0

    private fun addNode(node: Node<T>) : Node<T> {
        require(node.parent == null)

        if (size == 0) {
            head = node
        } else {
            tail!!.next = node
            node.prev = tail
            tail = node
        }

        tail = node
        node.parent = this

        size++

        return node
    }

    private fun removeNode(node: Node<T>): Node<T> {
        require(node.parent == this)

        if (node == head)
            head = node.next

        if (node == tail)
            tail = node.prev

        node.apply {
            prev?.next = next
            next?.prev = prev

            prev = null
            next = null
        }

        node.parent = null

        size -= 1

        return node
    }
}

fun main() {
    val list = MyLinkedList<Int>()

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
}

private object CudaCache {
    val cache = MyLinkedList<MemoryLocation>()
}

internal class MemoryLocation(val hostDataPtr: Pointer, val size: Int) {
    val deviceDataPtr = Pointer()

    var isLoaded: Boolean = false
        private set

    fun alloc(set: Boolean = true) {
        if (isLoaded) {
            logger.warn { "Trying to allocate memory that is already allocated" }
            return
        }

        JCuda.cudaMalloc(deviceDataPtr, size.toLong())
        if (set) {
            checkResult(JCuda.cudaMemcpy(deviceDataPtr, hostDataPtr, size.toLong(), cudaMemcpyHostToDevice))
        }
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
        isLoaded = false
    }
}

class CudaMemoryView<T : Number>(private val baseView: MemoryView<T>) : MemoryView<T>() {
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
//        throw UnsupportedOperationException()
    }

    override fun copyOf(): MemoryView<T> {
        return CudaMemoryView(baseView.copyOf())
    }

    internal val memoryLocation = MemoryLocation(getHostDataPointer(), size)

    internal fun getHostDataPointer() =
        dtype.getDataPointer(this)

    override fun finalize() {

    }
}

//class CudaFloatMemoryView(data: FloatArray) : MemoryViewFloatArray(data) {
//    internal val memoryLocation = MemoryLocation()
//
//    override fun copyOf(): MemoryView<Float> {
//        return CudaFloatMemoryView(data.copyOf())
//    }
//}

//class CudaNDArray<T: Number, V: Dimension> : NDArray<T, V>() {
//    class ArrayInfo {
//        val deviceDataPtr: Pointer? = null
//    }
//
//    private val arrayInfo = ArrayInfo()
//
//
//    companion object {
//        fun ababca() {
//            LinkedList<ArrayInfo>()
//        }
//    }
//}