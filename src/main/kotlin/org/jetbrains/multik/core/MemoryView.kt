package org.jetbrains.multik.core

import java.nio.*

sealed class MemoryView<T : Number> {

    public abstract fun duplicate(position: Int = 0, limit: Int): MemoryView<T>

    public abstract fun sliceInPlace(position: Int = 0, limit: Int): Unit

    abstract operator fun get(index: Int): T

    abstract fun get(): T

    open operator fun get(dst: ByteArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open operator fun get(dst: ShortArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open operator fun get(dst: IntArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open operator fun get(dst: LongArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open operator fun get(dst: FloatArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open operator fun get(dst: DoubleArray, offset: Int = 0, length: Int = dst.size): Unit =
        throw IllegalStateException("")

    open fun put(src: Array<out T>): Unit = throw IllegalStateException("")

    open fun put(src: ByteArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    open fun put(src: ShortArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    open fun put(src: IntArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    open fun put(src: LongArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    open fun put(src: FloatArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    open fun put(src: DoubleArray, offset: Int = 0, length: Int = src.size): Unit = throw IllegalStateException("")

    abstract fun put(el: T): Unit

    abstract fun put(el: Int): Unit

    abstract fun put(el: Double): Unit

    abstract fun put(index: Int, el: T): Unit

    operator fun set(index: Int, el: T): Unit {
        put(index, el)
    }

    abstract fun <E : Buffer> put(src: E): Unit

    abstract fun <E : Iterable<T>> put(src: E): Unit

    abstract fun rewind(): Unit

    abstract fun getData(): Buffer

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        return true
    }

    override fun hashCode(): Int {
        return javaClass.hashCode()
    }
}

internal class MemoryViewByte(private val data: ByteBuffer) : MemoryView<Byte>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Byte> {
        val ret = MemoryViewByte(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    override fun get(index: Int): Byte = data[index]

    override fun get(): Byte = data.get()


    override operator fun get(dst: ByteArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Byte>) {
        data.put(src.toByteArray())
    }

    override fun put(src: ByteArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Byte) {
        data.put(el)
    }

    override fun put(el: Int) {
        data.put(el.toByte())
    }

    override fun put(el: Double) {
        data.put(el.toByte())
    }

    override fun put(index: Int, el: Byte): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is ByteBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Byte>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): ByteBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewByte

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

internal class MemoryViewShort(private val data: ShortBuffer) : MemoryView<Short>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Short> {
        val ret = MemoryViewShort(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    override fun get(index: Int): Short = data[index]

    override fun get(): Short = data.get()


    override operator fun get(dst: ShortArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Short>) {
        data.put(src.toShortArray())
    }

    override fun put(src: ShortArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Short) {
        data.put(el)
    }

    override fun put(el: Int) {
        data.put(el.toShort())
    }

    override fun put(el: Double) {
        data.put(el.toShort())
    }

    override fun put(index: Int, el: Short): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is ShortBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Short>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): ShortBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewShort

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

internal class MemoryViewInt(private val data: IntBuffer) : MemoryView<Int>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Int> {
        val ret = MemoryViewInt(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    public override fun get(index: Int): Int = data[index]

    override fun get(): Int = data.get()


    override operator fun get(dst: IntArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Int>) {
        data.put(src.toIntArray())
    }

    override fun put(src: IntArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Int) {
        data.put(el)
    }

    override fun put(el: Double) {
        data.put(el.toInt())
    }

    override fun put(index: Int, el: Int): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is IntBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Int>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): IntBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewInt

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

internal class MemoryViewLong(private val data: LongBuffer) : MemoryView<Long>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Long> {
        val ret = MemoryViewLong(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    override fun get(index: Int): Long = data[index]

    override fun get(): Long = data.get()


    override operator fun get(dst: LongArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Long>) {
        data.put(src.toLongArray())
    }

    override fun put(src: LongArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Long) {
        data.put(el)
    }

    override fun put(el: Int) {
        data.put(el.toLong())
    }

    override fun put(el: Double) {
        data.put(el.toLong())
    }

    override fun put(index: Int, el: Long): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is LongBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Long>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): LongBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewLong

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

internal class MemoryViewFloat(private val data: FloatBuffer) : MemoryView<Float>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Float> {
        val ret = MemoryViewFloat(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    override fun get(index: Int): Float = data[index]

    override fun get(): Float = data.get()


    override operator fun get(dst: FloatArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Float>): Unit {
        data.put(src.toFloatArray())
    }

    override fun put(src: FloatArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Float) {
        data.put(el)
    }

    override fun put(el: Int) {
        data.put(el.toFloat())
    }

    override fun put(el: Double) {
        data.put(el.toFloat())
    }

    override fun put(index: Int, el: Float): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is FloatBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Float>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): FloatBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewFloat

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

internal class MemoryViewDouble(private val data: DoubleBuffer) : MemoryView<Double>() {

    override fun duplicate(position: Int, limit: Int): MemoryView<Double> {
        val ret = MemoryViewDouble(data.duplicate())
        ret.data.position(position)
        ret.data.limit(limit)
        return ret
    }

    public override fun sliceInPlace(position: Int, limit: Int): Unit {
        data.position(position)
        data.limit(limit)
    }

    override fun get(index: Int): Double = data[index]

    override fun get(): Double = data.get()


    override operator fun get(dst: DoubleArray, offset: Int, length: Int): Unit {
        data.get(dst, offset, length)
    }

    override fun put(src: Array<out Double>): Unit {
        data.put(src.toDoubleArray())
    }

    override fun put(src: DoubleArray, offset: Int, length: Int): Unit {
        data.put(src, offset, length)
    }

    override fun put(el: Double) {
        data.put(el)
    }

    override fun put(el: Int) {
        data.put(el.toDouble())
    }

    override fun put(index: Int, el: Double): Unit {
        data.put(index, el)
    }

    override fun <E : Buffer> put(src: E): Unit {
        if (src is DoubleBuffer) {
            data.put(src)
        } else {
            throw IllegalArgumentException("")
        }
    }

    override fun <E : Iterable<Double>> put(src: E) {
        for (item in src)
            data.put(item)
    }

    override fun rewind(): Unit {
        data.rewind()
    }

    override fun getData(): DoubleBuffer = data

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        if (!super.equals(other)) return false

        other as MemoryViewDouble

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

@Suppress("UNCHECKED_CAST")
fun <T : Number> initMemoryView(numElements: Int, dataType: DataType): MemoryView<T> {
    val t = when (dataType.nativeCode) {
        1 -> MemoryViewByte(
            allocateDirectByteBuffer(numElements, dataType)
        )
        2 -> MemoryViewShort(
            allocateDirectByteBuffer(numElements, dataType).asShortBuffer()
        )
        3 -> MemoryViewInt(
            allocateDirectByteBuffer(numElements, dataType).asIntBuffer()
        )
        4 -> MemoryViewLong(
            allocateDirectByteBuffer(numElements, dataType).asLongBuffer()
        )
        5 -> MemoryViewFloat(
            allocateDirectByteBuffer(numElements, dataType).asFloatBuffer()
        )
        6 -> MemoryViewDouble(
            allocateDirectByteBuffer(numElements, dataType)
                .asDoubleBuffer()
        )
        else -> throw Exception("Unknown datatype: ${dataType.name}")
    }
    return t as MemoryView<T>
}

private fun allocateDirectByteBuffer(numElements: Int, dataType: DataType) =
    ByteBuffer.allocateDirect(numElements * dataType.itemSize).order(ByteOrder.nativeOrder())
