package org.jetbrains.multik.core

sealed class MemoryView<T : Number> : Iterable<T> {
    internal abstract val data: Any

    public abstract var size: Int

    public abstract var indices: IntRange

    public abstract var lastIndex: Int

    public abstract operator fun get(index: Int): T

    public abstract operator fun set(index: Int, value: T): Unit

    public abstract fun getData(): Array<T>

    public abstract override fun iterator(): Iterator<T>

    //todo
    override fun equals(other: Any?): Boolean {
        return super.equals(other)
    }

    override fun hashCode(): Int {
        return super.hashCode()
    }
}

@PublishedApi
internal class MemoryViewByteArray(override val data: ByteArray) : MemoryView<Byte>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Byte = data[index]

    override fun set(index: Int, value: Byte): Unit {
        data[index] = value
    }

    override fun getData(): Array<Byte> = data.toTypedArray()

    override fun iterator(): Iterator<Byte> = data.iterator()
}

@PublishedApi
internal class MemoryViewShortArray(override val data: ShortArray) : MemoryView<Short>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Short = data[index]

    override fun set(index: Int, value: Short): Unit {
        data[index] = value
    }

    override fun getData(): Array<Short> = data.toTypedArray()

    override fun iterator(): Iterator<Short> = data.iterator()
}

@PublishedApi
internal class MemoryViewIntArray(override val data: IntArray) : MemoryView<Int>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Int = data[index]

    override fun set(index: Int, value: Int): Unit {
        data[index] = value
    }

    override fun getData(): Array<Int> = data.toTypedArray()

    override fun iterator(): Iterator<Int> = data.iterator()
}

@PublishedApi
internal class MemoryViewLongArray(override val data: LongArray) : MemoryView<Long>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Long = data[index]

    override fun set(index: Int, value: Long): Unit {
        data[index] = value
    }

    override fun getData(): Array<Long> = data.toTypedArray()

    override fun iterator(): Iterator<Long> = data.iterator()
}

@PublishedApi
internal class MemoryViewFloatArray(override val data: FloatArray) : MemoryView<Float>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Float = data[index]

    override fun set(index: Int, value: Float): Unit {
        data[index] = value
    }

    override fun getData(): Array<Float> = data.toTypedArray()

    override fun iterator(): Iterator<Float> = data.iterator()
}

@PublishedApi
internal class MemoryViewDoubleArray(override val data: DoubleArray) : MemoryView<Double>() {
    override var size: Int = data.size

    override var indices: IntRange = data.indices

    override var lastIndex: Int = data.lastIndex

    override fun get(index: Int): Double = data[index]

    override fun set(index: Int, value: Double): Unit {
        data[index] = value
    }

    override fun getData(): Array<Double> = data.toTypedArray()

    override fun iterator(): Iterator<Double> = data.iterator()
}

fun <T : Number> initMemoryView(size: Int, dataType: DataType): MemoryView<T> {
    val t = when (dataType.nativeCode) {
        1 -> MemoryViewByteArray(ByteArray(size))
        2 -> MemoryViewShortArray(ShortArray(size))
        3 -> MemoryViewIntArray(IntArray(size))
        4 -> MemoryViewLongArray(LongArray(size))
        5 -> MemoryViewFloatArray(FloatArray(size))
        6 -> MemoryViewDoubleArray(DoubleArray(size))
        else -> throw Exception("Unknown datatype: ${dataType.name}")
    }
    @Suppress("UNCHECKED_CAST")
    return t as MemoryView<T>
}

@Suppress("UNCHECKED_CAST")
fun <T : Number> initMemoryView(size: Int, dataType: DataType, init: (Int) -> T): MemoryView<T> {
    val t = when (dataType.nativeCode) {
        1 -> MemoryViewByteArray(ByteArray(size, init as (Int) -> Byte))
        2 -> MemoryViewShortArray(ShortArray(size, init as (Int) -> Short))
        3 -> MemoryViewIntArray(IntArray(size, init as (Int) -> Int))
        4 -> MemoryViewLongArray(LongArray(size, init as (Int) -> Long))
        5 -> MemoryViewFloatArray(FloatArray(size, init as (Int) -> Float))
        6 -> MemoryViewDoubleArray(DoubleArray(size, init as (Int) -> Double))
        else -> throw Exception("Unknown datatype: ${dataType.name}")
    }
    return t as MemoryView<T>
}

@Suppress("UNCHECKED_CAST")
fun <T : Number> List<T>.toViewPrimitiveArray(dataType: DataType): MemoryView<T> {
    val t = when (dataType.nativeCode) {
        1 -> MemoryViewByteArray((this as List<Byte>).toByteArray())
        2 -> MemoryViewShortArray((this as List<Short>).toShortArray())
        3 -> MemoryViewIntArray((this as List<Int>).toIntArray())
        4 -> MemoryViewLongArray((this as List<Long>).toLongArray())
        5 -> MemoryViewFloatArray((this as List<Float>).toFloatArray())
        6 -> MemoryViewDoubleArray((this as List<Double>).toDoubleArray())
        else -> throw Exception("Unknown datatype: ${dataType.name}")
    }
    return t as MemoryView<T>
}
