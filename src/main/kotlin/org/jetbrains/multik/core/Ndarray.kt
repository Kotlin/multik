package org.jetbrains.multik.core

import kotlin.reflect.KClass
import kotlin.reflect.jvm.jvmName


/**
 * Multidimensional array. Stores a [MemoryView] object.
 */
open class Ndarray<T : Number, D : DN> constructor(
    val data: MemoryView<T>,
    val offset: Int = 0,
    val shape: IntArray,
    val strides: IntArray,
    val dtype: DataType,
    val dim: D
) {

    val size: Int = shape.reduce { acc, i -> acc * i }

    val indices: IntRange
        get() {
            if (dim.d != 1) throw IllegalStateException("Ndarray of dimension ${dim.d}, use multiIndex.")
            return 0..size - 1
        }

    val multiIndices: MultiIndexProgression get() = IntArray(dim.d)..shape


    val V: View<T, D> by lazy(LazyThreadSafetyMode.PUBLICATION) { View(this) }

    fun getData(): Array<T> = data.getData()

    public fun isEmpty() = size == 0

    public fun isNotEmpty() = !isEmpty()

    @JvmName("getVararg")
    public operator fun get(vararg index: Int): T = get(index)

    @JvmName("getIntArray")
    public operator fun get(index: IntArray): T {
        check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
        return data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }]
    }

    @JvmName("setVararg")
    public operator fun set(vararg index: Int, value: T): Unit {
        set(index, value)
    }

    @JvmName("setIntArray")
    public operator fun set(index: IntArray, value: T): Unit {
        check(index.size == dim.d) { "number of indices doesn't match dimension: ${index.size} != ${dim.d}" }
        data[strides.foldIndexed(offset) { i, acc, stride -> acc + index[i] * stride }] = value
    }

    public operator fun set(index: Int, value: T): Unit {
        data[offset + strides.first() * index] = value
    }


    operator fun iterator(): Iterator<T> =
        NdarrayIterator(data, offset, strides, shape)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Ndarray<*, *>

        if (!shape.contentEquals(other.shape)) return false
        if (dtype != other.dtype) return false
        if (dim != other.dim) return false


        for (index in multiIndices) {
            if (this[index] != other.get(*index)) {
                return false
            }
        }

        return true
    }

    override fun hashCode(): Int {
        var result = 1
        for (el in this) {
            result = 31 * result + el.hashCode()
        }
        return result
    }

    fun view(index: Int, axis: Int = 0): Ndarray<T, DN> {
        return initNdarray(
            data, offset + strides[axis] * index,
            shape.remove(axis), strides.remove(axis), this.dtype, DN.of(this.dim.d - 1)
        )
    }

    fun view(index: IntArray, axes: IntArray): Ndarray<T, DN> {
        val newShape = shape.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
        val newStrides = strides.filterIndexed { i, _ -> !axes.contains(i) }.toIntArray()
        var newOffset = offset
        for (i in axes.indices)
            newOffset += strides[axes[i]] * index[i]
        return initNdarray(data, newOffset, newShape, newStrides, this.dtype, DN.of(this.dim.d - axes.size))
    }

    inline fun <reified E : Number> asType(): Ndarray<E, D> {
        val dataType = DataType.of(E::class)
        val newData = initMemoryView<E>(this.data.size, dataType) { this.data[it].toPrimitiveType() }
        return initNdarray(newData, this.offset, this.shape, this.strides, dataType, this.dim)
    }

    fun <E : Number> asType(dataType: DataType): Ndarray<E, D> {
        val newData = initMemoryView<E>(this.data.size, dataType) { this.data[it] as E }
        return initNdarray(newData, this.offset, this.shape, this.strides, dataType, this.dim)
    }

    inline fun <reified M : DN> reshape(vararg dims: Int): Ndarray<T, M> {
        // todo negative shape?
        dims.forEach { require(it > 0) { "Shape must be positive but was $it" } }
        check(dims.fold(1, Int::times) == size) {
            "Cannot reshape array of size $size into a new shape ${dims.joinToString(prefix = "(", postfix = ")")}"
        }
        val newDim = DN.of<M>()
        requireDimension(newDim, dims.size)
        return if (shape.contentEquals(dims)) {
            this as Ndarray<T, M>
        } else {
            initNdarray(this.data, this.offset, dims, dtype = this.dtype, dim = newDim)
        }

    }

    //todo strides?
    fun toD1Array(): D1Array<T> {
        if (this is D1Array) return this
        return D1Array(this.data, this.offset, intArrayOf(this.size), dtype = this.dtype)
    }

    fun asD2Array(): D2Array<T> {
        if (this is D2Array) return this
        else throw Exception("Cannot cast Ndarray to D2Array.")
    }

    fun asD3Array(): D3Array<T> {
        if (this is D3Array) return this
        else throw Exception("Cannot cast Ndarray to D3Array.")
    }

    fun asD4Array(): D4Array<T> {
        if (this is D4Array) return this
        else throw Exception("Cannot cast Ndarray to D4Array.")
    }

    override fun toString(): String {
        val sb = StringBuilder()
        sb.append('[')
        if (this.dim.d == 1) {
            for (pos in 0 until shape.first()) {
                sb.append(this[pos])
                if (pos < shape.first() - 1) {
                    sb.append(", ")
                }
            }
            sb.append(']')
            return sb.toString()
        }

        for (ind in 0 until shape.first()) {
            sb.append(V[ind].toString())
            if (ind < shape.first() - 1) {
                val newLine = "\n".repeat(dim.d - 1)
                sb.append(",$newLine")
            }
        }

        sb.append(']')
        return sb.toString()
    }
}

internal fun IntArray.remove(pos: Int) = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
}

public class View<T : Number, D : DN>(private val base: Ndarray<T, D>) /*: BaseNdarray by base */ {
    operator fun get(vararg indices: Int): Ndarray<T, DN> {
        return indices.fold(this.base) { m, pos -> m.view(pos) as Ndarray<T, D> } as Ndarray<T, DN>
    }
}

public class D1Array<T : Number>(
    data: MemoryView<T>,
    offset: Int = 0,
    shape: IntArray,
    strides: IntArray = computeStrides(shape),
    dtype: DataType
) : Ndarray<T, D1>(data, offset, shape, strides, dtype, D1) {

    operator fun get(index: Int): T = data[offset + strides.first() * index]

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        javaClass != other?.javaClass -> false
        other !is Ndarray<*, *> -> false
        dtype != other.dtype -> false
        dim != other.dim -> false
        !shape.contentEquals(other.shape) -> false
        else -> (0 until size).all { this[it] == other[it] }
    }

    override fun toString(): String = buildString {
        append('[')
        for (i in 0 until shape.first()) {
            append(this@D1Array[i])
            if (i < shape.first() - 1)
                append(", ")
        }
        append(']')
    }

}

public class D2Array<T : Number>(
    data: MemoryView<T>,
    offset: Int = 0,
    shape: IntArray,
    strides: IntArray = computeStrides(shape),
    dtype: DataType
) : Ndarray<T, D2>(data, offset, shape, strides, dtype, D2) {

    operator fun get(index: Int): D1Array<T> = view(index, 0) as D1Array<T>

    operator fun get(ind1: Int, ind2: Int): T = data[offset + strides[0] * ind1 + strides[1] * ind2]

    override fun equals(other: Any?): Boolean {
        when {
            this === other -> return true
            javaClass != other?.javaClass -> return false
            other !is Ndarray<*, *> -> return false
            dtype != other.dtype -> return false
            dim != other.dim -> return false
            !shape.contentEquals(other.shape) -> return false
            else -> {
                for (ax0 in 0 until shape[0])
                    for (ax1 in 0 until shape[1])
                        if (this[ax0, ax1] != other[ax0, ax1])
                            return false
                return true
            }
        }
    }

    override fun toString(): String = buildString {
        append('[')
        for (ax0 in 0 until shape[0]) {
            append('[')
            for (ax1 in 0 until shape[1]) {
                append(this@D2Array[ax0, ax1])
                if (ax1 < shape[1] - 1)
                    append(", ")
            }
            append(']')
            if (ax0 < shape[0] - 1)
                append(",\n")
        }
        append(']')
    }
}

public class D3Array<T : Number>(
    data: MemoryView<T>,
    offset: Int = 0,
    shape: IntArray,
    strides: IntArray = computeStrides(shape),
    dtype: DataType
) : Ndarray<T, D3>(data, offset, shape, strides, dtype, D3) {

    operator fun get(index: Int): D2Array<T> = view(index, 0) as D2Array<T>

    operator fun get(ind1: Int, ind2: Int): D1Array<T> = view(intArrayOf(ind1, ind2), intArrayOf(0, 1)) as D1Array<T>

    operator fun get(ind1: Int, ind2: Int, ind3: Int): T =
        data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3]

    override fun equals(other: Any?): Boolean {
        when {
            this === other -> return true
            javaClass != other?.javaClass -> return false
            other !is Ndarray<*, *> -> return false
            dtype != other.dtype -> return false
            dim != other.dim -> return false
            !shape.contentEquals(other.shape) -> return false
            else -> {
                for (ax0 in 0 until shape[0])
                    for (ax1 in 0 until shape[1])
                        for (ax2 in 0 until shape[2])
                            if (this[ax0, ax1, ax2] != other[ax0, ax1, ax2])
                                return false
                return true
            }
        }
    }

    override fun toString(): String = buildString {
        append('[')
        for (ax0 in 0 until shape[0]) {
            append('[')
            for (ax1 in 0 until shape[1]) {
                append('[')
                for (ax2 in 0 until shape[2]) {
                    append(this@D3Array[ax0, ax1, ax2])
                    if (ax2 < shape[2] - 1)
                        append(", ")
                }
                append(']')
                if (ax1 < shape[1] - 1)
                    append(",\n")
            }
            append(']')
            if (ax0 < shape[0] - 1)
                append(",\n\n")
        }
        append(']')
    }
}

public class D4Array<T : Number>(
    data: MemoryView<T>,
    offset: Int = 0,
    shape: IntArray,
    strides: IntArray = computeStrides(shape),
    dtype: DataType
) : Ndarray<T, D4>(data, offset, shape, strides, dtype, D4) {

    operator fun get(index: Int): D3Array<T> = view(index, 0) as D3Array<T>

    operator fun get(ind1: Int, ind2: Int): D2Array<T> =
        view(intArrayOf(ind1, ind2), intArrayOf(0, 1)) as D2Array<T>

    operator fun get(ind1: Int, ind2: Int, ind3: Int): D1Array<T> =
        view(intArrayOf(ind1, ind2, ind3), intArrayOf(0, 1, 2)) as D1Array<T>

    operator fun get(ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        data[offset + strides[0] * ind1 + strides[1] * ind2 + strides[2] * ind3 + strides[3] * ind4]

    override fun equals(other: Any?): Boolean {
        when {
            this === other -> return true
            javaClass != other?.javaClass -> return false
            other !is Ndarray<*, *> -> return false
            dtype != other.dtype -> return false
            dim != other.dim -> return false
            !shape.contentEquals(other.shape) -> return false
            else -> {
                for (ax0 in 0 until shape[0])
                    for (ax1 in 0 until shape[1])
                        for (ax2 in 0 until shape[2])
                            for (ax3 in 0 until shape[3])
                                if (this[ax0, ax1, ax2, ax3] != other[ax0, ax1, ax2, ax3])
                                    return false
                return true
            }
        }
    }

    override fun toString(): String = buildString {
        append('[')
        for (ax0 in 0 until shape[0]) {
            append('[')
            for (ax1 in 0 until shape[1]) {
                append('[')
                for (ax2 in 0 until shape[2]) {
                    append('[')
                    for (ax3 in 0 until shape[3]) {
                        append(this@D4Array[ax0, ax1, ax2, ax3])
                        if (ax3 < shape[3] - 1)
                            append(", ")
                    }
                    append(']')
                    if (ax2 < shape[2] - 1)
                        append(",\n")
                }
                append(']')
                if (ax1 < shape[1] - 1)
                    append(",\n\n")
            }
            append(']')
            if (ax0 < shape[0] - 1)
                append(",\n\n\n")
        }
        append(']')
    }
}


enum class DataType(val nativeCode: Int, val itemSize: Int, val clazz: KClass<out Number>) {
    ByteDataType(1, 1, Byte::class),
    ShortDataType(2, 2, Short::class),
    IntDataType(3, 4, Int::class),
    LongDataType(4, 8, Long::class),
    FloatDataType(5, 4, Float::class),
    DoubleDataType(6, 8, Double::class);

    companion object {
        fun of(i: Int): DataType {
            return when (i) {
                1 -> ByteDataType
                2 -> ShortDataType
                3 -> IntDataType
                4 -> LongDataType
                5 -> FloatDataType
                6 -> DoubleDataType
                else -> throw IllegalStateException("One of the primitive types was expected")
            }
        }

        fun <T : Number> of(element: T): DataType {
            return when (element) {
                is Byte -> ByteDataType
                is Short -> ShortDataType
                is Int -> IntDataType
                is Long -> LongDataType
                is Float -> FloatDataType
                is Double -> DoubleDataType
                else -> throw IllegalStateException("One of the primitive types was expected")
            }
        }

        inline fun <reified T : Number> of(type: KClass<out T>) = when (type) {
            Byte::class -> ByteDataType
            Short::class -> ShortDataType
            Int::class -> IntDataType
            Long::class -> LongDataType
            Float::class -> FloatDataType
            Double::class -> DoubleDataType
            else -> throw IllegalStateException("One of the primitive types was expected, got ${type.jvmName}")
        }
    }
}

/**
 *
 */
class NdarrayIterator<T : Number>(
    private val data: MemoryView<T>,
    private val offset: Int = 0,
    private val strides: IntArray,
    private val shape: IntArray
) : Iterator<T> {
    private val index = IntArray(shape.size)

    override fun hasNext(): Boolean {
        for (i in shape.indices) {
            if (index[i] >= shape[i])
                return false
        }
        return true
    }

    override fun next(): T {
        var p = offset
        for (i in shape.indices) {
            p += strides[i] * index[i]
        }

        for (i in shape.size - 1 downTo 0) {
            val t = index[i] + 1
            if (t >= shape[i] && i != 0) {
                index[i] = 0
            } else {
                index[i] = t
                break
            }
        }

        return data[p]
    }
}

class MultiIndexProgression(public val first: IntArray, public val last: IntArray, public val step: Int = 1) {

    init {
        if (step == 0) throw IllegalArgumentException("Step must be non-zero.")
        if (step == Int.MIN_VALUE) throw IllegalArgumentException("Step must be greater than Int.MIN_VALUE to avoid overflow on negation.")
        if (first.size != last.size) throw IllegalArgumentException("Sizes first and last must be identical.")
    }

    operator fun iterator(): Iterator<IntArray> = MultiIndexIterator(first, last, step)

    override fun equals(other: Any?): Boolean =
        other is MultiIndexProgression && (first.contentEquals(other.first) && last.contentEquals(other.last))

    override fun hashCode(): Int {
        return (first + last).hashCode()
    }

    override fun toString(): String {
        return "${first.joinToString(prefix = "(", postfix = ")")}..${last.joinToString(prefix = "(", postfix = ")")}"
    }
}

internal class MultiIndexIterator(first: IntArray, last: IntArray, private val step: Int) : Iterator<IntArray> {
    private val finalElement: IntArray = IntArray(last.size) { last[it] - 1 }
    private var hasNext: Boolean = if (step > 0) {
        var ret: Boolean = true
        for (i in first.size - 1 downTo 0) {
            if (first[i] > last[i]) {
                ret = false
                break
            }
        }
        ret
    } else {
        var ret: Boolean = true
        for (i in first.size - 1 downTo 0) {
            if (first[i] < last[i]) {
                ret = false
                break
            }
        }
        ret
    }

    //todo (-1???)
    private val next = if (hasNext) first.apply { set(lastIndex, -1) } else finalElement

    override fun hasNext(): Boolean = hasNext

    override fun next(): IntArray {
        next += step
        if (next.contentEquals(finalElement)) {
            if (!hasNext) throw NoSuchElementException()
            hasNext = false
        }
        return next
    }

    private operator fun IntArray.plusAssign(value: Int) {
        for (i in this.size - 1 downTo 0) {
            val t = this[i] + value
            if (t > finalElement[i] && i != 0) {
                this[i] = 0
            } else {
                this[i] = t
                break
            }
        }
    }
}

public operator fun IntArray.rangeTo(other: IntArray): MultiIndexProgression {
    return MultiIndexProgression(this, other)
}

public infix fun MultiIndexProgression.step(step: Int): MultiIndexProgression {
    if (step <= 0) throw IllegalArgumentException("Step must be posotove, was: $step.")
    return MultiIndexProgression(first, last, step)
}

public infix fun IntArray.downTo(to: IntArray): MultiIndexProgression {
    return MultiIndexProgression(this, to, -1)
}
