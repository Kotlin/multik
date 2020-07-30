package org.jetbrains.multik.core

import kotlin.reflect.KClass
import kotlin.reflect.jvm.jvmName

typealias D1Array<T> = Ndarray<T, D1>
typealias D2Array<T> = Ndarray<T, D2>
typealias D3Array<T> = Ndarray<T, D3>
typealias D4Array<T> = Ndarray<T, D4>

/**
 * Multidimensional array. Stores a [MemoryView] object.
 */
class Ndarray<T : Number, D : Dimension> @PublishedApi internal constructor(
    val data: MemoryView<T>,
    val offset: Int = 0,
    val shape: IntArray,
    val strides: IntArray = computeStrides(shape),
    val dtype: DataType,
    val dim: D
) : MutableMultiArray<T, D> {

    val size: Int = shape.fold(1, Int::times)

    override val indices: IntRange
        get() {
            if (dim.d != 1) throw IllegalStateException("Ndarray of dimension ${dim.d}, use multiIndex.")
            return 0..size - 1
        }

    override val multiIndices: MultiIndexProgression get() = IntArray(dim.d)..shape

    fun getData(): Array<T> = data.getData()

    public override fun isEmpty(): Boolean = size == 0

    public override fun isNotEmpty(): Boolean = !isEmpty()

    public override operator fun iterator(): Iterator<T> =
        NdarrayIterator(data, offset, strides, shape)

    public inline fun <reified E : Number> asType(): Ndarray<E, D> {
        val dataType = DataType.of(E::class)
        return this.asType(dataType)
    }

    public fun <E : Number> asType(dataType: DataType): Ndarray<E, D> {
        val newData = initMemoryView<E>(this.data.size, dataType) { this.data[it] as E }
        return Ndarray<E, D>(newData, this.offset, this.shape, this.strides, dataType, this.dim)
    }

    public fun asD1Array(): D1Array<T> {
        if (this.dim.d == 1) return this as D1Array<T>
        else throw ClassCastException("Cannot cast Ndarray of dimension ${this.dim.d} to Ndarray of dimension 1.")
    }

    public fun asD2Array(): D2Array<T> {
        if (this.dim.d == 2) return this as D2Array<T>
        else throw ClassCastException("Cannot cast Ndarray of dimension ${this.dim.d} to Ndarray of dimension 2.")
    }

    public fun asD3Array(): D3Array<T> {
        if (this.dim.d == 3) return this as D3Array<T>
        else throw ClassCastException("Cannot cast Ndarray of dimension ${this.dim.d} to Ndarray of dimension 3.")
    }

    public fun asD4Array(): D4Array<T> {
        if (this.dim.d == 4) return this as D4Array<T>
        else throw ClassCastException("Cannot cast Ndarray of dimension ${this.dim.d} to Ndarray of dimension 4.")
    }

    public fun asDNArray(): Ndarray<T, DN> {
        if (this.dim.d == -1) throw Exception("Array dimension is undefined")
        if (this.dim.d > 4) return this as Ndarray<T, DN>

        return Ndarray(this.data, this.offset, this.shape, this.strides, this.dtype, DN(this.dim.d))
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Ndarray<*, *>

        if (size != other.size) return false
        if (!shape.contentEquals(other.shape)) return false
        if (dtype != other.dtype) return false
        if (dim != other.dim) return false

        val thIt = this.iterator()
        val othIt = other.iterator()
        while (thIt.hasNext() && othIt.hasNext()) {
            if (thIt.next() != othIt.next())
                return false
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

    override fun toString(): String {
        return when (dim.d) {
            1 -> buildString {
                this@Ndarray as Ndarray<T, D1>
                append('[')
                for (i in 0 until shape.first()) {
                    append(this@Ndarray[i])
                    if (i < shape.first() - 1)
                        append(", ")
                }
                append(']')
            }

            2 -> buildString {
                this@Ndarray as Ndarray<T, D2>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append(this@Ndarray[ax0, ax1])
                        if (ax1 < shape[1] - 1)
                            append(", ")
                    }
                    append(']')
                    if (ax0 < shape[0] - 1)
                        append(",\n")
                }
                append(']')
            }

            3 -> buildString {
                this@Ndarray as Ndarray<T, D3>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append('[')
                        for (ax2 in 0 until shape[2]) {
                            append(this@Ndarray[ax0, ax1, ax2])
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

            4 -> buildString {
                this@Ndarray as Ndarray<T, D4>
                append('[')
                for (ax0 in 0 until shape[0]) {
                    append('[')
                    for (ax1 in 0 until shape[1]) {
                        append('[')
                        for (ax2 in 0 until shape[2]) {
                            append('[')
                            for (ax3 in 0 until shape[3]) {
                                append(this@Ndarray[ax0, ax1, ax2, ax3])
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

            else -> buildString {
                this@Ndarray as Ndarray<*, DN>
                append('[')
                for (ind in 0 until shape.first()) {
                    append(this@Ndarray.V[ind].toString())
                    if (ind < shape.first() - 1) {
                        val newLine = "\n".repeat(dim.d - 1)
                        append(",$newLine")
                    }
                }
                append(']')
            }
        }
    }
}

// TODO(strides? : view.reshape().reshape()?)
public fun <T : Number, D : Dimension> Ndarray<T, D>.reshape(dim1: Int): D1Array<T> {
    // todo negative shape?
    requirePositiveShape(dim1)
    require(dim1 == size) { "Cannot reshape array of size $size into a new shape ($dim1)" }

    return if (this.dim.d == 1 && this.shape.first() == dim1) {
        this as D1Array<T>
    } else {
        D1Array<T>(this.data, this.offset, intArrayOf(dim1), dtype = this.dtype, dim = D1)
    }
}

public fun <T : Number, D : Dimension> Ndarray<T, D>.reshape(dim1: Int, dim2: Int): D2Array<T> {
    val newShape = intArrayOf(dim1, dim2)
    newShape.forEach { requirePositiveShape(it) }
    require(dim1 * dim2 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2)" }

    return if (this.shape.contentEquals(newShape)) {
        this as D2Array<T>
    } else {
        D2Array<T>(this.data, this.offset, newShape, dtype = this.dtype, dim = D2)
    }
}

public fun <T : Number, D : Dimension> Ndarray<T, D>.reshape(dim1: Int, dim2: Int, dim3: Int): D3Array<T> {
    val newShape = intArrayOf(dim1, dim2, dim3)
    newShape.forEach { requirePositiveShape(it) }
    require(dim1 * dim2 * dim3 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3)" }

    return if (this.shape.contentEquals(newShape)) {
        this as D3Array<T>
    } else {
        D3Array<T>(this.data, this.offset, newShape, dtype = this.dtype, dim = D3)
    }
}

public fun <T : Number, D : Dimension> Ndarray<T, D>.reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T> {
    val newShape = intArrayOf(dim1, dim2, dim3, dim4)
    newShape.forEach { requirePositiveShape(it) }
    require(dim1 * dim2 * dim3 * dim4 == size) { "Cannot reshape array of size $size into a new shape ($dim1, $dim2, $dim3, $dim4)" }

    return if (this.shape.contentEquals(newShape)) {
        this as D4Array<T>
    } else {
        D4Array<T>(this.data, this.offset, newShape, dtype = this.dtype, dim = D4)
    }
}

public fun <T : Number, D : Dimension> Ndarray<T, D>.reshape(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): Ndarray<T, DN> {
    val newShape = intArrayOf(dim1, dim2, dim3, dim4) + dims
    newShape.forEach { requirePositiveShape(it) }
    require(newShape.fold(1, Int::times) == size) {
        "Cannot reshape array of size $size into a new shape ${newShape.joinToString(prefix = "(", postfix = ")")}"
    }

    return if (this.shape.contentEquals(newShape)) {
        this as Ndarray<T, DN>
    } else {
        Ndarray<T, DN>(this.data, this.offset, newShape, dtype = this.dtype, dim = DN(newShape.size))
    }
}


internal fun IntArray.remove(pos: Int) = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
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
    if (step <= 0) throw IllegalArgumentException("Step must be positive, was: $step.")
    return MultiIndexProgression(first, last, step)
}

public infix fun IntArray.downTo(to: IntArray): MultiIndexProgression {
    return MultiIndexProgression(this, to, -1)
}
