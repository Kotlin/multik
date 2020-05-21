package org.jetbrains.multik.core

import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.api.toNdarray
import org.jetbrains.multik.jni.Basic
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.LinkedHashMap
import kotlin.reflect.KClass
import kotlin.reflect.jvm.jvmName

/**
 * Dimension class.
 */
open class DN(val d: Int) {
    companion object : DN(5) {
        @Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
        inline fun <D : DN> of(dim: Int): D = when (dim) {
            1 -> D1
            2 -> D2
            3 -> D3
            4 -> D4
            else -> DN(dim)
        } as D
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = d

    override fun toString(): String {
        return "dimension: $d"
    }
}

sealed class D4(d: Int = 4) : DN(d) {
    companion object : D4()
}

sealed class D3(d: Int = 3) : D4(d) {
    companion object : D3()
}

sealed class D2(d: Int = 2) : D3(d) {
    companion object : D2()
}

sealed class D1(d: Int = 1) : D2(d) {
    companion object : D1()
}

/**
 * Multidimensional array. Stores a pointer ([handle]) to a native class and a [MemoryView] object.
 */
class Ndarray<T : Number, D : DN>(
    val handle: Long,
    val data: MemoryView<T>,
    var shape: IntArray,
    var size: Int = shape.reduce { acc, i -> acc + i },
    val dtype: DataType,
    val dim: D
) : Iterable<T> {

    val strides: IntArray = shape.clone().apply {
        this[this.lastIndex] = 1
        for (i in this.lastIndex - 1 downTo 0) {
            this[i] = this[i + 1] * shape[i + 1]
        }
    }

    val indices: IntRange
        get() = 0..size - 1

    fun getData(): List<T> {
        data.rewind()
//        val list = ArrayList<T>()
        val list = ArrayList<T>(data.getData().remaining())
        for (i in 0 until data.getData().remaining()) {
            list.add(data[i])
        }
        return list
    }

    public operator fun get(index: Int): T {
        return data[index]
    }

    public operator fun set(index: Int, value: T): Unit {
        data.put(index, value)
    }

    /**
     * Returns a new array containing only distinct elements from the given array.
     */
    public fun distinct(): Ndarray<T, D1> {
        return this.toMutableSet().toNdarray()
    }

    /**
     * Returns a new array containing only elements from the given array
     * having distinct keys returned by the given [selector] function.
     */
    public inline fun <K> distinctBy(selector: (T) -> K): Ndarray<T, D1> {
        val set = HashSet<K>()
        val list = ArrayList<T>()
        for (e in this) {
            val key = selector(e)
            if (set.add(key))
                list.add(e)
        }
        return list.toNdarray()
    }

    /**
     * Drop first n elements.
     */
    public fun drop(n: Int): Ndarray<T, D1> {
        require(n >= 0) { "Requested element count $n is less than zero." }
        val resultSize = size - n
        val d = initMemoryView<T>(resultSize, dtype).apply {
            for (index in n until size)
                put(this@Ndarray[index])
            rewind()
        }
        val handle = Basic.allocate(d.getData())
        return Ndarray(handle, d, intArrayOf(resultSize), resultSize, dtype, D1)
    }

    /**
     *
     */
    public inline fun dropWhile(predicate: (T) -> Boolean): Ndarray<T, D1> {
        var yielding = false
        val list = ArrayList<T>()
        for (item in this)
            if (yielding)
                list.add(item)
            else if (!predicate(item)) {
                list.add(item)
                yielding = true
            }
        return mk.ndarray(list, intArrayOf(list.size), D1)
    }

    /**
     * Return a new array contains elements matching filter.
     */
    public inline fun filter(predicate: (T) -> Boolean): Ndarray<T, D1> {
        val tmpData = initMemoryView<T>(size, dtype)
        val handle = Basic.allocate(tmpData.getData())
        return filterTo(Ndarray<T, D1>(handle, tmpData, intArrayOf(size), size, dtype, D1), predicate)
    }

    /**
     * Return a new array contains elements matching filter.
     */
    public inline fun filterIndexed(predicate: (index: Int, T) -> Boolean): Ndarray<T, D1> {
        val tmpData = initMemoryView<T>(size, dtype)
        val handle = Basic.allocate(tmpData.getData())
        return filterIndexedTo(Ndarray<T, D1>(handle, tmpData, intArrayOf(size), size, dtype, D1), predicate)
    }

    /**
     * Appends elements matching filter to [destination].
     */
    public inline fun <C : Ndarray<in T, D1>> filterIndexedTo(
        destination: C,
        predicate: (index: Int, T) -> Boolean
    ): C {
        var count = 0
        forEachIndexed { index, element ->
            if (predicate(index, element)) {
                destination.data.put(element)
                count++
            }
        }
        destination.data.rewind()
        destination.size = count
        destination.shape = intArrayOf(count)
        destination.data.sliceInPlace(0, destination.size)
        return destination
    }

    /**
     * Return a new array contains elements matching filter.
     */
    public inline fun filterNot(predicate: (T) -> Boolean): Ndarray<T, D1> {
        val tmpData = initMemoryView<T>(size, dtype)
        val handle = Basic.allocate(tmpData.getData())
        return filterNotTo(Ndarray<T, D1>(handle, tmpData, intArrayOf(size), size, dtype, D1), predicate)
    }

    /**
     * Appends elements matching filter to [destination].
     */
    public inline fun <C : Ndarray<in T, D1>> filterNotTo(destination: C, predicate: (T) -> Boolean): C {
        var count = 0
        for (element in this) if (!predicate(element)) {
            destination.data.put(element)
            count++
        }
        destination.data.rewind()
        destination.size = count
        destination.shape = intArrayOf(count)
        destination.data.sliceInPlace(0, destination.size)
        return destination
    }

    /**
     * Appends elements matching filter to [destination].
     */
    public inline fun <C : Ndarray<in T, D1>> filterTo(destination: C, predicate: (T) -> Boolean): C {
        var count = 0
        for (element in this) {
            if (predicate(element)) {
                destination.data.put(element)
                count++
            }
        }
        destination.data.rewind()
        destination.size = count
        destination.shape = intArrayOf(count)
        destination.data.sliceInPlace(0, destination.size)
        return destination
    }

     //TODO(size!)
//    public inline fun <reified R : Number> flatMap(transform: (T) -> Iterable<R>): Ndarray<R, D> {
//        val d = initMemoryView<R>(size, DataType.of(R::class))
//        val handle = Basic.allocate(d.getData())
//        return flatMapTo(Ndarray(handle, d, shape, size, dtype, dim), transform)
//    }
//
//    public inline fun <R : Number, C : Ndarray<in R, D>> flatMapTo(destination: C, transform: (T) -> Iterable<R>): C {
//        for (element in this) {
//            val list = transform(element)
//            destination.data.put(list)
//        }
//        return destination
//    }

    /**
     *
     */
    public inline fun <K> groupNdarrayBy(keySelector: (T) -> K): Map<K, Ndarray<T, D1>> {
        return groupNdarrayByTo(LinkedHashMap<K, Ndarray<T, D1>>(), keySelector)
    }

    public inline fun <K, V : Number> groupNdarrayBy(
        keySelector: (T) -> K,
        valueTransform: (T) -> V
    ): Map<K, Ndarray<V, D1>> {
        return groupNdarrayByTo(LinkedHashMap<K, Ndarray<V, D1>>(), keySelector, valueTransform)
    }

    //todo(add?)
    public inline fun <K, M : MutableMap<in K, Ndarray<T, D1>>> groupNdarrayByTo(
        destination: M,
        keySelector: (T) -> K
    ): M {
        val map = LinkedHashMap<K, MutableList<T>>()
        for (element in this) {
            val key = keySelector(element)
            val list = map.getOrPut(key) { ArrayList<T>() }
            list.add(element)
        }
        for (item in map)
            destination.put(item.key, item.value.toNdarray())
        return destination
    }

    //todo(add?)
    public inline fun <K, V : Number, M : MutableMap<in K, Ndarray<V, D1>>> groupNdarrayByTo(
        destination: M, keySelector: (T) -> K, valueTransform: (T) -> V
    ): M {
        val map = LinkedHashMap<K, MutableList<V>>()
        for (element in this) {
            val key = keySelector(element)
            val list = map.getOrPut(key) { ArrayList<V>() }
            list.add(valueTransform(element))
        }
        for (item in map)
            destination.put(item.key, item.value.toNdarray())
        return destination
    }

    public inline fun <K> groupingNdarrayBy(crossinline keySelector: (T) -> K): Grouping<T, K> {
        return object : Grouping<T, K> {
            override fun sourceIterator(): Iterator<T> = this@Ndarray.iterator()
            override fun keyOf(element: T): K = keySelector(element)
        }
    }

    /**
     * Return a new array contains elements after applying [transform].
     */
    public inline fun <reified R : Number> map(transform: (T) -> R): Ndarray<R, D> {
        val newDtype = DataType.of(R::class)
        val d = initMemoryView<R>(size, newDtype)
        val handle = Basic.allocate(d.getData())
        return mapTo(Ndarray(handle, d, shape, size, newDtype, dim), transform)
    }

    /**
     * Return a new array contains elements after applying [transform].
     */
    public inline fun <reified R : Number> mapIndexed(transform: (index: Int, T) -> R): Ndarray<R, D> {
        val newDtype = DataType.of(R::class)
        val d = initMemoryView<R>(size, newDtype)
        val handle = Basic.allocate(d.getData())
        return mapIndexedTo(Ndarray(handle, d, shape, size, newDtype, dim), transform)
    }

    /**
     * Return a new array contains elements after applying [transform].
     */
    public inline fun <reified R : Number> mapIndexedNotNull(transform: (index: Int, T) -> R?): Ndarray<R, D> {
        val newDtype = DataType.of(R::class)
        val d = initMemoryView<R>(size, newDtype)
        val handle = Basic.allocate(d.getData())
        return mapIndexedNotNullTo(Ndarray(handle, d, shape, size, newDtype, dim), transform)
    }

    /**
     * Appends elements after applying [transform] to [destination].
     */
    public inline fun <R : Any, C : Ndarray<in R, D>> mapIndexedNotNullTo(
        destination: C,
        transform: (index: Int, T) -> R?
    ): C {
        forEachIndexed { index, element ->
            transform(index, element)?.let {
                destination.data.put(it)
            }
        }
        destination.data.rewind()
        return destination
    }

    /**
     * Appends elements after applying [transform] to [destination].
     */
    public inline fun <R, C : Ndarray<in R, D>> mapIndexedTo(destination: C, transform: (index: Int, T) -> R): C {
        var index = 0
        for (item in this)
            destination.data.put(transform(index++, item))
        destination.data.rewind()
        return destination
    }

    /**
     * Return a new array contains elements after applying [transform].
     */
    public inline fun <reified R : Number> mapNotNull(transform: (T) -> R?): Ndarray<R, D> {
        val newDtype = DataType.of(R::class)
        val d = initMemoryView<R>(size, newDtype)
        val handle = Basic.allocate(d.getData())
        return mapNotNullTo(Ndarray(handle, d, shape, size, newDtype, dim), transform)
    }

    /**
     * Appends elements after applying [transform] to [destination].
     */
    public inline fun <R : Any, C : Ndarray<in R, D>> mapNotNullTo(destination: C, transform: (T) -> R?): C {
        forEach { element -> transform(element)?.let { destination.data.put(it) } }
        destination.data.rewind()
        return destination
    }

    /**
     * Appends elements after applying [transform] to [destination].
     */
    public inline fun <R, C : Ndarray<in R, D>> mapTo(destination: C, transform: (T) -> R): C {
        for (item in this)
            destination.data.put(transform(item))
        destination.data.rewind()
        return destination
    }

    //todo(remove element or minus?)
    public inline fun <T : Number, D : DN> Ndarray<T, D>.minusElement(element: T): Ndarray<T, D> {
        minusAssign(element)
        return this
    }

    public inline fun <T : Number, D : DN> Ndarray<T, D>.partitionNdarray(predicate: (T) -> Boolean): Pair<Ndarray<T, D1>, Ndarray<T, D1>> {
        val tmpFirstData = initMemoryView<T>(size, dtype)
        val tmpSecondData = initMemoryView<T>(size, dtype)
        var firstCount = 0
        var secondCount = 0
        for (element in this) {
            if (predicate(element)) {
                tmpFirstData.put(element)
                firstCount++
            } else {
                tmpSecondData.put(element)
                secondCount++
            }
        }
        val firstData = tmpFirstData.duplicate(limit = firstCount)
        val secondData = tmpFirstData.duplicate(limit = secondCount)
        return Pair<Ndarray<T, D1>, Ndarray<T, D1>>(
            Ndarray(Basic.allocate(firstData.getData()), firstData, intArrayOf(firstCount), firstCount, dtype, D1),
            Ndarray(Basic.allocate(secondData.getData()), secondData, intArrayOf(secondCount), secondCount, dtype, D1)
        )
    }

    //todo(add element or plus?)
    public inline fun <T : Number, D : DN> Ndarray<T, D>.plusElement(element: T): Ndarray<T, D> {
        plusAssign(element)
        return this
    }

    //TODO (create view: swap position and limit)
    //public fun <T: Number, D: DN> Ndarray<T, D>.reversed(): Ndarray<T, D> {}

    override fun iterator(): Iterator<T> =
        NdarrayIterator(data, 0, strides, shape)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Ndarray<*, *>

        if (handle == other.handle) return true
        if (data != other.data) return false
        if (!shape.contentEquals(other.shape)) return false
        if (dtype != other.dtype) return false
        if (dim != other.dim) return false
        if (!strides.contentEquals(other.strides)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = handle.hashCode()
        result = 31 * result + data.hashCode()
        result = 31 * result + shape.contentHashCode()
        result = 31 * result + dtype.hashCode()
        result = 31 * result + dim.hashCode()
        result = 31 * result + strides.contentHashCode()
        return result
    }

    override fun toString(): String {
        return this.joinToString(prefix = "[", postfix = "]")
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