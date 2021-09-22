/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarrayCommon
import org.jetbrains.kotlinx.multik.api.toCommonNDArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.abs
import kotlin.math.min
import kotlin.jvm.JvmName

public fun <T, D : Dimension> MultiArray<T, D>.isTransposed(): Boolean {
    val x = this as NDArray<T, D>
    if (x.dim == D1) return false
    return x.offset == 0 && x.size == x.data.size
        && x.strides.reversedArray().contentEquals(computeStrides(x.shape.reversedArray()))
}


/**
 * Returns `true` if all elements match the given [predicate].
 * If an ndarray is empty, then always returns `true`.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.all(predicate: (T) -> Boolean): Boolean {
    if (isEmpty()) return true
    for (element in this) if (!predicate(element)) return false
    return true
}

/**
 * Returns `true` if collection has at least one element.
 * @see NDArray.isNotEmpty
 */
public fun <T, D : Dimension> MultiArray<T, D>.any(): Boolean {
    return isNotEmpty()
}

/**
 * Returns `true` if at least one element matches the given [predicate].
 * If an ndarray is empty, then always returns `false`.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.any(predicate: (T) -> Boolean): Boolean {
    if (isEmpty()) return false
    for (element in this) if (predicate(element)) return true
    return false
}

/**
 * Creates a [Sequence] instance that wraps the original collection returning its elements when being iterated.
 */
public fun <T, D : Dimension> MultiArray<T, D>.asSequence(): Sequence<T> {
    return Sequence { this.iterator() }
}

/**
 * Returns a [Map] containing key-value pairs provided by [transform] function
 * applied to elements of the given collection.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <T, D : Dimension, K, V> MultiArray<T, D>.associate(transform: (T) -> Pair<K, V>): Map<K, V> {
    val capacity = mapCapacity(this.size).coerceAtLeast(16)
    return associateTo(LinkedHashMap(capacity), transform)
}

/**
 * Returns a [Map] containing the elements from the given collection indexed by the key
 * returned from [keySelector] function applied to each element.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <T, D : Dimension, K> MultiArray<T, D>.associateBy(keySelector: (T) -> K): Map<K, T> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector)
}

/**
 * Returns a [Map] containing the values provided by [valueTransform] and indexed by [keySelector] functions applied to elements of the given collection.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <T, D : Dimension, K, V> MultiArray<T, D>.associateBy(
    keySelector: (T) -> K, valueTransform: (T) -> V
): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector, valueTransform)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function applied to each element of the given collection
 * and value is the element itself.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <T, D : Dimension, K, M : MutableMap<in K, in T>> MultiArray<T, D>.associateByTo(
    destination: M, keySelector: (T) -> K
): M {
    for (element in this)
        destination.put(keySelector(element), element)
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function and
 * and value is provided by the [valueTransform] function applied to elements of the given collection.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <T, D : Dimension, K, V, M : MutableMap<in K, in V>> MultiArray<T, D>.associateByTo(
    destination: M, keySelector: (T) -> K, valueTransform: (T) -> V
): M {
    for (element in this)
        destination.put(keySelector(element), valueTransform(element))
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs
 * provided by [transform] function applied to each element of the given collection.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 */
public inline fun <T, D : Dimension, K, V, M : MutableMap<in K, in V>> MultiArray<T, D>.associateTo(
    destination: M, transform: (T) -> Pair<K, V>
): M {
    for (element in this)
        destination += transform(element)
    return destination
}

/**
 * Returns a [Map] where keys are elements from the given collection and values are
 * produced by the [valueSelector] function applied to each element.
 *
 * If any two elements are equal, the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <K, D : Dimension, V> MultiArray<K, D>.associateWith(valueSelector: (K) -> V): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateWithTo(LinkedHashMap(capacity), valueSelector)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs for each element of the given collection,
 * where key is the element itself and value is provided by the [valueSelector] function applied to that key.
 *
 * If any two elements are equal, the last one overwrites the former value in the map.
 */
public inline fun <K, D : Dimension, V, M : MutableMap<in K, in V>> MultiArray<K, D>.associateWithTo(
    destination: M, valueSelector: (K) -> V
): M {
    for (element in this)
        destination.put(element, valueSelector(element))
    return destination
}

/**
 * Returns an average value of elements in the ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.average(): Double {
    var sum = 0.0
    var count = 0
    for (element in this) {
        sum += element.toDouble()
        if (++count < 0) throw ArithmeticException("Count overflow has happened.")
    }
    return if (count == 0) Double.NaN else sum / count
}

/**
 * Splits this ndarray into a 2-D ndarray.
 * The last elements in the resulting ndarray may be zero.
 *
 * @param size number of elements in axis 1.
 */
public fun <T, D : Dimension> MultiArray<T, D>.chunked(size: Int): NDArray<T, D2> {
    return windowed(size, size, limit = false)
}

/**
 * Returns `true` if [element] is found in the collection.
 */
public operator fun <T, D : Dimension> MultiArray<T, D>.contains(element: T): Boolean {
    return indexOf(element) >= 0
}

/**
 * Returns the number of elements in an ndarray.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun <T, D : Dimension> MultiArray<T, D>.count(): Int = size

/**
 * Returns the number of elements matching the given [predicate].
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.count(predicate: (T) -> Boolean): Int {
    if (isEmpty()) return 0
    var count = 0
    for (element in this) if (predicate(element)) if (++count < 0) throw ArithmeticException("Count overflow has happened.")
    return count
}

/**
 * Returns a new array containing only distinct elements from the given array.
 */
public fun <T, D : Dimension> MultiArray<T, D>.distinct(): NDArray<T, D1> = this.toMutableSet().toCommonNDArray()

/**
 * Returns a new array containing only elements from the given array
 * having distinct keys returned by the given [selector] function.
 */
public inline fun <T, D : Dimension, K> MultiArray<T, D>.distinctBy(selector: (T) -> K): NDArray<T, D1> {
    val set = HashSet<K>()
    val list = ArrayList<T>()
    for (e in this) {
        val key = selector(e)
        if (set.add(key))
            list.add(e)
    }
    return list.toCommonNDArray()
}

/**
 * Drops first n elements.
 */
public fun <T> MultiArray<T, D1>.drop(n: Int): D1Array<T> {
    if (n == 0) return D1Array(this.data.copyOf(), shape = shape.copyOf(), dim = D1)
    val resultSize = size - abs(n)
    if (resultSize < 0) return D1Array(initMemoryView(0, dtype), shape = intArrayOf(0), dim = D1)
    val k = if (n < 0) 0 else n
    val d = initMemoryView(resultSize, dtype) { this[it + k] }
    val shape = intArrayOf(resultSize)
    return D1Array(d, shape = shape, dim = D1)
}

/**
 * Drops elements that don't satisfy the [predicate].
 */
public inline fun <T> MultiArray<T, D1>.dropWhile(predicate: (T) -> Boolean): NDArray<T, D1> {
    var yielding = false
    val list = ArrayList<T>()
    for (item in this)
        if (yielding)
            list.add(item)
        else if (!predicate(item)) {
            list.add(item)
            yielding = true
        }
    return ndarrayCommon(list, intArrayOf(list.size), D1)
}

/**
 * Return a new array contains elements matching filter.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.filter(predicate: (T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    forEach { if (predicate(it)) list.add(it) }
    return list.toCommonNDArray()
}

/**
 * Return a new array contains elements matching filter.
 */
@JvmName("filterD1Indexed")
public inline fun <T> MultiArray<T, D1>.filterIndexed(predicate: (index: Int, T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    forEachIndexed { index, element -> if (predicate(index, element)) list.add(element) }
    return list.toCommonNDArray()
}


/**
 * Return a new array contains elements matching filter.
 */
@JvmName("filterDNIndexed")
public inline fun <T, D : Dimension> MultiArray<T, D>.filterMultiIndexed(predicate: (index: IntArray, T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    forEachMultiIndexed { index, element -> if (predicate(index, element)) list.add(element) }
    return list.toCommonNDArray()
}

/**
 * Return a new array contains elements matching filter.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.filterNot(predicate: (T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    for (element in this) if (!predicate(element)) list.add(element)
    return list.toCommonNDArray()
}

/**
 * Returns the first element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.find(predicate: (T) -> Boolean): T? {
    return firstOrNull(predicate)
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.findLast(predicate: (T) -> Boolean): T? {
    return lastOrNull(predicate)
}

/**
 * Returns first element.
 * @throws [NoSuchElementException] if the collection is empty.
 */
public fun <T, D : Dimension> MultiArray<T, D>.first(): T {
    if (isEmpty()) throw NoSuchElementException("NDArray is empty.")
    return this.data[this.offset]
}

/**
 * Returns the first element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.first(predicate: (T) -> Boolean): T {
    for (element in this) if (predicate(element)) return element
    throw NoSuchElementException("NDArray contains no element matching the predicate.")
}

/**
 * Returns the first element, or `null` if the collection is empty.
 */
public fun <T, D : Dimension> MultiArray<T, D>.firstOrNull(): T? {
    return if (isEmpty()) null else return this.first()
}

/**
 * Returns the first element matching the given [predicate], or `null` if element was not found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.firstOrNull(predicate: (T) -> Boolean): T? {
    for (element in this) if (predicate(element)) return element
    return null
}

/**
 * Returns a flat ndarray of all elements resulting from calling the [transform] function on each element
 * in this ndarray.
 */
public inline fun <T, D : Dimension, reified R> MultiArray<T, D>.flatMap(transform: (T) -> Iterable<R>): D1Array<R> {
    val destination = ArrayList<R>()
    for (element in this) {
        val list = transform(element)
        destination.addAll(list)
    }
    return destination.toCommonNDArray()
}

/**
 * Returns a flat ndarray of all elements resulting from calling the [transform] function on each element and its single
 * index in this d1 ndarray.
 */
@JvmName("flatMapD1Indexed")
public inline fun <T, reified R> MultiArray<T, D1>.flatMapIndexed(transform: (index: Int, T) -> Iterable<R>): D1Array<R> {
    var index = 0
    val destination = ArrayList<R>()
    for (element in this) {
        val list = transform(checkIndexOverflow(index++), element)
        destination.addAll(list)
    }
    return destination.toCommonNDArray()
}

/**
 * Returns a flat ndarray of all elements resulting from calling the [transform] function on each element and its multi
 * index in this dn ndarray.
 */
@JvmName("flatMapDNIndexed")
public inline fun <T, reified R> MultiArray<T, D1>.flatMapMultiIndexed(transform: (index: IntArray, T) -> Iterable<R>): D1Array<R> {
    val indexIter = this.multiIndices.iterator()
    val destination = ArrayList<R>()
    for (element in this) {
        if (indexIter.hasNext()) {
            val list = transform(indexIter.next(), element)
            destination.addAll(list)
        } else {
            throw ArithmeticException("Index overflow has happened.")
        }
    }
    return destination.toCommonNDArray()
}


/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right to current accumulator value and each element.
 */
public inline fun <T, D : Dimension, R> MultiArray<T, D>.fold(initial: R, operation: (acc: R, T) -> R): R {
    var accumulator = initial
    for (element in this) accumulator = operation(accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original ndarray.
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
@JvmName("foldD1Indexed")
public inline fun <T, R> MultiArray<T, D1>.foldIndexed(
    initial: R, operation: (index: Int, acc: R, T) -> R
): R {
    var index = 0
    var accumulator = initial
    for (element in this) accumulator = operation(checkIndexOverflow(index++), accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original ndarray.
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
@JvmName("foldDNIndexed")
public inline fun <T, D : Dimension, R> MultiArray<T, D>.foldMultiIndexed(
    initial: R, operation: (index: IntArray, acc: R, T) -> R
): R {
    val indexIter = this.multiIndices.iterator()
    var accumulator = initial
    for (element in this) {
        if (indexIter.hasNext())
            accumulator = operation(indexIter.next(), accumulator, element)
        else
            throw ArithmeticException("Index overflow has happened.")
    }
    return accumulator
}

/**
 * Performs the given [action] on each element.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.forEach(action: (T) -> Unit) {
    for (element in this) action(element)
}

/**
 * Performs the given [action] on each element, providing sequential index with the element.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the desired action on the element.
 */
@JvmName("forEachD1Indexed")
public inline fun <T> MultiArray<T, D1>.forEachIndexed(action: (index: Int, T) -> Unit) {
    var index = 0
    for (item in this) action(checkIndexOverflow(index++), item)
}

/**
 * Performs the given [action] on each element, providing sequential index with the element.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the desired action on the element.
 */
@JvmName("forEachDNIndexed")
public inline fun <T, D : Dimension> MultiArray<T, D>.forEachMultiIndexed(action: (index: IntArray, T) -> Unit) {
    val indexIter = this.multiIndices.iterator()
    for (item in this) {
        if (indexIter.hasNext())
            action(indexIter.next(), item)
        else
            throw ArithmeticException("Index overflow has happened.")
    }
}


/**
 * Groups elements of a given ndarray by the key returned by [keySelector] for each element,
 * and returns a map where each group key is associated with an ndarray of matching elements.
 */
public inline fun <T, D : Dimension, K> MultiArray<T, D>.groupNDArrayBy(keySelector: (T) -> K): Map<K, NDArray<T, D1>> {
    return groupNDArrayByTo(LinkedHashMap(), keySelector)
}

/**
 * Groups values returned by [valueTransform] applied to each element of the given ndarray
 * with the key returned by [keySelector] applied to each element,
 * and returns a map where each group key is associated with an ndarray of matching values.
 */
public inline fun <T, D : Dimension, K, V : Number> MultiArray<T, D>.groupNDArrayBy(
    keySelector: (T) -> K, valueTransform: (T) -> V
): Map<K, NDArray<V, D1>> {
    return groupNDArrayByTo(LinkedHashMap(), keySelector, valueTransform)
}

/**
 * Groups elements of the given array by the key returned by [keySelector] function applied to each
 * element and puts to the [destination] map each group key associated with an ndarray of corresponding elements.
 */
public inline fun <T, D : Dimension, K, M : MutableMap<in K, NDArray<T, D1>>> MultiArray<T, D>.groupNDArrayByTo(
    destination: M, keySelector: (T) -> K
): M {
    val map = LinkedHashMap<K, MutableList<T>>()
    for (element in this) {
        val key = keySelector(element)
        val list = map.getOrPut(key) { ArrayList() }
        list.add(element)
    }
    for (item in map)
        destination.put(item.key, item.value.toCommonNDArray())
    return destination
}

/**
 * Groups values returned by the [valueTransform] function applied to each element of the given ndarray by the key
 * returned by [keySelector] function applied to the element and puts to the destination map each group key
 * associated with an ndarray of corresponding values.
 */
public inline fun <T, D : Dimension, K, V, M : MutableMap<in K, NDArray<V, D1>>> MultiArray<T, D>.groupNDArrayByTo(
    destination: M, keySelector: (T) -> K, valueTransform: (T) -> V
): M {
    val map = LinkedHashMap<K, MutableList<V>>()
    for (element in this) {
        val key = keySelector(element)
        val list = map.getOrPut(key) { ArrayList() }
        list.add(valueTransform(element))
    }
    for (item in map)
        destination.put(item.key, item.value.toCommonNDArray())
    return destination
}

/**
 * Creates a [Grouping] source from an ndarray to be used later with one of group-and-fold operations using the
 * specified [keySelector] function to extract a key from each element.
 */
public inline fun <T, D : Dimension, K> MultiArray<T, D>.groupingNDArrayBy(crossinline keySelector: (T) -> K): Grouping<T, K> {
    return object : Grouping<T, K> {
        override fun sourceIterator(): Iterator<T> = this@groupingNDArrayBy.iterator()
        override fun keyOf(element: T): K = keySelector(element)
    }
}

/**
 * Returns first index of [element], or -1 if the collection does not contain element.
 */
public fun <T, D : Dimension> MultiArray<T, D>.indexOf(element: T): Int {
    var index = 0
    for (item in this) {
        checkIndexOverflow(index)
        if (element == item)
            return index
        index++
    }
    return -1
}

/**
 * Returns index of the first element matching the given [predicate], or -1 if the collection does not contain such element.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.indexOfFirst(predicate: (T) -> Boolean): Int {
    var index = 0
    for (item in this) {
        checkIndexOverflow(index)
        if (predicate(item))
            return index
        index++
    }
    return -1
}

/**
 * Returns index of the last element matching the given [predicate], or -1 if the collection does not contain such element.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.indexOfLast(predicate: (T) -> Boolean): Int {
    var lastIndex = -1
    var index = 0
    for (item in this) {
        checkIndexOverflow(index)
        if (predicate(item))
            lastIndex = index
        index++
    }
    return lastIndex
}

/**
 * Returns a set containing all elements that are contained by both this collection and the specified collection.
 *
 * The returned set preserves the element iteration order of the original collection.
 *
 * To get a set containing all elements that are contained at least in one of these collections use [union].
 */
public infix fun <T, D : Dimension> MultiArray<T, D>.intersect(other: Iterable<T>): Set<T> {
    val set = this.toMutableSet()
    set.retainAll(other)
    return set
}

/**
 * Appends the string from all the elements separated using [separator] and using the given [prefix] and [postfix] if supplied.
 *
 * If the collection could be huge, you can specify a non-negative value of [limit], in which case only the first [limit]
 * elements will be appended, followed by the [truncated] string (which defaults to "...").
 */
public fun <T, D : Dimension, A : Appendable> MultiArray<T, D>.joinTo(
    buffer: A, separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "",
    limit: Int = -1, truncated: CharSequence = "...", transform: ((T) -> CharSequence)? = null
): A {
    buffer.append(prefix)
    var count = 0
    for (element in this) {
        if (++count > 1) buffer.append(separator)
        if (limit < 0 || count <= limit) {
            when {
                transform != null -> buffer.append(transform(element))
                element is CharSequence -> buffer.append(element)
                else -> buffer.append(element.toString())
            }
        } else break
    }
    if (limit in 0 until count) buffer.append(truncated)
    buffer.append(postfix)
    return buffer
}

/**
 * Creates a string from all the elements separated using [separator] and using the given [prefix] and [postfix] if supplied.
 *
 * If the collection could be huge, you can specify a non-negative value of [limit], in which case only the first [limit]
 * elements will be appended, followed by the [truncated] string (which defaults to "...").
 */
public fun <T, D : Dimension> MultiArray<T, D>.joinToString(
    separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "",
    limit: Int = -1, truncated: CharSequence = "...", transform: ((T) -> CharSequence)? = null
): String {
    return joinTo(StringBuilder(), separator, prefix, postfix, limit, truncated, transform).toString()
}


/**
 * Returns the last element.
 * @throws [NoSuchElementException] if the collection is empty.
 */
public fun <T, D : Dimension> MultiArray<T, D>.last(): T {
    if (isEmpty()) throw NoSuchElementException("NDArray is empty.")
    val index = IntArray(dim.d) { shape[it] - 1 }
    return this.asDNArray()[index]
}


/**
 * Returns the last element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.last(predicate: (T) -> Boolean): T {
    val ndarray = this.asDNArray()
    for (i in this.multiIndices.reverse) {
        val element = ndarray[i]
        if (predicate(element)) return element
    }
    throw NoSuchElementException("NDArray contains no element matching the predicate.")
}

/**
 * Returns last index of [element], or -1 if the collection does not contain element.
 */
public fun <T, D : Dimension> MultiArray<T, D>.lastIndexOf(element: T): Int {
    var lastIndex = -1
    var index = 0
    for (item in this) {
        if (index < 0) throw ArithmeticException("Index overflow has happened.")
        if (element == item)
            lastIndex = index
        index++
    }
    return lastIndex
}

/**
 * Returns the last element, or `null` if the list is empty.
 */
public fun <T, D : Dimension> MultiArray<T, D>.lastOrNull(): T? {
    return if (isEmpty()) null else this.asDNArray()[this.multiIndices.last]
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.lastOrNull(predicate: (T) -> Boolean): T? {
    var last: T? = null
    for (element in this) {
        if (predicate(element)) {
            last = element
        }
    }
    return last
}


/**
 * Return a new array contains elements after applying [transform].
 */
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.map(transform: (T) -> R): NDArray<R, D> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    var count = 0
    for (el in this)
        data[count++] = transform(el)
    return NDArray(data, shape = shape, dim = dim)
}

/**
 * Return a new array contains elements after applying [transform].
 */
@JvmName("mapD1Indexed")
public inline fun <T, reified R : Number> MultiArray<T, D1>.mapIndexed(transform: (index: Int, T) -> R): D1Array<R> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    var index = 0
    for (item in this)
        data[index] = transform(index++, item)
    return D1Array(data, shape = shape, dim = D1)
}

/**
 * Return a new array contains elements after applying [transform].
 */
@JvmName("mapDNIndexed")
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.mapMultiIndexed(transform: (index: IntArray, T) -> R): NDArray<R, D> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    val indexIter = this.multiIndices.iterator()
    var index = 0
    for (item in this) {
        if (indexIter.hasNext()) {
            data[index++] = transform(indexIter.next(), item)
        } else {
            throw ArithmeticException("Index overflow has happened.")
        }
    }
    return NDArray(data, shape = shape, dim = dim)
}

/**
 * Return a new array contains elements after applying [transform].
 */
@JvmName("mapD1IndexedNotNull")
public inline fun <T, reified R : Number> MultiArray<T, D1>.mapIndexedNotNull(transform: (index: Int, T) -> R?): D1Array<R> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    var count = 0
    forEachIndexed { index, element -> transform(index, element)?.let { data[count++] = it } }
    return D1Array(data, shape = shape, dim = D1)
}

/**
 * Return a new array contains elements after applying [transform].
 */
@JvmName("mapDNIndexedNotNull")
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.mapMultiIndexedNotNull(transform: (index: IntArray, T) -> R?): NDArray<R, D> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    var count = 0
    forEachMultiIndexed { index, element -> transform(index, element)?.let { data[count++] = it } }
    return NDArray(data, shape = shape.copyOf(), dim = dim)
}

/**
 * Return a new array contains elements after applying [transform].
 */
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.mapNotNull(transform: (T) -> R?): NDArray<R, D> {
    val newDtype = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(size, newDtype)
    var index = 0
    forEach { element -> transform(element)?.let { data[index++] = it } }
    return NDArray(data, shape = shape, dim = dim)
}

/**
 * Returns the largest element or `null` if there are no elements.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.max(): T? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var max = iterator.next()
    while (iterator.hasNext()) {
        val e = iterator.next()
        if (max < e) max = e
    }
    return max
}

/**
 * Returns the first element yielding the largest value of the given function or `null` if there are no elements.
 */
public inline fun <T, D : Dimension, R : Comparable<R>> MultiArray<T, D>.maxBy(selector: (T) -> R): T? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var maxElem = iterator.next()
    if (!iterator.hasNext()) return maxElem
    var maxValue = selector(maxElem)
    do {
        val e = iterator.next()
        val v = selector(e)
        if (maxValue < v) {
            maxElem = e
            maxValue = v
        }
    } while (iterator.hasNext())
    return maxElem
}

/**
 * Returns the first element having the largest value according to the provided [comparator] or `null` if there are no elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.maxWith(comparator: Comparator<in T>): T? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var max = iterator.next()
    while (iterator.hasNext()) {
        val e = iterator.next()
        if (comparator.compare(max, e) < 0) max = e
    }
    return max
}

/**
 * Returns the smallest element or `null` if there are no elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.min(): T? where T : Number, T : Comparable<T> {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var min = iterator.next()
    while (iterator.hasNext()) {
        val e = iterator.next()
        if (min > e) min = e
    }
    return min
}

/**
 * Returns the first element yielding the smallest value of the given function or `null` if there are no elements.
 */
public inline fun <T, D : Dimension, R : Comparable<R>> MultiArray<T, D>.minBy(selector: (T) -> R): T? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var minElem = iterator.next()
    if (!iterator.hasNext()) return minElem
    var minValue = selector(minElem)
    do {
        val e = iterator.next()
        val v = selector(e)
        if (minValue > v) {
            minElem = e
            minValue = v
        }
    } while (iterator.hasNext())
    return minElem
}

/**
 * Returns the first element having the smallest value according to the provided [comparator] or `null` if there are no elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.minWith(comparator: Comparator<in T>): T? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var min = iterator.next()
    while (iterator.hasNext()) {
        val e = iterator.next()
        if (comparator.compare(min, e) > 0) min = e
    }
    return min
}

/**
 * Performs the given [action] on each element and returns the collection itself afterwards.
 */
public inline fun <T, D : Dimension, C : MultiArray<T, D>> C.onEach(action: (T) -> Unit): C {
    return apply { for (element in this) action(element) }
}

/**
 * Splits the original collection into pair of lists,
 * where *first* list contains elements for which [predicate] yielded `true`,
 * while *second* list contains elements for which [predicate] yielded `false`.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.partition(predicate: (T) -> Boolean): Pair<NDArray<T, D1>, NDArray<T, D1>> {
    val first = ArrayList<T>()
    val second = ArrayList<T>()
    for (element in this) {
        if (predicate(element)) {
            first.add(element)
        } else {
            second.add(element)
        }
    }
    return Pair(first.toCommonNDArray(), second.toCommonNDArray())
}

/**
 * Returns a 2-D ndarray of window segments of the specified size, sliding over this ndarray with the specified step.
 *
 * The last few arrays are filled with zeros if limit is false.
 * The [size] and [step] must be positive and can be greater than the number of elements in this ndarray.
 *
 * @param size the size and step must be positive and can be greater than the number of elements in this array
 * @param step the number of elements to move the window forward by on an each step, by default 1
 * @param limit sets a limit on the set of significant elements in the result, otherwise it fills in with zeros
 */
public fun <T, D : Dimension> MultiArray<T, D>.windowed(
    size: Int, step: Int = 1, limit: Boolean = true
): NDArray<T, D2> {
    require(size > 0 && step > 0) {
        if (size != step)
            "Both size $size and step $step must be greater than zero."
        else
            "size $size must be greater than zero."
    }

    val thisSize = this.size
    val rSize = min(thisSize, size)
    val rStep = min(thisSize, step)
    val resultCapacity = when {
        limit -> thisSize / rStep * rSize
        thisSize % rStep == 0 -> thisSize / rStep * rSize
        else -> (thisSize / rStep + 1) * rSize
    }
    val resData = initMemoryView<T>(resultCapacity, this.dtype)
    val thisNDArray = this.flatten()
    var index = 0
    var resIndex = 0
    while (index in 0 until thisSize) {
        for (i in 0 until rSize) {
            if (i + index >= thisSize) {
                resIndex++
                continue
            }
            resData[resIndex++] = thisNDArray[i + index]
        }
        index += rStep
    }
    return D2Array(resData, 0, intArrayOf(resultCapacity / rSize, rSize), dim = D2)
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right to current accumulator value and each element.
 */
public inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduce(operation: (acc: S, T) -> S): S {
    val iterator = this.iterator()
    if (!iterator.hasNext()) throw UnsupportedOperationException("Empty ndarray can't be reduced.")
    var accumulator: S = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(accumulator, iterator.next())
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original collection.
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself and calculates the next accumulator value.
 */
@JvmName("reduceD1Indexed")
public inline fun <S, T : S> MultiArray<T, D1>.reduceIndexed(operation: (index: Int, acc: S, T) -> S): S {
    val iterator = this.iterator()
    if (!iterator.hasNext()) throw UnsupportedOperationException("Empty ndarray can't be reduced.")
    var index = 1
    var accumulator: S = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(checkIndexOverflow(index++), accumulator, iterator.next())
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original collection.
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself and calculates the next accumulator value.
 */
@JvmName("reduceDNIndexed")
public inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduceMultiIndexed(operation: (index: IntArray, acc: S, T) -> S): S {
    val iterator = this.iterator()
    if (!iterator.hasNext()) throw UnsupportedOperationException("Empty ndarray can't be reduced.")
    val indexIter = this.multiIndices.iterator()
    var accumulator: S = iterator.next()
    while (iterator.hasNext() && indexIter.hasNext()) {
        accumulator = operation(indexIter.next(), accumulator, iterator.next())
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right to current accumulator value and each element. Returns null if the collection is empty.
 */
public inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduceOrNull(operation: (acc: S, T) -> S): S? {
    val iterator = this.iterator()
    if (!iterator.hasNext()) return null
    var accumulator: S = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(accumulator, iterator.next())
    }
    return accumulator
}

/**
 *
 */
public fun <T, D : Dimension> MultiArray<T, D>.reversed(): NDArray<T, D> {
    if (size <= 1) return this.copy() as NDArray<T, D>
    val data = initMemoryView<T>(this.size, this.dtype)
    var index = this.size - 1
    for (element in this)
        data[index--] = element
    return NDArray(data, 0, this.shape.copyOf(), dim = this.dim)
}


/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting ndarray.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.scan(
    initial: R, operation: (acc: R, T) -> R
): NDArray<R, D> {
    val dataType = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(this.size + 1, dataType)
    data[0] = initial
    var index = 1
    var accumulator = initial
    for (element in this) {
        accumulator = operation(accumulator, element)
        data[index++] = accumulator
    }
    return NDArray(data, 0, this.shape.copyOf(), dim = this.dim)
}

/**
 * Return a flat ndarray containing successive accumulation values generated by applying [operation] from left to right to
 * each element, its index in this d1 ndarray and current accumulator value that starts with [initial] value.
 */
@JvmName("scanD1Indexed")
public inline fun <T, reified R : Number> MultiArray<T, D1>.scanIndexed(
    initial: R, operation: (index: Int, acc: R, T) -> R
): D1Array<R> {
    val dataType = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(this.size + 1, dataType)
    data[0] = initial
    var count = 1
    var accumulator = initial
    val ndarrayIter = this.iterator()
    while (ndarrayIter.hasNext()) {
        accumulator = operation(count, accumulator, ndarrayIter.next())
        data[count++] = accumulator
    }
    return D1Array(data, 0, this.shape.copyOf(), dim = D1)
}

/**
 * Return a flat ndarray containing successive accumulation values generated by applying [operation] from left to right to
 * each element, its multi index in this dn ndarray and current accumulator value that starts with [initial] value.
 */
public inline fun <T, D : Dimension, reified R : Number> MultiArray<T, D>.scanMultiIndexed(
    initial: R, operation: (index: IntArray, acc: R, T) -> R
): NDArray<R, D> {
    val dataType = DataType.ofKClass(R::class)
    val data = initMemoryView<R>(this.size + 1, dataType)
    data[0] = initial
    var count = 1
    var accumulator = initial
    val ndarrayIter = this.iterator()
    val indexIter = this.multiIndices.iterator()
    while (ndarrayIter.hasNext() && indexIter.hasNext()) {
        accumulator = operation(indexIter.next(), accumulator, ndarrayIter.next())
        data[count++] = accumulator
    }
    return NDArray(data, 0, this.shape.copyOf(), dim = this.dim)
}

/**
 *
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.sorted(): NDArray<T, D> {
    val ret = this.deepCopy() as NDArray<T, D>
    when (this.dtype) {
        DataType.ByteDataType -> ret.data.getByteArray().sort()
        DataType.ShortDataType -> ret.data.getShortArray().sort()
        DataType.IntDataType -> ret.data.getIntArray().sort()
        DataType.LongDataType -> ret.data.getLongArray().sort()
        DataType.FloatDataType -> ret.data.getFloatArray().sort()
        DataType.DoubleDataType -> ret.data.getDoubleArray().sort()
        DataType.ComplexFloatDataType, DataType.ComplexDoubleDataType ->
            throw Exception("Complex numbers cannot be sorted.")
    }
    return ret
}

/**
 * Returns the sum of all elements in the collection.
 */
@Suppress("UNCHECKED_CAST")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sum(): T = mk.math.sum(this)

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> Int): Int {
    var sum = 0
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> Double): Double {
    var sum = 0.0
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> ComplexFloat): ComplexFloat {
    var sum = ComplexFloat.zero
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> ComplexDouble): ComplexDouble {
    var sum = ComplexDouble.zero
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Appends all elements to the given [destination] collection.
 */
public fun <T, D : Dimension, C : MutableCollection<in T>> MultiArray<T, D>.toCollection(destination: C): C {
    for (item in this) {
        destination.add(item)
    }
    return destination
}

/**
 * Returns a [HashSet] of all elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toHashSet(): HashSet<T> {
    return toCollection(HashSet(mapCapacity(size)))
}

/**
 * Returns a [List] containing all elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toList(): List<T> {
    return when (size) {
        0 -> emptyList()
        1 -> listOf(this.first())
        else -> this.toMutableList()
    }
}

/**
 * Returns a [MutableList] filled with all elements of this collection.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toMutableList(): MutableList<T> {
    return toCollection(ArrayList())
}

/**
 * Returns a mutable set containing all distinct elements from the given collection.
 *
 * The returned set preserves the element iteration order of the original collection.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toMutableSet(): MutableSet<T> {
    return toCollection(LinkedHashSet())
}

/**
 * Returns a [Set] of all elements.
 *
 * The returned set preserves the element iteration order of the original collection.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toSet(): Set<T> {
    return when (size) {
        0 -> emptySet()
        1 -> setOf(this.first())
        else -> toCollection(LinkedHashSet(mapCapacity(size)))
    }
}

/**
 *
 */
public enum class CopyStrategy { FULL, MEANINGFUL }

/**
 *
 */
public inline fun <T, reified O : Any, D : Dimension> MultiArray<T, D>.toType(copy: CopyStrategy = CopyStrategy.FULL): NDArray<O, D> {
    val dtype = DataType.ofKClass(O::class)
    return this.toType(dtype, copy)
}

/**
 *
 */
public fun <T, O : Any, D : Dimension> MultiArray<T, D>.toType(dtype: DataType, copy: CopyStrategy = CopyStrategy.FULL): NDArray<O, D> {
    if (this.dtype == dtype) {
        return (if (copy == CopyStrategy.FULL) this.copy() else this.deepCopy()) as NDArray<O, D>
    }

    val size: Int
    val iterData: Iterator<T>
    val strides: IntArray

    if (copy == CopyStrategy.FULL) {
        size = this.data.size
        iterData = this.data.iterator()
        strides = this.strides.copyOf()
    } else {
        size = this.size
        iterData = this.iterator()
        strides = computeStrides(this.shape)
    }

    val isNumber = this.dtype.isNumber()
    val view = initMemoryView<O>(size, dtype)

    when {
        isNumber && dtype == DataType.FloatDataType -> {
            val d = view.getFloatArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toFloat()
                }
            }
        }
        isNumber && dtype == DataType.DoubleDataType -> {
            val d = view.getDoubleArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toDouble()
                }
            }
        }
        isNumber && dtype == DataType.IntDataType -> {
            val d = view.getIntArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toInt()
                }
            }
        }
        isNumber && dtype == DataType.LongDataType -> {
            val d = view.getLongArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toLong()
                }
            }
        }
        !isNumber && dtype == DataType.ComplexFloatDataType -> {
            val d = view.getComplexFloatArray()
            var count = 0
            (iterData as Iterator<ComplexDouble>).apply {
                while (hasNext()) {
                    val c = next()
                    d[count++] = ComplexFloat(c.re, c.im)
                }
            }
        }
        !isNumber && dtype == DataType.ComplexDoubleDataType -> {
            val d = view.getComplexDoubleArray()
            var count = 0
            (iterData as Iterator<ComplexFloat>).apply {
                while (hasNext()) {
                    val c = next()
                    d[count++] = ComplexDouble(c.re, c.im)
                }
            }
        }
        isNumber && dtype == DataType.ShortDataType -> {
            val d = view.getShortArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toShort()
                }
            }
        }
        isNumber && dtype == DataType.ByteDataType -> {
            val d = view.getByteArray()
            var count = 0
            (iterData as Iterator<Number>).apply {
                while (hasNext()) {
                    d[count++] = next().toByte()
                }
            }
        }
        else -> throw Exception()
    }
    return NDArray(view, this.offset, this.shape.copyOf(), strides, this.dim)
}

@PublishedApi
internal fun mapCapacity(size: Int): Int {
    return when {
        size < 3 -> size + 1
        size < 1 shl (Int.SIZE_BITS - 2) -> ((size / 0.75F) + 1.0F).toInt()
        else -> Int.MAX_VALUE
    }
}

@Suppress("NOTHING_TO_INLINE")
@PublishedApi
internal inline fun checkIndexOverflow(index: Int): Int {
    if (index < 0) throw ArithmeticException("Index overflow has happened.")
    return index
}