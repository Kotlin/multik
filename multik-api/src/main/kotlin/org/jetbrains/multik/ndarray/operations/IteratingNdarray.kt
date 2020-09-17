package org.jetbrains.multik.ndarray.operations

import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.api.toNdarray
import org.jetbrains.multik.ndarray.data.*

//TODO(check all methods)

/**
 * Returns `true` if all elements match the given [predicate].
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.all(predicate: (T) -> Boolean): Boolean {
    if (isEmpty()) return true
    for (element in this) if (!predicate(element)) return false
    return true
}

/**
 * Returns `true` if collection has at least one element.
 */
public fun <T : Number, D : Dimension> Ndarray<T, D>.any(): Boolean {
    return !isEmpty()
}

/**
 * Returns `true` if at least one element matches the given [predicate].
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.any(predicate: (T) -> Boolean): Boolean {
    if (isEmpty()) return false
    for (element in this) if (predicate(element)) return true
    return false
}

/**
 * Creates a [Sequence] instance that wraps the original collection returning its elements when being iterated.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.asSequence(): Sequence<T> {
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
public inline fun <T : Number, D : Dimension, K, V> MultiArray<T, D>.associate(transform: (T) -> Pair<K, V>): Map<K, V> {
    val capacity = mapCapacity(this.size).coerceAtLeast(16)
    return associateTo(LinkedHashMap<K, V>(capacity), transform)
}

/**
 * Returns a [Map] containing the elements from the given collection indexed by the key
 * returned from [keySelector] function applied to each element.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <T : Number, D : Dimension, K> MultiArray<T, D>.associateBy(keySelector: (T) -> K): Map<K, T> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap<K, T>(capacity), keySelector)
}

/**
 * Returns a [Map] containing the values provided by [valueTransform] and indexed by [keySelector] functions applied to elements of the given collection.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original collection.
 */
public inline fun <T : Number, D : Dimension, K, V> MultiArray<T, D>.associateBy(
    keySelector: (T) -> K,
    valueTransform: (T) -> V
): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap<K, V>(capacity), keySelector, valueTransform)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function applied to each element of the given collection
 * and value is the element itself.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <T : Number, D : Dimension, K, M : MutableMap<in K, in T>> MultiArray<T, D>.associateByTo(
    destination: M,
    keySelector: (T) -> K
): M {
    for (element in this) {
        destination.put(keySelector(element), element)
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function and
 * and value is provided by the [valueTransform] function applied to elements of the given collection.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <T : Number, D : Dimension, K, V, M : MutableMap<in K, in V>> MultiArray<T, D>.associateByTo(
    destination: M,
    keySelector: (T) -> K,
    valueTransform: (T) -> V
): M {
    for (element in this) {
        destination.put(keySelector(element), valueTransform(element))
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs
 * provided by [transform] function applied to each element of the given collection.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 */
public inline fun <T : Number, D : Dimension, K, V, M : MutableMap<in K, in V>> MultiArray<T, D>.associateTo(
    destination: M,
    transform: (T) -> Pair<K, V>
): M {
    for (element in this) {
        destination += transform(element)
    }
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
public inline fun <K : Number, D : Dimension, V> MultiArray<K, D>.associateWith(valueSelector: (K) -> V): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateWithTo(LinkedHashMap<K, V>(capacity), valueSelector)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs for each element of the given collection,
 * where key is the element itself and value is provided by the [valueSelector] function applied to that key.
 *
 * If any two elements are equal, the last one overwrites the former value in the map.
 */
public inline fun <K : Number, D : Dimension, V, M : MutableMap<in K, in V>> MultiArray<K, D>.associateWithTo(
    destination: M,
    valueSelector: (K) -> V
): M {
    for (element in this) {
        destination.put(element, valueSelector(element))
    }
    return destination
}

/**
 * Returns an average value of elements in the collection.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.average(): Double {
    var sum: Double = 0.0
    var count: Int = 0
    for (element in this) {
        sum += element.toDouble()
        if (++count < 0) throw ArithmeticException("Count overflow has happened.")
    }
    return if (count == 0) Double.NaN else sum / count
}


//todo (chunked())
///**
// * Splits this collection into a list of lists each not exceeding the given [size].
// *
// * The last list in the resulting list may have less elements than the given [size].
// *
// * @param size the number of elements to take in each list, must be positive and can be greater than the number of elements in this collection.
// */
//public fun <T : Number> MultiArray<T, D2>.chunked(): List<List<T>> {
//    val firstSize = this.shape[0]
//    val secondSize = this.shape[1]
//    val result = ArrayList<List<T>>(firstSize)
//    for (index in 0 until size step secondSize) {
//        result.add(List(secondSize) { this[it + index] })
//    }
//    return result
//}

//TODO
///**
// * Splits this collection into several lists each not exceeding the given [size]
// * and applies the given [transform] function to an each.
// *
// * @return list of results of the [transform] applied to an each list.
// *
// * Note that the list passed to the [transform] function is ephemeral and is valid only inside that function.
// * You should not store it or allow it to escape in some way, unless you made a snapshot of it.
// * The last list may have less elements than the given [size].
// *
// * @param size the number of elements to take in each list, must be positive and can be greater than the number of elements in this collection.
// */
//public fun <T: Number, R> MultiArray<T, D2>.chunked(transform: (List<T>) -> R): List<R> {
//    val firstSize = this.shape[0]
//    val secondSize = this.shape[1]
//    val result = ArrayList<List<T>>(firstSize)
//    for (index in 0 until size step secondSize) {
//        result.add(List(secondSize) { transform(this[it + index]) })
//    }
//    return result
//}

/**
 * Returns `true` if [element] is found in the collection.
 */
public operator fun <T : Number, D : Dimension> MultiArray<T, D>.contains(element: T): Boolean {
    return indexOf(element) >= 0
}

/**
 * Returns the number of elements matching the given [predicate].
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.count(predicate: (T) -> Boolean): Int {
    if (isEmpty()) return 0
    var count = 0
    for (element in this) if (predicate(element)) if (++count < 0) throw ArithmeticException("Count overflow has happened.")
    return count
}

/**
 * Returns a new array containing only distinct elements from the given array.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.distinct(): Ndarray<T, D1> {
    return this.toMutableSet().toNdarray()
}

/**
 * Returns a new array containing only elements from the given array
 * having distinct keys returned by the given [selector] function.
 */
public inline fun <T : Number, D : Dimension, K> MultiArray<T, D>.distinctBy(selector: (T) -> K): Ndarray<T, D1> {
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
public fun <T : Number> MultiArray<T, D1>.drop(n: Int): D1Array<T> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    val resultSize = size - n
    val d = initMemoryView<T>(resultSize, dtype) { this@drop.data[it + n] }
    val shape = intArrayOf(resultSize)
    return D1Array(d, shape = shape, dtype = dtype, dim = D1)
}

/**
 *
 */
public inline fun <T : Number> MultiArray<T, D1>.dropWhile(predicate: (T) -> Boolean): Ndarray<T, D1> {
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
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.filter(predicate: (T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    forEach { if (predicate(it)) list.add(it) }
    val data = list.toViewPrimitiveArray(this.dtype)
    return D1Array<T>(data, 0, intArrayOf(data.size), dtype = dtype, dim = D1)
}

/**
 * Return a new array contains elements matching filter.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.filterIndexed(predicate: (index: Int, T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    forEachIndexed { index, element -> if (predicate(index, element)) list.add(element) }
    val data = list.toViewPrimitiveArray(dtype)
    return D1Array<T>(data, shape = intArrayOf(data.size), dtype = dtype, dim = D1)
}

//todo (View and slice)
/**
 * Appends elements matching filter to [destination].
 */
public inline fun <T : Number, D : Dimension, C : D1Array<T>> MultiArray<T, D>.filterIndexedTo(
    destination: C,
    predicate: (index: Int, T) -> Boolean
): C {
    var count = 0
    forEachIndexed { index, element -> if (predicate(index, element)) destination[count++] = element }
    return destination
}

/**
 * Return a new array contains elements matching filter.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.filterNot(predicate: (T) -> Boolean): D1Array<T> {
    val list = ArrayList<T>()
    for (element in this) if (!predicate(element)) list.add(element)
    val data = list.toViewPrimitiveArray(dtype)
    return D1Array<T>(data, shape = intArrayOf(data.size), dtype = dtype, dim = D1)
}

//todo (View and slice)
/**
 * Appends elements matching filter to [destination].
 */
public inline fun <T : Number, D : Dimension, C : D1Array<T>> MultiArray<T, D>.filterNotTo(
    destination: C,
    predicate: (T) -> Boolean
): C {
    var count = 0
    for (element in this) if (!predicate(element)) destination[count++] = element
    return destination
}

//todo check size
/**
 * Appends elements matching filter to [destination].
 */
public inline fun <T : Number, D : Dimension, C : D1Array<T>> MultiArray<T, D>.filterTo(
    destination: C,
    predicate: (T) -> Boolean
): C {
    var count = 0
    for (element in this) if (predicate(element)) destination[count++] = element
    return destination
}

/**
 * Returns the first element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.find(predicate: (T) -> Boolean): T? {
    return firstOrNull(predicate)
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.findLast(predicate: (T) -> Boolean): T? {
    return lastOrNull(predicate)
}

/**
 * Returns first element.
 * @throws [NoSuchElementException] if the collection is empty.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.first(): T {
    if (isEmpty()) throw NoSuchElementException("Ndarray is empty.")
    return this.data[this.offset]
}

/**
 * Returns the first element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.first(predicate: (T) -> Boolean): T {
    for (element in this) if (predicate(element)) return element
    throw NoSuchElementException("Ndarray contains no element matching the predicate.")
}

/**
 * Returns the first element, or `null` if the collection is empty.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.firstOrNull(): T? {
    return if (isEmpty()) null else return this.first()
}

/**
 * Returns the first element matching the given [predicate], or `null` if element was not found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.firstOrNull(predicate: (T) -> Boolean): T? {
    for (element in this) if (predicate(element)) return element
    return null
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
 * Accumulates value starting with [initial] value and applying [operation] from left to right to current accumulator value and each element.
 */
public inline fun <T : Number, D : Dimension, R> MultiArray<T, D>.fold(initial: R, operation: (acc: R, T) -> R): R {
    var accumulator = initial
    for (element in this) accumulator = operation(accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original collection.
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <T : Number, D : Dimension, R> MultiArray<T, D>.foldIndexed(
    initial: R,
    operation: (index: Int, acc: R, T) -> R
): R {
    var index = 0
    var accumulator = initial
    for (element in this) accumulator = operation(
        if (index++ < 0) throw ArithmeticException("Index overflow has happened.") else index,
        accumulator,
        element
    )
    return accumulator
}

/**
 * Performs the given [action] on each element.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.forEach(action: (T) -> Unit): Unit {
    for (element in this) action(element)
}

/**
 * Performs the given [action] on each element, providing sequential index with the element.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the desired action on the element.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.forEachIndexed(action: (index: Int, T) -> Unit): Unit {
    var index = 0
    for (item in this) action(
        if (index < 0) throw ArithmeticException("Index overflow has happened.") else index++,
        item
    )
}


/**
 *
 */
public inline fun <T : Number, D : Dimension, K> MultiArray<T, D>.groupNdarrayBy(keySelector: (T) -> K): Map<K, Ndarray<T, D1>> {
    return groupNdarrayByTo(LinkedHashMap<K, Ndarray<T, D1>>(), keySelector)
}

public inline fun <T : Number, D : Dimension, K, V : Number> MultiArray<T, D>.groupNdarrayBy(
    keySelector: (T) -> K,
    valueTransform: (T) -> V
): Map<K, Ndarray<V, D1>> {
    return groupNdarrayByTo(LinkedHashMap<K, Ndarray<V, D1>>(), keySelector, valueTransform)
}

//todo(add?)
public inline fun <T : Number, D : Dimension, K, M : MutableMap<in K, Ndarray<T, D1>>> MultiArray<T, D>.groupNdarrayByTo(
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
public inline fun <T : Number, D : Dimension, K, V : Number, M : MutableMap<in K, Ndarray<V, D1>>> MultiArray<T, D>.groupNdarrayByTo(
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

public inline fun <T : Number, D : Dimension, K> MultiArray<T, D>.groupingNdarrayBy(crossinline keySelector: (T) -> K): Grouping<T, K> {
    return object : Grouping<T, K> {
        override fun sourceIterator(): Iterator<T> = this@groupingNdarrayBy.iterator()
        override fun keyOf(element: T): K = keySelector(element)
    }
}

/**
 * Returns first index of [element], or -1 if the collection does not contain element.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.indexOf(element: T): Int {
    var index = 0
    for (item in this) {
        if (index < 0) throw ArithmeticException("Index overflow has happened.")
        if (element == item)
            return index
        index++
    }
    return -1
}

/**
 * Returns index of the first element matching the given [predicate], or -1 if the collection does not contain such element.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.indexOfFirst(predicate: (T) -> Boolean): Int {
    var index = 0
    for (item in this) {
        if (index < 0) throw ArithmeticException("Index overflow has happened.")
        if (predicate(item))
            return index
        index++
    }
    return -1
}

/**
 * Returns index of the last element matching the given [predicate], or -1 if the collection does not contain such element.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.indexOfLast(predicate: (T) -> Boolean): Int {
    var lastIndex = -1
    var index = 0
    for (item in this) {
        if (index < 0) throw ArithmeticException("Index overflow has happened.")
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
public infix fun <T : Number, D : Dimension> MultiArray<T, D>.intersect(other: Iterable<T>): Set<T> {
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
public fun <T : Number, D : Dimension, A : Appendable> MultiArray<T, D>.joinTo(
    buffer: A,
    separator: CharSequence = ", ",
    prefix: CharSequence = "",
    postfix: CharSequence = "",
    limit: Int = -1,
    truncated: CharSequence = "...",
    transform: ((T) -> CharSequence)? = null
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
public fun <T : Number, D : Dimension> MultiArray<T, D>.joinToString(
    separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "",
    limit: Int = -1, truncated: CharSequence = "...", transform: ((T) -> CharSequence)? = null
): String {
    return joinTo(StringBuilder(), separator, prefix, postfix, limit, truncated, transform).toString()
}


// todo (last!!!)
/**
 * Returns the last element.
 * @throws [NoSuchElementException] if the collection is empty.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.last(): T {
    if (isEmpty()) throw NoSuchElementException("Ndarray is empty.")
    return this.asDNArray()[size - 1]
}


// todo (last!!!)
/**
 * Returns the last element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.last(predicate: (T) -> Boolean): T {
    this as MultiArray<T, DN>
    for (i in size - 1..0) {
        val element = this[i]
        if (predicate(element)) return element
    }
    throw NoSuchElementException("Ndarray contains no element matching the predicate.")
}

/**
 * Returns last index of [element], or -1 if the collection does not contain element.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.lastIndexOf(element: T): Int {
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

// todo (last!!!)
/**
 * Returns the last element, or `null` if the list is empty.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.lastOrNull(): T? {
    this as MultiArray<T, DN>
    return if (isEmpty()) null else this[size - 1]
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.lastOrNull(predicate: (T) -> Boolean): T? {
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
public inline fun <T : Number, D : Dimension, reified R : Number> MultiArray<T, D>.map(transform: (T) -> R): Ndarray<R, D> {
    val newDtype = DataType.of(R::class)
    val data = initMemoryView<R>(size, newDtype)
    return mapTo(Ndarray<R, D>(data, shape = shape, dtype = newDtype, dim = dim), transform)
}

/**
 * Return a new array contains elements after applying [transform].
 */
public inline fun <T : Number, D : Dimension, reified R : Number> MultiArray<T, D>.mapIndexed(transform: (index: Int, T) -> R): Ndarray<R, D> {
    val newDtype = DataType.of(R::class)
    val data = initMemoryView<R>(size, newDtype)
    return mapIndexedTo(Ndarray<R, D>(data, shape = shape, dtype = newDtype, dim = dim), transform)
}

/**
 * Return a new array contains elements after applying [transform].
 */
public inline fun <T : Number, D : Dimension, reified R : Number> MultiArray<T, D>.mapIndexedNotNull(transform: (index: Int, T) -> R?): Ndarray<R, D> {
    val newDtype = DataType.of(R::class)
    val data = initMemoryView<R>(size, newDtype)
    return mapIndexedNotNullTo(Ndarray<R, D>(data, shape = shape, dtype = newDtype, dim = dim), transform)
}

/**
 * Appends elements after applying [transform] to [destination].
 */
public inline fun <T : Number, D : Dimension, R : Any, C : Ndarray<in R, D>> MultiArray<T, D>.mapIndexedNotNullTo(
    destination: C,
    transform: (index: Int, T) -> R?
): C {
    var count = 0
    forEachIndexed { index, element ->
        transform(index, element)?.let {
            destination.data[count++] = it
        }
    }
    return destination
}

/**
 * Appends elements after applying [transform] to [destination].
 */
public inline fun <T : Number, D : Dimension, R, C : Ndarray<in R, D>> MultiArray<T, D>.mapIndexedTo(
    destination: C,
    transform: (index: Int, T) -> R
): C {
    var index = 0
    for (item in this)
        destination.data[index] = transform(index++, item)
    return destination
}

/**
 * Return a new array contains elements after applying [transform].
 */
public inline fun <T : Number, D : Dimension, reified R : Number> MultiArray<T, D>.mapNotNull(transform: (T) -> R?): Ndarray<R, D> {
    val newDtype = DataType.of(R::class)
    val data = initMemoryView<R>(size, newDtype)
    return mapNotNullTo(Ndarray<R, D>(data, shape = shape, dtype = newDtype, dim = dim), transform)
}

/**
 * Appends elements after applying [transform] to [destination].
 */
public inline fun <T : Number, D : Dimension, R : Any, C : Ndarray<in R, D>> MultiArray<T, D>.mapNotNullTo(
    destination: C,
    transform: (T) -> R?
): C {
    var index = 0
    forEach { element -> transform(element)?.let { destination.data[index++] = it } }
    return destination
}


// todo(map, dimension)
/**
 * Appends elements after applying [transform] to [destination].
 */
public inline fun <T : Number, D : Dimension, R : Number, C : Ndarray<R, D>> MultiArray<T, D>.mapTo(
    destination: C, transform: (T) -> R
): C {
    val op = this.asDNArray()
    val des = destination.asDNArray()
    for (i in this.multiIndices)
        des[i] = transform(op[i])
    return destination
}

/**
 * Returns the largest element or `null` if there are no elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.max(): T? where T : Number, T : Comparable<T> {
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
public inline fun <T : Number, D : Dimension, R : Comparable<R>> MultiArray<T, D>.maxBy(selector: (T) -> R): T? {
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
public fun <T : Number, D : Dimension> MultiArray<T, D>.maxWith(comparator: Comparator<in T>): T? {
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
public inline fun <T : Number, D : Dimension, R : Comparable<R>> MultiArray<T, D>.minBy(selector: (T) -> R): T? {
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
public fun <T : Number, D : Dimension> MultiArray<T, D>.minWith(comparator: Comparator<in T>): T? {
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
public inline fun <T : Number, D : Dimension, C : MultiArray<T, D>> C.onEach(action: (T) -> Unit): C {
    return apply { for (element in this) action(element) }
}

/**
 * Splits the original collection into pair of lists,
 * where *first* list contains elements for which [predicate] yielded `true`,
 * while *second* list contains elements for which [predicate] yielded `false`.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.partition(predicate: (T) -> Boolean): Pair<Ndarray<T, D1>, Ndarray<T, D1>> {
    val first = ArrayList<T>()
    val second = ArrayList<T>()
    for (element in this) {
        if (predicate(element)) {
            first.add(element)
        } else {
            second.add(element)
        }
    }
    return Pair(first.toNdarray(), second.toNdarray())
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right to current accumulator value and each element.
 */
public inline fun <S : Number, D : Dimension, T : S> MultiArray<T, D>.reduce(operation: (acc: S, T) -> S): S {
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
public inline fun <S : Number, D : Dimension, T : S> MultiArray<T, D>.reduceIndexed(operation: (index: Int, acc: S, T) -> S): S {
    val iterator = this.iterator()
    if (!iterator.hasNext()) throw UnsupportedOperationException("Empty ndarray can't be reduced.")
    var index = 1
    var accumulator: S = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(
            if (index++ < 0) throw ArithmeticException("Index overflow has happened.") else index,
            accumulator,
            iterator.next()
        )
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right to current accumulator value and each element. Returns null if the collection is empty.
 */
public inline fun <S : Number, D : Dimension, T : S> MultiArray<T, D>.reduceOrNull(operation: (acc: S, T) -> S): S? {
    val iterator = this.iterator()
    if (!iterator.hasNext()) return null
    var accumulator: S = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(accumulator, iterator.next())
    }
    return accumulator
}

//TODO (create view: swap position and limit)
//public fun <T: Number, D: Dimension> MultiArray<T, D>.reversed(): Ndarray<T, D> {}


//todo (shape and dim?)
/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <T : Number, D : Dimension, reified R : Number> MultiArray<T, D>.scan(
    initial: R,
    operation: (acc: R, T) -> R
): Ndarray<R, D> {
    val retList = ArrayList<R>(size + 1).apply { add(initial) }
    var accumulator = initial
    for (element in this) {
        accumulator = operation(accumulator, element)
        retList.add(accumulator)
    }
    val dataType = DataType.of(R::class)
    val data = retList.toViewPrimitiveArray(dataType)
    return Ndarray<R, D>(data, shape = shape, dtype = dataType, dim = dim)
}

//TODO sort

/**
 * Returns the sum of all elements in the collection.
 */
@Suppress("UNCHECKED_CAST")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sum(): T {
    var sum: Number = zeroNumber(this.dtype)
    for (element in this) {
        sum += element
    }
    return sum as T
}

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> Int): Int {
    var sum: Int = 0
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Returns the sum of all values produced by [selector] function applied to each element in the collection.
 */
public inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(selector: (T) -> Double): Double {
    var sum: Double = 0.0
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

/**
 * Appends all elements to the given [destination] collection.
 */
public fun <T : Number, D : Dimension, C : MutableCollection<in T>> MultiArray<T, D>.toCollection(destination: C): C {
    for (item in this) {
        destination.add(item)
    }
    return destination
}

/**
 * Returns a [HashSet] of all elements.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toHashSet(): HashSet<T> {
    return toCollection(HashSet<T>(mapCapacity(size)))
}

/**
 * Returns a [List] containing all elements.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toList(): List<T> {
    return when (size) {
        0 -> emptyList()
        1 -> listOf(this.first())
        else -> this.toMutableList()
    }
}

/**
 * Returns a [MutableList] filled with all elements of this collection.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toMutableList(): MutableList<T> {
    return toCollection(ArrayList<T>())
}

/**
 * Returns a mutable set containing all distinct elements from the given collection.
 *
 * The returned set preserves the element iteration order of the original collection.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toMutableSet(): MutableSet<T> {
    return toCollection(LinkedHashSet<T>())
}

/**
 * Returns a [Set] of all elements.
 *
 * The returned set preserves the element iteration order of the original collection.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toSet(): Set<T> {
    return when (size) {
        0 -> emptySet()
        1 -> setOf(this.first())
        else -> toCollection(LinkedHashSet<T>(mapCapacity(size)))
    }
}

/**
 * Returns a [SortedSet][java.util.SortedSet] of all elements.
 */
public fun <T, D : Dimension> MultiArray<T, D>.toSortedSet(): java.util.SortedSet<T> where T : Number, T : Comparable<T> {
    return toCollection(java.util.TreeSet<T>())
}

/**
 * Returns a [SortedSet][java.util.SortedSet] of all elements.
 *
 * Elements in the set returned are sorted according to the given [comparator].
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toSortedSet(comparator: Comparator<in T>): java.util.SortedSet<T> {
    return toCollection(java.util.TreeSet<T>(comparator))
}

@PublishedApi
internal fun mapCapacity(size: Int): Int {
    return when {
        size < 3 -> size + 1
        size < 1 shl (Int.SIZE_BITS - 2) -> ((size / 0.75F) + 1.0F).toInt()
        else -> Int.MAX_VALUE
    }
}