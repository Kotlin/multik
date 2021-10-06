/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.ndarray.operations.mapCapacity
import kotlin.math.min
import kotlin.random.Random

public fun complexFloatArrayOf(vararg elements: ComplexFloat): ComplexFloatArray =
    if (elements.isEmpty()) ComplexFloatArray(0) else ComplexFloatArray(elements.size) { elements[it] }

public fun complexDoubleArrayOf(vararg elements: ComplexDouble): ComplexDoubleArray =
    if (elements.isEmpty()) ComplexDoubleArray(0) else ComplexDoubleArray(elements.size) { elements[it] }

/**
 * Returns 1st element from the array.
 *
 * If the size of this array is less than 1, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexFloatArray.component1(): ComplexFloat = get(0)

/**
 * Returns 1st element from the array.
 *
 * If the size of this array is less than 1, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexDoubleArray.component1(): ComplexDouble = get(0)

/**
 * Returns 2nd *element* from the array.
 *
 * If the size of this array is less than 2, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexFloatArray.component2(): ComplexFloat = get(1)

/**
 * Returns 2nd *element* from the array.
 *
 * If the size of this array is less than 2, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexDoubleArray.component2(): ComplexDouble = get(1)

/**
 * Returns 3rd *element* from the array.
 *
 * If the size of this array is less than 3, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexFloatArray.component3(): ComplexFloat = get(2)

/**
 * Returns 3rd *element* from the array.
 *
 * If the size of this array is less than 3, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexDoubleArray.component3(): ComplexDouble = get(2)

/**
 * Returns 4th *element* from the array.
 *
 * If the size of this array is less than 4, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexFloatArray.component4(): ComplexFloat = get(3)

/**
 * Returns 4th *element* from the array.
 *
 * If the size of this array is less than 4, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexDoubleArray.component4(): ComplexDouble = get(3)

/**
 * Returns 5th *element* from the array.
 *
 * If the size of this array is less than 5, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexFloatArray.component5(): ComplexFloat = get(4)

/**
 * Returns 5th *element* from the array.
 *
 * If the size of this array is less than 5, throws an [IndexOutOfBoundsException].
 */
@Suppress("NOTHING_TO_INLINE")
public inline operator fun ComplexDoubleArray.component5(): ComplexDouble = get(4)

/**
 * Returns `true` if [element] is found in the array.
 */
public operator fun ComplexFloatArray.contains(element: ComplexFloat): Boolean = indexOf(element) >= 0

/**
 * Returns `true` if [element] is found in the array.
 */
public operator fun ComplexDoubleArray.contains(element: ComplexDouble): Boolean = indexOf(element) >= 0

/**
 * Returns an element at the given [index] or throws an [IndexOutOfBoundsException] if the [index] is out of bounds of this array.
 */
public fun ComplexFloatArray.elementAt(index: Int): ComplexFloat = get(index)

/**
 * Returns an element at the given [index] or throws an [IndexOutOfBoundsException] if the [index] is out of bounds of this array.
 */
public fun ComplexDoubleArray.elementAt(index: Int): ComplexDouble = get(index)

/**
 * Returns an element at the given [index] or the result of calling the [defaultValue] function if the [index] is out of bounds of this array.
 */
public inline fun ComplexFloatArray.elementAtOrElse(index: Int, defaultValue: (Int) -> ComplexFloat): ComplexFloat =
    if (index in 0..lastIndex) get(index) else defaultValue(index)

/**
 * Returns an element at the given [index] or the result of calling the [defaultValue] function if the [index] is out of bounds of this array.
 */
public inline fun ComplexDoubleArray.elementAtOrElse(index: Int, defaultValue: (Int) -> ComplexDouble): ComplexDouble =
    if (index in 0..lastIndex) get(index) else defaultValue(index)

/**
 * Returns an element at the given [index] or `null` if the [index] is out of bounds of this array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.elementAtOrNull(index: Int): ComplexFloat? = this.getOrNull(index)

/**
 * Returns an element at the given [index] or `null` if the [index] is out of bounds of this array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.elementAtOrNull(index: Int): ComplexDouble? = this.getOrNull(index)

/**
 * Returns the first element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexFloatArray.find(predicate: (ComplexFloat) -> Boolean): ComplexFloat? = firstOrNull(predicate)

/**
 * Returns the first element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexDoubleArray.find(predicate: (ComplexDouble) -> Boolean): ComplexDouble? =
    firstOrNull(predicate)

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexFloatArray.findLast(predicate: (ComplexFloat) -> Boolean): ComplexFloat? =
    lastOrNull(predicate)

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexDoubleArray.findLast(predicate: (ComplexDouble) -> Boolean): ComplexDouble? =
    lastOrNull(predicate)

/**
 * Returns first element.
 * @throws [NoSuchElementException] if the array is empty.
 */
public fun ComplexFloatArray.first(): ComplexFloat =
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    else
        this[0]

/**
 * Returns first element.
 * @throws [NoSuchElementException] if the array is empty.
 */
public fun ComplexDoubleArray.first(): ComplexDouble =
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    else
        this[0]

/**
 * Returns the first element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun ComplexFloatArray.first(predicate: (ComplexFloat) -> Boolean): ComplexFloat {
    for (element in this) if (predicate(element)) return element
    throw NoSuchElementException("Array contains no element matching the predicate.")
}

/**
 * Returns the first element matching the given [predicate].
 * @throws [NoSuchElementException] if no such element is found.
 */
public inline fun ComplexDoubleArray.first(predicate: (ComplexDouble) -> Boolean): ComplexDouble {
    for (element in this) if (predicate(element)) return element
    throw NoSuchElementException("Array contains no element matching the predicate.")
}

/**
 * Returns the first element, or `null` if the array is empty.
 */
public fun ComplexFloatArray.firstOrNull(): ComplexFloat? = if (isEmpty()) null else this[0]

/**
 * Returns the first element, or `null` if the array is empty.
 */
public fun ComplexDoubleArray.firstOrNull(): ComplexDouble? = if (isEmpty()) null else this[0]

/**
 * Returns the first element matching the given [predicate], or `null` if element was not found.
 */
public inline fun ComplexFloatArray.firstOrNull(predicate: (ComplexFloat) -> Boolean): ComplexFloat? {
    for (element in this) if (predicate(element)) return element
    return null
}

/**
 * Returns the first element matching the given [predicate], or `null` if element was not found.
 */
public inline fun ComplexDoubleArray.firstOrNull(predicate: (ComplexDouble) -> Boolean): ComplexDouble? {
    for (element in this) if (predicate(element)) return element
    return null
}

/**
 * Returns an element at the given [index] or the result of calling the [defaultValue] function if the [index] is out of bounds of this array.
 */
public inline fun ComplexFloatArray.getOrElse(index: Int, defaultValue: (Int) -> ComplexFloat): ComplexFloat =
    if (index in 0..lastIndex) get(index) else defaultValue(index)

/**
 * Returns an element at the given [index] or the result of calling the [defaultValue] function if the [index] is out of bounds of this array.
 */
public inline fun ComplexDoubleArray.getOrElse(index: Int, defaultValue: (Int) -> ComplexDouble): ComplexDouble =
    if (index in 0..lastIndex) get(index) else defaultValue(index)

/**
 * Returns an element at the given [index] or `null` if the [index] is out of bounds of this array.
 */
public fun ComplexFloatArray.getOrNull(index: Int): ComplexFloat? =
    if (index in 0..lastIndex) get(index) else null

/**
 * Returns an element at the given [index] or `null` if the [index] is out of bounds of this array.
 */
public fun ComplexDoubleArray.getOrNull(index: Int): ComplexDouble? =
    if (index in 0..lastIndex) get(index) else null

/**
 * Returns first index of [element], or -1 if the array does not contain element.
 */
public fun ComplexFloatArray.indexOf(element: ComplexFloat): Int {
    for (index in indices) {
        if (element == this[index]) {
            return index
        }
    }
    return -1
}

/**
 * Returns first index of [element], or -1 if the array does not contain element.
 */
public fun ComplexDoubleArray.indexOf(element: ComplexDouble): Int {
    for (index in indices) {
        if (element == this[index]) {
            return index
        }
    }
    return -1
}

/**
 * Returns index of the first element matching the given [predicate], or -1 if the array does not contain such element.
 */
public inline fun ComplexFloatArray.indexOfFirst(predicate: (ComplexFloat) -> Boolean): Int {
    for (index in indices) {
        if (predicate(this[index])) {
            return index
        }
    }
    return -1
}

/**
 * Returns index of the first element matching the given [predicate], or -1 if the array does not contain such element.
 */
public inline fun ComplexDoubleArray.indexOfFirst(predicate: (ComplexDouble) -> Boolean): Int {
    for (index in indices) {
        if (predicate(this[index])) {
            return index
        }
    }
    return -1
}

/**
 * Returns index of the last element matching the given [predicate], or -1 if the array does not contain such element.
 */
public inline fun ComplexFloatArray.indexOfLast(predicate: (ComplexFloat) -> Boolean): Int {
    for (index in indices.reversed()) {
        if (predicate(this[index])) {
            return index
        }
    }
    return -1
}

/**
 * Returns index of the last element matching the given [predicate], or -1 if the array does not contain such element.
 */
public inline fun ComplexDoubleArray.indexOfLast(predicate: (ComplexDouble) -> Boolean): Int {
    for (index in indices.reversed()) {
        if (predicate(this[index])) {
            return index
        }
    }
    return -1
}

/**
 * Returns the last element.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public fun ComplexFloatArray.last(): ComplexFloat {
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    return this[lastIndex]
}

/**
 * Returns the last element.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public fun ComplexDoubleArray.last(): ComplexDouble {
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    return this[lastIndex]
}

/**
 * Returns the last element matching the given [predicate].
 *
 * @throws NoSuchElementException if no such element is found.
 */
public inline fun ComplexFloatArray.last(predicate: (ComplexFloat) -> Boolean): ComplexFloat {
    for (index in this.indices.reversed()) {
        val element = this[index]
        if (predicate(element)) return element
    }
    throw NoSuchElementException("Array contains no element matching the predicate.")
}

/**
 * Returns the last element matching the given [predicate].
 *
 * @throws NoSuchElementException if no such element is found.
 */
public inline fun ComplexDoubleArray.last(predicate: (ComplexDouble) -> Boolean): ComplexDouble {
    for (index in this.indices.reversed()) {
        val element = this[index]
        if (predicate(element)) return element
    }
    throw NoSuchElementException("Array contains no element matching the predicate.")
}


/**
 * Returns last index of [element], or -1 if the array does not contain element.
 */
public fun ComplexFloatArray.lastIndexOf(element: ComplexFloat): Int {
    for (index in indices.reversed()) {
        if (element == this[index]) {
            return index
        }
    }
    return -1
}

/**
 * Returns last index of [element], or -1 if the array does not contain element.
 */
public fun ComplexDoubleArray.lastIndexOf(element: ComplexDouble): Int {
    for (index in indices.reversed()) {
        if (element == this[index]) {
            return index
        }
    }
    return -1
}

/** Returns the last element, or `null` if the array is empty. */
public fun ComplexFloatArray.lastOrNull(): ComplexFloat? {
    return if (isEmpty()) null else this[size - 1]
}

/** Returns the last element, or `null` if the array is empty. */
public fun ComplexDoubleArray.lastOrNull(): ComplexDouble? {
    return if (isEmpty()) null else this[size - 1]
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexFloatArray.lastOrNull(predicate: (ComplexFloat) -> Boolean): ComplexFloat? {
    for (index in this.indices.reversed()) {
        val element = this[index]
        if (predicate(element)) return element
    }
    return null
}

/**
 * Returns the last element matching the given [predicate], or `null` if no such element was found.
 */
public inline fun ComplexDoubleArray.lastOrNull(predicate: (ComplexDouble) -> Boolean): ComplexDouble? {
    for (index in this.indices.reversed()) {
        val element = this[index]
        if (predicate(element)) return element
    }
    return null
}

/**
 * Returns a random element from this array.
 *
 * @throws NoSuchElementException if this array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.random(): ComplexFloat = random(Random)

/**
 * Returns a random element from this array.
 *
 * @throws NoSuchElementException if this array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.random(): ComplexDouble = random(Random)

/**
 * Returns a random element from this array using the specified source of randomness.
 *
 * @throws NoSuchElementException if this array is empty.
 */
public fun ComplexFloatArray.random(random: Random): ComplexFloat =
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    else
        get(random.nextInt(size))

/**
 * Returns a random element from this array using the specified source of randomness.
 *
 * @throws NoSuchElementException if this array is empty.
 */
public fun ComplexDoubleArray.random(random: Random): ComplexDouble =
    if (isEmpty())
        throw NoSuchElementException("Array is empty.")
    else
        get(random.nextInt(size))

/**
 * Returns a random element from this array, or `null` if this array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.randomOrNull(): ComplexFloat? = randomOrNull(Random)

/**
 * Returns a random element from this array, or `null` if this array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.randomOrNull(): ComplexDouble? = randomOrNull(Random)

/**
 * Returns a random element from this array using the specified source of randomness, or `null` if this array is empty.
 */
public fun ComplexFloatArray.randomOrNull(random: Random): ComplexFloat? =
    if (isEmpty()) null else get(random.nextInt(size))

/**
 * Returns a random element from this array using the specified source of randomness, or `null` if this array is empty.
 */
public fun ComplexDoubleArray.randomOrNull(random: Random): ComplexDouble? =
    if (isEmpty()) null else get(random.nextInt(size))

/**
 * Returns the single element, or throws an exception if the array is empty or has more than one element.
 */
public fun ComplexFloatArray.single(): ComplexFloat = when (size) {
    0 -> throw NoSuchElementException("Array is empty.")
    1 -> this[0]
    else -> throw IllegalArgumentException("Array has more than one element.")
}

/**
 * Returns the single element, or throws an exception if the array is empty or has more than one element.
 */
public fun ComplexDoubleArray.single(): ComplexDouble = when (size) {
    0 -> throw NoSuchElementException("Array is empty.")
    1 -> this[0]
    else -> throw IllegalArgumentException("Array has more than one element.")
}

/**
 * Returns the single element matching the given [predicate], or throws exception if there is no or more than one matching element.
 */
public inline fun ComplexFloatArray.single(predicate: (ComplexFloat) -> Boolean): ComplexFloat {
    var single: ComplexFloat? = null
    var found = false
    for (element in this) {
        if (predicate(element)) {
            if (found) throw IllegalArgumentException("Array contains more than one matching element.")
            single = element
            found = true
        }
    }
    if (!found) throw NoSuchElementException("Array contains no element matching the predicate.")
    return single as ComplexFloat
}

/**
 * Returns the single element matching the given [predicate], or throws exception if there is no or more than one matching element.
 */
public inline fun ComplexDoubleArray.single(predicate: (ComplexDouble) -> Boolean): ComplexDouble {
    var single: ComplexDouble? = null
    var found = false
    for (element in this) {
        if (predicate(element)) {
            if (found) throw IllegalArgumentException("Array contains more than one matching element.")
            single = element
            found = true
        }
    }
    if (!found) throw NoSuchElementException("Array contains no element matching the predicate.")
    return single as ComplexDouble
}

/**
 * Returns single element, or `null` if the array is empty or has more than one element.
 */
public fun ComplexFloatArray.singleOrNull(): ComplexFloat? = if (size == 1) this[0] else null

/**
 * Returns single element, or `null` if the array is empty or has more than one element.
 */
public fun ComplexDoubleArray.singleOrNull(): ComplexDouble? = if (size == 1) this[0] else null

/**
 * Returns the single element matching the given [predicate], or `null` if element was not found or more than one element was found.
 */
public inline fun ComplexFloatArray.singleOrNull(predicate: (ComplexFloat) -> Boolean): ComplexFloat? {
    var single: ComplexFloat? = null
    var found = false
    for (element in this) {
        if (predicate(element)) {
            if (found) return null
            single = element
            found = true
        }
    }
    if (!found) return null
    return single
}

/**
 * Returns the single element matching the given [predicate], or `null` if element was not found or more than one element was found.
 */
public inline fun ComplexDoubleArray.singleOrNull(predicate: (ComplexDouble) -> Boolean): ComplexDouble? {
    var single: ComplexDouble? = null
    var found = false
    for (element in this) {
        if (predicate(element)) {
            if (found) return null
            single = element
            found = true
        }
    }
    if (!found) return null
    return single
}

/**
 * Returns a list containing all elements except first [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexFloatArray.drop(n: Int): List<ComplexFloat> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    return takeLast((size - n).coerceAtLeast(0))
}

/**
 * Returns a list containing all elements except first [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexDoubleArray.drop(n: Int): List<ComplexDouble> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    return takeLast((size - n).coerceAtLeast(0))
}

/**
 * Returns a list containing all elements except last [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexFloatArray.dropLast(n: Int): List<ComplexFloat> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    return take((size - n).coerceAtLeast(0))
}

/**
 * Returns a list containing all elements except last [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexDoubleArray.dropLast(n: Int): List<ComplexDouble> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    return take((size - n).coerceAtLeast(0))
}

/**
 * Returns a list containing all elements except last elements that satisfy the given [predicate].
 */
public inline fun ComplexFloatArray.dropLastWhile(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> {
    for (index in lastIndex downTo 0) {
        if (!predicate(this[index])) {
            return take(index + 1)
        }
    }
    return emptyList()
}

/**
 * Returns a list containing all elements except last elements that satisfy the given [predicate].
 */
public inline fun ComplexDoubleArray.dropLastWhile(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> {
    for (index in lastIndex downTo 0) {
        if (!predicate(this[index])) {
            return take(index + 1)
        }
    }
    return emptyList()
}

/**
 * Returns a list containing all elements except first elements that satisfy the given [predicate].
 */
public inline fun ComplexFloatArray.dropWhile(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> {
    var yielding = false
    val list = ArrayList<ComplexFloat>()
    for (item in this)
        if (yielding)
            list.add(item)
        else if (!predicate(item)) {
            list.add(item)
            yielding = true
        }
    return list
}

/**
 * Returns a list containing all elements except first elements that satisfy the given [predicate].
 */
public inline fun ComplexDoubleArray.dropWhile(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> {
    var yielding = false
    val list = ArrayList<ComplexDouble>()
    for (item in this)
        if (yielding)
            list.add(item)
        else if (!predicate(item)) {
            list.add(item)
            yielding = true
        }
    return list
}

/**
 * Returns a list containing only elements matching the given [predicate].
 */
public inline fun ComplexFloatArray.filter(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> =
    filterTo(ArrayList(), predicate)

/**
 * Returns a list containing only elements matching the given [predicate].
 */
public inline fun ComplexDoubleArray.filter(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> =
    filterTo(ArrayList(), predicate)

/**
 * Returns a list containing only elements matching the given [predicate].
 * @param [predicate] function that takes the index of an element and the element itself
 * and returns the result of predicate evaluation on the element.
 */
public inline fun ComplexFloatArray.filterIndexed(predicate: (index: Int, ComplexFloat) -> Boolean): List<ComplexFloat> =
    filterIndexedTo(ArrayList(), predicate)

/**
 * Returns a list containing only elements matching the given [predicate].
 * @param [predicate] function that takes the index of an element and the element itself
 * and returns the result of predicate evaluation on the element.
 */
public inline fun ComplexDoubleArray.filterIndexed(predicate: (index: Int, ComplexDouble) -> Boolean): List<ComplexDouble> =
    filterIndexedTo(ArrayList(), predicate)

/**
 * Appends all elements matching the given [predicate] to the given [destination].
 * @param [predicate] function that takes the index of an element and the element itself
 * and returns the result of predicate evaluation on the element.
 */
public inline fun <C : MutableCollection<in ComplexFloat>> ComplexFloatArray.filterIndexedTo(
    destination: C,
    predicate: (index: Int, ComplexFloat) -> Boolean
): C {
    forEachIndexed { index, element ->
        if (predicate(index, element)) destination.add(element)
    }
    return destination
}

/**
 * Appends all elements matching the given [predicate] to the given [destination].
 * @param [predicate] function that takes the index of an element and the element itself
 * and returns the result of predicate evaluation on the element.
 */
public inline fun <C : MutableCollection<in ComplexDouble>> ComplexDoubleArray.filterIndexedTo(
    destination: C,
    predicate: (index: Int, ComplexDouble) -> Boolean
): C {
    forEachIndexed { index, element ->
        if (predicate(index, element)) destination.add(element)
    }
    return destination
}

/**
 * Returns a list containing all elements not matching the given [predicate].
 */
public inline fun ComplexFloatArray.filterNot(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> =
    filterNotTo(ArrayList(), predicate)

/**
 * Returns a list containing all elements not matching the given [predicate].
 */
public inline fun ComplexDoubleArray.filterNot(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> =
    filterNotTo(ArrayList(), predicate)

/**
 * Appends all elements not matching the given [predicate] to the given [destination].
 */
public inline fun <C : MutableCollection<in ComplexFloat>> ComplexFloatArray.filterNotTo(
    destination: C,
    predicate: (ComplexFloat) -> Boolean
): C {
    for (element in this) if (!predicate(element)) destination.add(element)
    return destination
}

/**
 * Appends all elements not matching the given [predicate] to the given [destination].
 */
public inline fun <C : MutableCollection<in ComplexDouble>> ComplexDoubleArray.filterNotTo(
    destination: C,
    predicate: (ComplexDouble) -> Boolean
): C {
    for (element in this) if (!predicate(element)) destination.add(element)
    return destination
}

/**
 * Appends all elements matching the given [predicate] to the given [destination].
 */
public inline fun <C : MutableCollection<in ComplexFloat>> ComplexFloatArray.filterTo(destination: C, predicate: (ComplexFloat) -> Boolean): C {
    for (element in this) if (predicate(element)) destination.add(element)
    return destination
}

/**
 * Appends all elements matching the given [predicate] to the given [destination].
 */
public inline fun <C : MutableCollection<in ComplexDouble>> ComplexDoubleArray.filterTo(destination: C, predicate: (ComplexDouble) -> Boolean): C {
    for (element in this) if (predicate(element)) destination.add(element)
    return destination
}

/**
 * Returns a list containing elements at indices in the specified [indices] range.
 */
public fun ComplexFloatArray.slice(indices: IntRange): List<ComplexFloat> {
    if (indices.isEmpty()) return listOf()
    return copyOfRange(indices.first, indices.last + 1).asList()
}

/**
 * Returns a list containing elements at indices in the specified [indices] range.
 */
public fun ComplexDoubleArray.slice(indices: IntRange): List<ComplexDouble> {
    if (indices.isEmpty()) return listOf()
    return copyOfRange(indices.first, indices.last + 1).asList()
}

/**
 * Returns a list containing elements at specified [indices].
 */
public fun ComplexFloatArray.slice(indices: Iterable<Int>): List<ComplexFloat> {
    val size = if (indices is Collection<*>) indices.size else 10
    if (size == 0) return emptyList()
    val list = ArrayList<ComplexFloat>(size)
    for (index in indices) {
        list.add(get(index))
    }
    return list
}

/**
 * Returns a list containing elements at specified [indices].
 */
public fun ComplexDoubleArray.slice(indices: Iterable<Int>): List<ComplexDouble> {
    val size = if (indices is Collection<*>) indices.size else 10
    if (size == 0) return emptyList()
    val list = ArrayList<ComplexDouble>(size)
    for (index in indices) {
        list.add(get(index))
    }
    return list
}

/**
 * Returns an array containing elements of this array at specified [indices].
 */
public fun ComplexFloatArray.sliceArray(indices: Collection<Int>): ComplexFloatArray {
    val result = ComplexFloatArray(indices.size)
    var targetIndex = 0
    for (sourceIndex in indices) {
        result[targetIndex++] = this[sourceIndex]
    }
    return result
}

/**
 * Returns an array containing elements of this array at specified [indices].
 */
public fun ComplexDoubleArray.sliceArray(indices: Collection<Int>): ComplexDoubleArray {
    val result = ComplexDoubleArray(indices.size)
    var targetIndex = 0
    for (sourceIndex in indices) {
        result[targetIndex++] = this[sourceIndex]
    }
    return result
}

/**
 * Returns an array containing elements at indices in the specified [indices] range.
 */
public fun ComplexFloatArray.sliceArray(indices: IntRange): ComplexFloatArray =
    if (indices.isEmpty())
        ComplexFloatArray(0)
    else
        copyOfRange(indices.first, indices.last + 1)

/**
 * Returns an array containing elements at indices in the specified [indices] range.
 */
public fun ComplexDoubleArray.sliceArray(indices: IntRange): ComplexDoubleArray =
    if (indices.isEmpty())
        ComplexDoubleArray(0)
    else
        copyOfRange(indices.first, indices.last + 1)

/**
 * Returns a list containing first [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexFloatArray.take(n: Int): List<ComplexFloat> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    if (n == 0) return emptyList()
    if (n >= size) return toList()
    if (n == 1) return listOf(this[0])
    var count = 0
    val list = ArrayList<ComplexFloat>(n)
    for (item in this) {
        list.add(item)
        if (++count == n)
            break
    }
    return list
}

/**
 * Returns a list containing first [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexDoubleArray.take(n: Int): List<ComplexDouble> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    if (n == 0) return emptyList()
    if (n >= size) return toList()
    if (n == 1) return listOf(this[0])
    var count = 0
    val list = ArrayList<ComplexDouble>(n)
    for (item in this) {
        list.add(item)
        if (++count == n)
            break
    }
    return list
}

/**
 * Returns a list containing last [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexFloatArray.takeLast(n: Int): List<ComplexFloat> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    if (n == 0) return emptyList()
    val size = size
    if (n >= size) return toList()
    if (n == 1) return listOf(this[size - 1])
    val list = ArrayList<ComplexFloat>(n)
    for (index in size - n until size)
        list.add(this[index])
    return list
}

/**
 * Returns a list containing last [n] elements.
 *
 * @throws IllegalArgumentException if [n] is negative.
 */
public fun ComplexDoubleArray.takeLast(n: Int): List<ComplexDouble> {
    require(n >= 0) { "Requested element count $n is less than zero." }
    if (n == 0) return emptyList()
    val size = size
    if (n >= size) return toList()
    if (n == 1) return listOf(this[size - 1])
    val list = ArrayList<ComplexDouble>(n)
    for (index in size - n until size)
        list.add(this[index])
    return list
}

/**
 * Returns a list containing last elements satisfying the given [predicate].
 */
public inline fun ComplexFloatArray.takeLastWhile(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> {
    for (index in lastIndex downTo 0) {
        if (!predicate(this[index])) {
            return drop(index + 1)
        }
    }
    return toList()
}

/**
 * Returns a list containing last elements satisfying the given [predicate].
 */
public inline fun ComplexDoubleArray.takeLastWhile(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> {
    for (index in lastIndex downTo 0) {
        if (!predicate(this[index])) {
            return drop(index + 1)
        }
    }
    return toList()
}

/**
 * Returns a list containing first elements satisfying the given [predicate].
 */
public inline fun ComplexFloatArray.takeWhile(predicate: (ComplexFloat) -> Boolean): List<ComplexFloat> {
    val list = ArrayList<ComplexFloat>()
    for (item in this) {
        if (!predicate(item)) break
        list.add(item)
    }
    return list
}

/**
 * Returns a list containing first elements satisfying the given [predicate].
 */
public inline fun ComplexDoubleArray.takeWhile(predicate: (ComplexDouble) -> Boolean): List<ComplexDouble> {
    val list = ArrayList<ComplexDouble>()
    for (item in this) {
        if (!predicate(item)) break
        list.add(item)
    }
    return list
}

/**
 * Reverses elements in the array in-place.
 */
public fun ComplexFloatArray.reverse(): Unit {
    val midPoint = (size / 2) - 1
    if (midPoint < 0) return
    var reverseIndex = lastIndex
    for (index in 0..midPoint) {
        val tmp = this[index]
        this[index] = this[reverseIndex]
        this[reverseIndex] = tmp
        reverseIndex--
    }
}

/**
 * Reverses elements in the array in-place.
 */
public fun ComplexDoubleArray.reverse(): Unit {
    val midPoint = (size / 2) - 1
    if (midPoint < 0) return
    var reverseIndex = lastIndex
    for (index in 0..midPoint) {
        val tmp = this[index]
        this[index] = this[reverseIndex]
        this[reverseIndex] = tmp
        reverseIndex--
    }
}

/**
 * Reverses elements of the array in the specified range in-place.
 *
 * @param fromIndex the start of the range (inclusive) to reverse.
 * @param toIndex the end of the range (exclusive) to reverse.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
public fun ComplexFloatArray.reverse(fromIndex: Int, toIndex: Int): Unit {
    checkRangeIndexes(fromIndex, toIndex, size)
    val midPoint = (fromIndex + toIndex) / 2
    if (fromIndex == midPoint) return
    var reverseIndex = toIndex - 1
    for (index in fromIndex until midPoint) {
        val tmp = this[index]
        this[index] = this[reverseIndex]
        this[reverseIndex] = tmp
        reverseIndex--
    }
}

/**
 * Reverses elements of the array in the specified range in-place.
 *
 * @param fromIndex the start of the range (inclusive) to reverse.
 * @param toIndex the end of the range (exclusive) to reverse.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
public fun ComplexDoubleArray.reverse(fromIndex: Int, toIndex: Int): Unit {
    checkRangeIndexes(fromIndex, toIndex, size)
    val midPoint = (fromIndex + toIndex) / 2
    if (fromIndex == midPoint) return
    var reverseIndex = toIndex - 1
    for (index in fromIndex until midPoint) {
        val tmp = this[index]
        this[index] = this[reverseIndex]
        this[reverseIndex] = tmp
        reverseIndex--
    }
}

/**
 * Returns a list with elements in reversed order.
 */
public fun ComplexFloatArray.reversed(): List<ComplexFloat> {
    if (isEmpty()) return emptyList()
    val list = toMutableList()
    list.reverse()
    return list
}

/**
 * Returns a list with elements in reversed order.
 */
public fun ComplexDoubleArray.reversed(): List<ComplexDouble> {
    if (isEmpty()) return emptyList()
    val list = toMutableList()
    list.reverse()
    return list
}

/**
 * Returns an array with elements of this array in reversed order.
 */
public fun ComplexFloatArray.reversedArray(): ComplexFloatArray {
    if (isEmpty()) return this
    val result = ComplexFloatArray(size)
    val lastIndex = lastIndex
    for (i in 0..lastIndex)
        result[lastIndex - i] = this[i]
    return result
}

/**
 * Returns an array with elements of this array in reversed order.
 */
public fun ComplexDoubleArray.reversedArray(): ComplexDoubleArray {
    if (isEmpty()) return this
    val result = ComplexDoubleArray(size)
    val lastIndex = lastIndex
    for (i in 0..lastIndex)
        result[lastIndex - i] = this[i]
    return result
}

/**
 * Randomly shuffles elements in this array in-place.
 */
public fun ComplexFloatArray.shuffle(): Unit {
    shuffle(Random)
}

/**
 * Randomly shuffles elements in this array in-place.
 */
public fun ComplexDoubleArray.shuffle(): Unit {
    shuffle(Random)
}

/**
 * Randomly shuffles elements in this array in-place using the specified [random] instance as the source of randomness.
 *
 * See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
 */
public fun ComplexFloatArray.shuffle(random: Random): Unit {
    for (i in lastIndex downTo 1) {
        val j = random.nextInt(i + 1)
        val copy = this[i]
        this[i] = this[j]
        this[j] = copy
    }
}

/**
 * Randomly shuffles elements in this array in-place using the specified [random] instance as the source of randomness.
 *
 * See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
 */
public fun ComplexDoubleArray.shuffle(random: Random): Unit {
    for (i in lastIndex downTo 1) {
        val j = random.nextInt(i + 1)
        val copy = this[i]
        this[i] = this[j]
        this[j] = copy
    }
}

/**
 * Returns a list of all elements sorted according to natural sort order of the value returned by specified [selector] function.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.sortedBy(crossinline selector: (ComplexFloat) -> R?): List<ComplexFloat> {
    return sortedWith(compareBy(selector))
}

/**
 * Returns a list of all elements sorted according to natural sort order of the value returned by specified [selector] function.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.sortedBy(crossinline selector: (ComplexDouble) -> R?): List<ComplexDouble> {
    return sortedWith(compareBy(selector))
}

/**
 * Returns a list of all elements sorted descending according to natural sort order of the value returned by specified [selector] function.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.sortedByDescending(crossinline selector: (ComplexFloat) -> R?): List<ComplexFloat> {
    return sortedWith(compareByDescending(selector))
}

/**
 * Returns a list of all elements sorted descending according to natural sort order of the value returned by specified [selector] function.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.sortedByDescending(crossinline selector: (ComplexDouble) -> R?): List<ComplexDouble> {
    return sortedWith(compareByDescending(selector))
}

/**
 * Returns a list of all elements sorted according to the specified [comparator].
 */
public fun ComplexFloatArray.sortedWith(comparator: Comparator<in ComplexFloat>): List<ComplexFloat> {
    return toTypedArray().apply { sortWith(comparator) }.asList()
}

/**
 * Returns a list of all elements sorted according to the specified [comparator].
 */
public fun ComplexDoubleArray.sortedWith(comparator: Comparator<in ComplexDouble>): List<ComplexDouble> {
    return toTypedArray().apply { sortWith(comparator) }.asList()
}

/**
 * Returns a [List] that wraps the original array.
 */
public fun ComplexFloatArray.asList(): List<ComplexFloat> = object : AbstractList<ComplexFloat>(), RandomAccess {
    override val size: Int get() = this@asList.size
    override fun isEmpty(): Boolean = this@asList.isEmpty()
    override fun contains(element: ComplexFloat): Boolean = this@asList.contains(element) // TODO: ?
    override fun get(index: Int): ComplexFloat = this@asList[index]
    override fun indexOf(element: ComplexFloat): Int = this@asList.indexOf(element) // TODO: ?
    override fun lastIndexOf(element: ComplexFloat): Int = this@asList.lastIndexOf(element) // TODO: ?
}

/**
 * Returns a [List] that wraps the original array.
 */
public fun ComplexDoubleArray.asList(): List<ComplexDouble> = object : AbstractList<ComplexDouble>(), RandomAccess {
    override val size: Int get() = this@asList.size
    override fun isEmpty(): Boolean = this@asList.isEmpty()
    override fun contains(element: ComplexDouble): Boolean = this@asList.contains(element) // TODO: ?
    override fun get(index: Int): ComplexDouble = this@asList[index]
    override fun indexOf(element: ComplexDouble): Int = this@asList.indexOf(element) // TODO: ?
    override fun lastIndexOf(element: ComplexDouble): Int = this@asList.lastIndexOf(element) // TODO: ?
}

/**
 * Returns `true` if the two specified arrays are *structurally* equal to one another,
 * i.e. contain the same number of the same elements in the same order.
 *
 * The elements are compared for equality with the [equals][Any.equals] function.
 */
public infix fun ComplexFloatArray?.contentEquals(other: ComplexFloatArray?): Boolean =
    java.util.Arrays.equals(this?.getFlatArray(), other?.getFlatArray())

/**
 * Returns `true` if the two specified arrays are *structurally* equal to one another,
 * i.e. contain the same number of the same elements in the same order.
 *
 * The elements are compared for equality with the [equals][Any.equals] function.
 */
public infix fun ComplexDoubleArray?.contentEquals(other: ComplexDoubleArray?): Boolean =
    java.util.Arrays.equals(this?.getFlatArray(), other?.getFlatArray())

/**
 * Returns a hash code based on the contents of this array as if it is [List].
 */
public fun ComplexFloatArray?.contentHashCode(): Int = java.util.Arrays.hashCode(this?.getFlatArray())

/**
 * Returns a hash code based on the contents of this array as if it is [List].
 */
public fun ComplexDoubleArray?.contentHashCode(): Int = java.util.Arrays.hashCode(this?.getFlatArray())

/**
 * Copies this array or its subrange into the [destination] array and returns that array.
 *
 * It's allowed to pass the same array in the [destination] and even specify the subrange so that it overlaps with the destination range.
 *
 * @param destination the array to copy to.
 * @param destinationOffset the position in the [destination] array to copy to, 0 by default.
 * @param startIndex the beginning (inclusive) of the subrange to copy, 0 by default.
 * @param endIndex the end (exclusive) of the subrange to copy, size of this array by default.
 *
 * @throws IndexOutOfBoundsException or [IllegalArgumentException] when [startIndex] or [endIndex] is out of range of this array indices or when `startIndex > endIndex`.
 * @throws IndexOutOfBoundsException when the subrange doesn't fit into the [destination] array starting at the specified [destinationOffset],
 * or when that index is out of the [destination] array indices range.
 *
 * @return the [destination] array.
 */
public fun ComplexFloatArray.copyInto(destination: ComplexFloatArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size): ComplexFloatArray {
    System.arraycopy(this.getFlatArray(), startIndex * 2, destination.getFlatArray(), destinationOffset * 2, (endIndex - startIndex) * 2)
    return destination
}

/**
 * Copies this array or its subrange into the [destination] array and returns that array.
 *
 * It's allowed to pass the same array in the [destination] and even specify the subrange so that it overlaps with the destination range.
 *
 * @param destination the array to copy to.
 * @param destinationOffset the position in the [destination] array to copy to, 0 by default.
 * @param startIndex the beginning (inclusive) of the subrange to copy, 0 by default.
 * @param endIndex the end (exclusive) of the subrange to copy, size of this array by default.
 *
 * @throws IndexOutOfBoundsException or [IllegalArgumentException] when [startIndex] or [endIndex] is out of range of this array indices or when `startIndex > endIndex`.
 * @throws IndexOutOfBoundsException when the subrange doesn't fit into the [destination] array starting at the specified [destinationOffset],
 * or when that index is out of the [destination] array indices range.
 *
 * @return the [destination] array.
 */
public fun ComplexDoubleArray.copyInto(destination: ComplexDoubleArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size): ComplexDoubleArray {
    System.arraycopy(this.getFlatArray(), startIndex * 2, destination.getFlatArray(), destinationOffset * 2, (endIndex - startIndex) * 2)
    return destination
}

/**
 * Returns new array which is a copy of the original array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.copyOf(): ComplexFloatArray {
    val ret = ComplexFloatArray(size)
    System.arraycopy(this.getFlatArray(), 0, ret.getFlatArray(), 0, ret.getFlatArray().size)
    return ret
}

/**
 * Returns new array which is a copy of the original array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.copyOf(): ComplexDoubleArray {
    val ret = ComplexDoubleArray(size)
    System.arraycopy(this.getFlatArray(), 0, ret.getFlatArray(), 0, ret.getFlatArray().size)
    return ret
}

/**
 * Returns new array which is a copy of the original array, resized to the given [newSize].
 * The copy is either truncated or padded at the end with zero values if necessary.
 *
 * - If [newSize] is less than the size of the original array, the copy array is truncated to the [newSize].
 * - If [newSize] is greater than the size of the original array, the extra elements in the copy array are filled with zero values.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.copyOf(newSize: Int): ComplexFloatArray {
    val ret = ComplexFloatArray(newSize)
    System.arraycopy(this.getFlatArray(), 0, ret.getFlatArray(), 0, min(ret.size, newSize) * 2)
    return ret
}

/**
 * Returns new array which is a copy of the original array, resized to the given [newSize].
 * The copy is either truncated or padded at the end with zero values if necessary.
 *
 * - If [newSize] is less than the size of the original array, the copy array is truncated to the [newSize].
 * - If [newSize] is greater than the size of the original array, the extra elements in the copy array are filled with zero values.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.copyOf(newSize: Int): ComplexDoubleArray {
    val ret = ComplexDoubleArray(newSize)
    System.arraycopy(this.getFlatArray(), 0, ret.getFlatArray(), 0, min(this.size, newSize) * 2)
    return ret
}

/**
 * Returns a new array which is a copy of the specified range of the original array.
 *
 * @param fromIndex the start of the range (inclusive) to copy.
 * @param toIndex the end of the range (exclusive) to copy.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.copyOfRange(fromIndex: Int, toIndex: Int): ComplexFloatArray {
    if (toIndex > size) throw IndexOutOfBoundsException("toIndex: $toIndex, size: $size")
    val newLength = toIndex - fromIndex
    require(newLength >= 0) { "$fromIndex > $toIndex" }
    val ret = ComplexFloatArray(newLength)
    System.arraycopy(this.getFlatArray(), fromIndex, ret.getFlatArray(), 0, min(this.size - fromIndex, newLength) * 2)
    return ret
}

/**
 * Returns a new array which is a copy of the specified range of the original array.
 *
 * @param fromIndex the start of the range (inclusive) to copy.
 * @param toIndex the end of the range (exclusive) to copy.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.copyOfRange(fromIndex: Int, toIndex: Int): ComplexDoubleArray {
    if (toIndex > size) throw IndexOutOfBoundsException("toIndex: $toIndex, size: $size")
    val newLength = toIndex - fromIndex
    require(newLength >= 0) { "$fromIndex > $toIndex" }
    val ret = ComplexDoubleArray(newLength)
    System.arraycopy(this.getFlatArray(), fromIndex, ret.getFlatArray(), 0, min(this.size - fromIndex, newLength) * 2)
    return ret
}

/**
 * Fills this array or its subrange with the specified [element] value.
 *
 * @param fromIndex the start of the range (inclusive) to fill, 0 by default.
 * @param toIndex the end of the range (exclusive) to fill, size of this array by default.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
public fun ComplexFloatArray.fill(element: ComplexFloat, fromIndex: Int = 0, toIndex: Int = size): Unit {
    checkRangeIndexes(fromIndex, toIndex, size)
    for (i in fromIndex until toIndex)
        this[i] = element
}

/**
 * Fills this array or its subrange with the specified [element] value.
 *
 * @param fromIndex the start of the range (inclusive) to fill, 0 by default.
 * @param toIndex the end of the range (exclusive) to fill, size of this array by default.
 *
 * @throws IndexOutOfBoundsException if [fromIndex] is less than zero or [toIndex] is greater than the size of this array.
 * @throws IllegalArgumentException if [fromIndex] is greater than [toIndex].
 */
public fun ComplexDoubleArray.fill(element: ComplexDouble, fromIndex: Int = 0, toIndex: Int = size): Unit {
    checkRangeIndexes(fromIndex, toIndex, size)
    for (i in fromIndex until toIndex)
        this[i] = element
}

/**
 * Returns the range of valid indices for the array.
 */
public val ComplexFloatArray.indices: IntRange
    get() = IntRange(0, lastIndex)

/**
 * Returns the range of valid indices for the array.
 */
public val ComplexDoubleArray.indices: IntRange
    get() = IntRange(0, lastIndex)

/**
 * Returns `true` if the array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.isEmpty(): Boolean = size == 0

/**
 * Returns `true` if the array is empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.isEmpty(): Boolean = size == 0

/**
 * Returns `true` if the array is not empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.isNotEmpty(): Boolean = !isEmpty()

/**
 * Returns `true` if the array is not empty.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.isNotEmpty(): Boolean = !isEmpty()

/**
 * Returns the last valid index for the array.
 */
public val ComplexFloatArray.lastIndex: Int
    get() = size - 1

/**
 * Returns the last valid index for the array.
 */
public val ComplexDoubleArray.lastIndex: Int
    get() = size - 1

/**
 * Returns an array containing all elements of the original array and then the given [element].
 */
public operator fun ComplexFloatArray.plus(element: ComplexFloat): ComplexFloatArray {
    val index = size
    val result = this.copyOf(index + 1)
    result[index] = element
    return result
}

/**
 * Returns an array containing all elements of the original array and then the given [element].
 */
public operator fun ComplexDoubleArray.plus(element: ComplexDouble): ComplexDoubleArray {
    val index = size
    val result = this.copyOf(index + 1)
    result[index] = element
    return result
}

/**
 * Returns an array containing all elements of the original array and then all elements of the given [elements] collection.
 */
public operator fun ComplexFloatArray.plus(elements: Collection<ComplexFloat>): ComplexFloatArray {
    var index = size
    val result = this.copyOf(index + elements.size)
    for (element in elements) result[index++] = element
    return result
}

/**
 * Returns an array containing all elements of the original array and then all elements of the given [elements] collection.
 */
public operator fun ComplexDoubleArray.plus(elements: Collection<ComplexDouble>): ComplexDoubleArray {
    var index = size
    val result = this.copyOf(index + elements.size)
    for (element in elements) result[index++] = element
    return result
}

/**
 * Returns an array containing all elements of the original array and then all elements of the given [elements] array.
 */
public operator fun ComplexFloatArray.plus(elements: ComplexFloatArray): ComplexFloatArray {
    val thisSize = size
    val arraySize = elements.size
    val result = this.copyOf(thisSize + arraySize)
    System.arraycopy(elements.getFlatArray(), 0, result.getFlatArray(), thisSize * 2, arraySize * 2)
    return result
}

/**
 * Returns an array containing all elements of the original array and then all elements of the given [elements] array.
 */
public operator fun ComplexDoubleArray.plus(elements: ComplexDoubleArray): ComplexDoubleArray {
    val thisSize = size
    val arraySize = elements.size
    val result = this.copyOf(thisSize + arraySize)
    System.arraycopy(elements.getFlatArray(), 0, result.getFlatArray(), thisSize * 2, arraySize * 2)
    return result
}

/**
 * Returns an array of ComplexFloat containing all of the elements of this generic array.
 */
public fun Array<out ComplexFloat>.toComplexFloatArray(): ComplexFloatArray =
    ComplexFloatArray(size) { index -> this[index] }

/**
 * Returns an array of ComplexDouble containing all of the elements of this generic array.
 */
public fun Array<out ComplexDouble>.toComplexDoubleArray(): ComplexDoubleArray =
    ComplexDoubleArray(size) { index -> this[index] }

/**
 * Returns an array of ComplexFloat containing all of the elements of this generic array.
 */
public fun FloatArray.toComplexFloatArray(): ComplexFloatArray =
    ComplexFloatArray(size).apply { this@toComplexFloatArray.copyInto(this.getFlatArray()) }

/**
 * Returns an array of ComplexDouble containing all of the elements of this generic array.
 */
public fun DoubleArray.toComplexDoubleArray(): ComplexDoubleArray =
    ComplexDoubleArray(size).apply { this@toComplexDoubleArray.copyInto(this.getFlatArray()) }

/**
 * Returns an array of ComplexFloat containing all of the elements of this collection.
 */
public fun Collection<ComplexFloat>.toComplexFloatArray(): ComplexFloatArray {
    val result = ComplexFloatArray(size)
    var index = 0
    for (element in this)
        result[index++] = element
    return result
}

/**
 * Returns an array of ComplexDouble containing all of the elements of this generic array.
 */
public fun Collection<ComplexDouble>.toComplexDoubleArray(): ComplexDoubleArray {
    val result = ComplexDoubleArray(size)
    var index = 0
    for (element in this)
        result[index++] = element
    return result
}

/**
 * Returns a *typed* object array containing all of the elements of this primitive array.
 */
@Suppress("UNCHECKED_CAST")
public fun ComplexFloatArray.toTypedArray(): Array<ComplexFloat> {
    val result = arrayOfNulls<ComplexFloat>(size)
    for (index in indices)
        result[index] = this[index]
    return result as Array<ComplexFloat>
}

/**
 * Returns a *typed* object array containing all of the elements of this primitive array.
 */
@Suppress("UNCHECKED_CAST")
public fun ComplexDoubleArray.toTypedArray(): Array<ComplexDouble> {
    val result = arrayOfNulls<ComplexDouble>(size)
    for (index in indices)
        result[index] = this[index]
    return result as Array<ComplexDouble>
}

/**
 * Returns a [Map] containing key-value pairs provided by [transform] function
 * applied to elements of the given array.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K, V> ComplexFloatArray.associate(transform: (ComplexFloat) -> Pair<K, V>): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateTo(LinkedHashMap(capacity), transform)
}

/**
 * Returns a [Map] containing key-value pairs provided by [transform] function
 * applied to elements of the given array.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K, V> ComplexDoubleArray.associate(transform: (ComplexDouble) -> Pair<K, V>): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateTo(LinkedHashMap(capacity), transform)
}

/**
 * Returns a [Map] containing the elements from the given array indexed by the key
 * returned from [keySelector] function applied to each element.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K> ComplexFloatArray.associateBy(keySelector: (ComplexFloat) -> K): Map<K, ComplexFloat> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector)
}

/**
 * Returns a [Map] containing the elements from the given array indexed by the key
 * returned from [keySelector] function applied to each element.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K> ComplexDoubleArray.associateBy(keySelector: (ComplexDouble) -> K): Map<K, ComplexDouble> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector)
}

/**
 * Returns a [Map] containing the values provided by [valueTransform] and indexed by [keySelector] functions applied to elements of the given array.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K, V> ComplexFloatArray.associateBy(keySelector: (ComplexFloat) -> K, valueTransform: (ComplexFloat) -> V): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector, valueTransform)
}

/**
 * Returns a [Map] containing the values provided by [valueTransform] and indexed by [keySelector] functions applied to elements of the given array.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <K, V> ComplexDoubleArray.associateBy(keySelector: (ComplexDouble) -> K, valueTransform: (ComplexDouble) -> V): Map<K, V> {
    val capacity = mapCapacity(size).coerceAtLeast(16)
    return associateByTo(LinkedHashMap(capacity), keySelector, valueTransform)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function applied to each element of the given array
 * and value is the element itself.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <K, M : MutableMap<in K, in ComplexFloat>> ComplexFloatArray.associateByTo(destination: M, keySelector: (ComplexFloat) -> K): M {
    for (element in this) {
        destination.put(keySelector(element), element)
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function applied to each element of the given array
 * and value is the element itself.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <K, M : MutableMap<in K, in ComplexDouble>> ComplexDoubleArray.associateByTo(destination: M, keySelector: (ComplexDouble) -> K): M {
    for (element in this) {
        destination.put(keySelector(element), element)
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function and
 * and value is provided by the [valueTransform] function applied to elements of the given array.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <K, V, M : MutableMap<in K, in V>> ComplexFloatArray.associateByTo(destination: M, keySelector: (ComplexFloat) -> K, valueTransform: (ComplexFloat) -> V): M {
    for (element in this) {
        destination.put(keySelector(element), valueTransform(element))
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs,
 * where key is provided by the [keySelector] function and
 * and value is provided by the [valueTransform] function applied to elements of the given array.
 *
 * If any two elements would have the same key returned by [keySelector] the last one gets added to the map.
 */
public inline fun <K, V, M : MutableMap<in K, in V>> ComplexDoubleArray.associateByTo(destination: M, keySelector: (ComplexDouble) -> K, valueTransform: (ComplexDouble) -> V): M {
    for (element in this) {
        destination.put(keySelector(element), valueTransform(element))
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs
 * provided by [transform] function applied to each element of the given array.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 */
public inline fun <K, V, M : MutableMap<in K, in V>> ComplexFloatArray.associateTo(destination: M, transform: (ComplexFloat) -> Pair<K, V>): M {
    for (element in this) {
        destination += transform(element)
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs
 * provided by [transform] function applied to each element of the given array.
 *
 * If any of two pairs would have the same key the last one gets added to the map.
 */
public inline fun <K, V, M : MutableMap<in K, in V>> ComplexDoubleArray.associateTo(destination: M, transform: (ComplexDouble) -> Pair<K, V>): M {
    for (element in this) {
        destination += transform(element)
    }
    return destination
}

/**
 * Returns a [Map] where keys are elements from the given array and values are
 * produced by the [valueSelector] function applied to each element.
 *
 * If any two elements are equal, the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <V> ComplexFloatArray.associateWith(valueSelector: (ComplexFloat) -> V): Map<ComplexFloat, V> {
    val result = LinkedHashMap<ComplexFloat, V>(mapCapacity(size).coerceAtLeast(16))
    return associateWithTo(result, valueSelector)
}

/**
 * Returns a [Map] where keys are elements from the given array and values are
 * produced by the [valueSelector] function applied to each element.
 *
 * If any two elements are equal, the last one gets added to the map.
 *
 * The returned map preserves the entry iteration order of the original array.
 */
public inline fun <V> ComplexDoubleArray.associateWith(valueSelector: (ComplexDouble) -> V): Map<ComplexDouble, V> {
    val result = LinkedHashMap<ComplexDouble, V>(mapCapacity(size).coerceAtLeast(16))
    return associateWithTo(result, valueSelector)
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs for each element of the given array,
 * where key is the element itself and value is provided by the [valueSelector] function applied to that key.
 *
 * If any two elements are equal, the last one overwrites the former value in the map.
 */
public inline fun <V, M : MutableMap<in ComplexFloat, in V>> ComplexFloatArray.associateWithTo(destination: M, valueSelector: (ComplexFloat) -> V): M {
    for (element in this) {
        destination.put(element, valueSelector(element))
    }
    return destination
}

/**
 * Populates and returns the [destination] mutable map with key-value pairs for each element of the given array,
 * where key is the element itself and value is provided by the [valueSelector] function applied to that key.
 *
 * If any two elements are equal, the last one overwrites the former value in the map.
 */
public inline fun <V, M : MutableMap<in ComplexDouble, in V>> ComplexDoubleArray.associateWithTo(destination: M, valueSelector: (ComplexDouble) -> V): M {
    for (element in this) {
        destination.put(element, valueSelector(element))
    }
    return destination
}

/**
 * Appends all elements to the given [destination] collection.
 */
public fun <C : MutableCollection<in ComplexFloat>> ComplexFloatArray.toCollection(destination: C): C {
    for (item in this) {
        destination.add(item)
    }
    return destination
}

/**
 * Appends all elements to the given [destination] collection.
 */
public fun <C : MutableCollection<in ComplexDouble>> ComplexDoubleArray.toCollection(destination: C): C {
    for (item in this) {
        destination.add(item)
    }
    return destination
}

/**
 * Returns a new [HashSet] of all elements.
 */
public fun ComplexFloatArray.toHashSet(): HashSet<ComplexFloat> {
    return toCollection(HashSet(mapCapacity(size)))
}

/**
 * Returns a new [HashSet] of all elements.
 */
public fun ComplexDoubleArray.toHashSet(): HashSet<ComplexDouble> {
    return toCollection(HashSet(mapCapacity(size)))
}

/**
 * Returns a [List] containing all elements.
 */
public fun ComplexFloatArray.toList(): List<ComplexFloat> {
    return when (size) {
        0 -> emptyList()
        1 -> listOf(this[0])
        else -> this.toMutableList()
    }
}

/**
 * Returns a [List] containing all elements.
 */
public fun ComplexDoubleArray.toList(): List<ComplexDouble> {
    return when (size) {
        0 -> emptyList()
        1 -> listOf(this[0])
        else -> this.toMutableList()
    }
}

/**
 * Returns a new [MutableList] filled with all elements of this array.
 */
public fun ComplexFloatArray.toMutableList(): MutableList<ComplexFloat> {
    val list = ArrayList<ComplexFloat>(size)
    for (item in this) list.add(item)
    return list
}

/**
 * Returns a new [MutableList] filled with all elements of this array.
 */
public fun ComplexDoubleArray.toMutableList(): MutableList<ComplexDouble> {
    val list = ArrayList<ComplexDouble>(size)
    for (item in this) list.add(item)
    return list
}

/**
 * Returns a [Set] of all elements.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public fun ComplexFloatArray.toSet(): Set<ComplexFloat> {
    return when (size) {
        0 -> emptySet()
        1 -> setOf(this[0])
        else -> toCollection(LinkedHashSet(mapCapacity(size)))
    }
}

/**
 * Returns a [Set] of all elements.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public fun ComplexDoubleArray.toSet(): Set<ComplexDouble> {
    return when (size) {
        0 -> emptySet()
        1 -> setOf(this[0])
        else -> toCollection(LinkedHashSet(mapCapacity(size)))
    }
}

/**
 * Returns a single list of all elements yielded from results of [transform] function being invoked on each element of original array.
 */
public inline fun <R> ComplexFloatArray.flatMap(transform: (ComplexFloat) -> Iterable<R>): List<R> {
    return flatMapTo(ArrayList(), transform)
}

/**
 * Returns a single list of all elements yielded from results of [transform] function being invoked on each element of original array.
 */
public inline fun <R> ComplexDoubleArray.flatMap(transform: (ComplexDouble) -> Iterable<R>): List<R> {
    return flatMapTo(ArrayList(), transform)
}

/**
 * Returns a single list of all elements yielded from results of [transform] function being invoked on each element
 * and its index in the original array.
 */
public inline fun <R> ComplexFloatArray.flatMapIndexed(transform: (index: Int, ComplexFloat) -> Iterable<R>): List<R> =
    flatMapIndexedTo(ArrayList(), transform)

/**
 * Returns a single list of all elements yielded from results of [transform] function being invoked on each element
 * and its index in the original array.
 */
public inline fun <R> ComplexDoubleArray.flatMapIndexed(transform: (index: Int, ComplexDouble) -> Iterable<R>): List<R> =
    flatMapIndexedTo(ArrayList(), transform)

/**
 * Appends all elements yielded from results of [transform] function being invoked on each element
 * and its index in the original array, to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexFloatArray.flatMapIndexedTo(destination: C, transform: (index: Int, ComplexFloat) -> Iterable<R>): C {
    var index = 0
    for (element in this) {
        val list = transform(index++, element)
        destination.addAll(list)
    }
    return destination
}

/**
 * Appends all elements yielded from results of [transform] function being invoked on each element
 * and its index in the original array, to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexDoubleArray.flatMapIndexedTo(destination: C, transform: (index: Int, ComplexDouble) -> Iterable<R>): C {
    var index = 0
    for (element in this) {
        val list = transform(index++, element)
        destination.addAll(list)
    }
    return destination
}

/**
 * Appends all elements yielded from results of [transform] function being invoked on each element of original array, to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexFloatArray.flatMapTo(destination: C, transform: (ComplexFloat) -> Iterable<R>): C {
    for (element in this) {
        val list = transform(element)
        destination.addAll(list)
    }
    return destination
}

/**
 * Appends all elements yielded from results of [transform] function being invoked on each element of original array, to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexDoubleArray.flatMapTo(destination: C, transform: (ComplexDouble) -> Iterable<R>): C {
    for (element in this) {
        val list = transform(element)
        destination.addAll(list)
    }
    return destination
}

/**
 * Groups elements of the original array by the key returned by the given [keySelector] function
 * applied to each element and returns a map where each group key is associated with a list of corresponding elements.
 *
 * The returned map preserves the entry iteration order of the keys produced from the original array.
 */
public inline fun <K> ComplexFloatArray.groupBy(keySelector: (ComplexFloat) -> K): Map<K, List<ComplexFloat>> {
    return groupByTo(LinkedHashMap(), keySelector)
}

/**
 * Groups elements of the original array by the key returned by the given [keySelector] function
 * applied to each element and returns a map where each group key is associated with a list of corresponding elements.
 *
 * The returned map preserves the entry iteration order of the keys produced from the original array.
 */
public inline fun <K> ComplexDoubleArray.groupBy(keySelector: (ComplexDouble) -> K): Map<K, List<ComplexDouble>> {
    return groupByTo(LinkedHashMap(), keySelector)
}

/**
 * Groups values returned by the [valueTransform] function applied to each element of the original array
 * by the key returned by the given [keySelector] function applied to the element
 * and returns a map where each group key is associated with a list of corresponding values.
 *
 * The returned map preserves the entry iteration order of the keys produced from the original array.
 */
public inline fun <K, V> ComplexFloatArray.groupBy(keySelector: (ComplexFloat) -> K, valueTransform: (ComplexFloat) -> V): Map<K, List<V>> {
    return groupByTo(LinkedHashMap(), keySelector, valueTransform)
}

/**
 * Groups values returned by the [valueTransform] function applied to each element of the original array
 * by the key returned by the given [keySelector] function applied to the element
 * and returns a map where each group key is associated with a list of corresponding values.
 *
 * The returned map preserves the entry iteration order of the keys produced from the original array.
 */
public inline fun <K, V> ComplexDoubleArray.groupBy(keySelector: (ComplexDouble) -> K, valueTransform: (ComplexDouble) -> V): Map<K, List<V>> {
    return groupByTo(LinkedHashMap(), keySelector, valueTransform)
}

/**
 * Groups elements of the original array by the key returned by the given [keySelector] function
 * applied to each element and puts to the [destination] map each group key associated with a list of corresponding elements.
 *
 * @return The [destination] map.
 */
public inline fun <K, M : MutableMap<in K, MutableList<ComplexFloat>>> ComplexFloatArray.groupByTo(destination: M, keySelector: (ComplexFloat) -> K): M {
    for (element in this) {
        val key = keySelector(element)
        val list = destination.getOrPut(key) { ArrayList() }
        list.add(element)
    }
    return destination
}

/**
 * Groups elements of the original array by the key returned by the given [keySelector] function
 * applied to each element and puts to the [destination] map each group key associated with a list of corresponding elements.
 *
 * @return The [destination] map.
 */
public inline fun <K, M : MutableMap<in K, MutableList<ComplexDouble>>> ComplexDoubleArray.groupByTo(destination: M, keySelector: (ComplexDouble) -> K): M {
    for (element in this) {
        val key = keySelector(element)
        val list = destination.getOrPut(key) { ArrayList() }
        list.add(element)
    }
    return destination
}

/**
 * Groups values returned by the [valueTransform] function applied to each element of the original array
 * by the key returned by the given [keySelector] function applied to the element
 * and puts to the [destination] map each group key associated with a list of corresponding values.
 *
 * @return The [destination] map.
 */
public inline fun <K, V, M : MutableMap<in K, MutableList<V>>> ComplexFloatArray.groupByTo(destination: M, keySelector: (ComplexFloat) -> K, valueTransform: (ComplexFloat) -> V): M {
    for (element in this) {
        val key = keySelector(element)
        val list = destination.getOrPut(key) { ArrayList() }
        list.add(valueTransform(element))
    }
    return destination
}

/**
 * Groups values returned by the [valueTransform] function applied to each element of the original array
 * by the key returned by the given [keySelector] function applied to the element
 * and puts to the [destination] map each group key associated with a list of corresponding values.
 *
 * @return The [destination] map.
 */
public inline fun <K, V, M : MutableMap<in K, MutableList<V>>> ComplexDoubleArray.groupByTo(destination: M, keySelector: (ComplexDouble) -> K, valueTransform: (ComplexDouble) -> V): M {
    for (element in this) {
        val key = keySelector(element)
        val list = destination.getOrPut(key) { ArrayList() }
        list.add(valueTransform(element))
    }
    return destination
}

/**
 * Returns a list containing the results of applying the given [transform] function
 * to each element in the original array.
 */
public inline fun <R> ComplexFloatArray.map(transform: (ComplexFloat) -> R): List<R> {
    return mapTo(ArrayList(size), transform)
}

/**
 * Returns a list containing the results of applying the given [transform] function
 * to each element in the original array.
 */
public inline fun <R> ComplexDoubleArray.map(transform: (ComplexDouble) -> R): List<R> {
    return mapTo(ArrayList(size), transform)
}

/**
 * Returns a list containing the results of applying the given [transform] function
 * to each element and its index in the original array.
 * @param [transform] function that takes the index of an element and the element itself
 * and returns the result of the transform applied to the element.
 */
public inline fun <R> ComplexFloatArray.mapIndexed(transform: (index: Int, ComplexFloat) -> R): List<R> {
    return mapIndexedTo(ArrayList(size), transform)
}

/**
 * Returns a list containing the results of applying the given [transform] function
 * to each element and its index in the original array.
 * @param [transform] function that takes the index of an element and the element itself
 * and returns the result of the transform applied to the element.
 */
public inline fun <R> ComplexDoubleArray.mapIndexed(transform: (index: Int, ComplexDouble) -> R): List<R> {
    return mapIndexedTo(ArrayList(size), transform)
}

/**
 * Applies the given [transform] function to each element and its index in the original array
 * and appends the results to the given [destination].
 * @param [transform] function that takes the index of an element and the element itself
 * and returns the result of the transform applied to the element.
 */
public inline fun <R, C : MutableCollection<in R>> ComplexFloatArray.mapIndexedTo(destination: C, transform: (index: Int, ComplexFloat) -> R): C {
    var index = 0
    for (item in this)
        destination.add(transform(index++, item))
    return destination
}

/**
 * Applies the given [transform] function to each element and its index in the original array
 * and appends the results to the given [destination].
 * @param [transform] function that takes the index of an element and the element itself
 * and returns the result of the transform applied to the element.
 */
public inline fun <R, C : MutableCollection<in R>> ComplexDoubleArray.mapIndexedTo(destination: C, transform: (index: Int, ComplexDouble) -> R): C {
    var index = 0
    for (item in this)
        destination.add(transform(index++, item))
    return destination
}

/**
 * Applies the given [transform] function to each element of the original array
 * and appends the results to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexFloatArray.mapTo(destination: C, transform: (ComplexFloat) -> R): C {
    for (item in this)
        destination.add(transform(item))
    return destination
}

/**
 * Applies the given [transform] function to each element of the original array
 * and appends the results to the given [destination].
 */
public inline fun <R, C : MutableCollection<in R>> ComplexDoubleArray.mapTo(destination: C, transform: (ComplexDouble) -> R): C {
    for (item in this)
        destination.add(transform(item))
    return destination
}

/**
 * Returns a lazy [Iterable] that wraps each element of the original array
 * into an [IndexedValue] containing the index of that element and the element itself.
 */
public fun ComplexFloatArray.withIndex(): Iterable<IndexedValue<ComplexFloat>> =
    object : Iterable<IndexedValue<ComplexFloat>> {
        override fun iterator(): Iterator<IndexedValue<ComplexFloat>> = object : Iterator<IndexedValue<ComplexFloat>> {
            private var index = 0
            private val iterator = this@withIndex.iterator()
            override fun hasNext(): Boolean = iterator.hasNext()
            override fun next(): IndexedValue<ComplexFloat> =
                IndexedValue(
                    if (index++ < 0) throw ArithmeticException("Index overflow has happened.") else index,
                    iterator.next()
                )
        }
    }

/**
 * Returns a lazy [Iterable] that wraps each element of the original array
 * into an [IndexedValue] containing the index of that element and the element itself.
 */
public fun ComplexDoubleArray.withIndex(): Iterable<IndexedValue<ComplexDouble>> =
    object : Iterable<IndexedValue<ComplexDouble>> {
        override fun iterator(): Iterator<IndexedValue<ComplexDouble>> =
            object : Iterator<IndexedValue<ComplexDouble>> {
                private var index = 0
                private val iterator = this@withIndex.iterator()
                override fun hasNext(): Boolean = iterator.hasNext()
                override fun next(): IndexedValue<ComplexDouble> =
                    IndexedValue(
                        if (index++ < 0) throw ArithmeticException("Index overflow has happened.") else index,
                        iterator.next()
                    )
            }
    }

/**
 * Returns a list containing only distinct elements from the given array.
 *
 * The elements in the resulting list are in the same order as they were in the source array.
 */
public fun ComplexFloatArray.distinct(): List<ComplexFloat> {
    return this.toMutableSet().toList()
}

/**
 * Returns a list containing only distinct elements from the given array.
 *
 * The elements in the resulting list are in the same order as they were in the source array.
 */
public fun ComplexDoubleArray.distinct(): List<ComplexDouble> {
    return this.toMutableSet().toList()
}

/**
 * Returns a list containing only elements from the given array
 * having distinct keys returned by the given [selector] function.
 *
 * The elements in the resulting list are in the same order as they were in the source array.
 */
public inline fun <K> ComplexFloatArray.distinctBy(selector: (ComplexFloat) -> K): List<ComplexFloat> {
    val set = HashSet<K>()
    val list = ArrayList<ComplexFloat>()
    for (e in this) {
        val key = selector(e)
        if (set.add(key))
            list.add(e)
    }
    return list
}

/**
 * Returns a list containing only elements from the given array
 * having distinct keys returned by the given [selector] function.
 *
 * The elements in the resulting list are in the same order as they were in the source array.
 */
public inline fun <K> ComplexDoubleArray.distinctBy(selector: (ComplexDouble) -> K): List<ComplexDouble> {
    val set = HashSet<K>()
    val list = ArrayList<ComplexDouble>()
    for (e in this) {
        val key = selector(e)
        if (set.add(key))
            list.add(e)
    }
    return list
}

/**
 * Returns a set containing all elements that are contained by both this array and the specified collection.
 *
 * The returned set preserves the element iteration order of the original array.
 *
 * To get a set containing all elements that are contained at least in one of these collections use [union].
 */
public infix fun ComplexFloatArray.intersect(other: Iterable<ComplexFloat>): Set<ComplexFloat> {
    val set = this.toMutableSet()
    set.retainAll(other)
    return set
}

/**
 * Returns a set containing all elements that are contained by both this array and the specified collection.
 *
 * The returned set preserves the element iteration order of the original array.
 *
 * To get a set containing all elements that are contained at least in one of these collections use [union].
 */
public infix fun ComplexDoubleArray.intersect(other: Iterable<ComplexDouble>): Set<ComplexDouble> {
    val set = this.toMutableSet()
    set.retainAll(other)
    return set
}

/**
 * Returns a set containing all elements that are contained by this array and not contained by the specified collection.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public infix fun ComplexFloatArray.subtract(other: Iterable<ComplexFloat>): Set<ComplexFloat> {
    val set = this.toMutableSet()
    set.removeAll(other)
    return set
}

/**
 * Returns a set containing all elements that are contained by this array and not contained by the specified collection.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public infix fun ComplexDoubleArray.subtract(other: Iterable<ComplexDouble>): Set<ComplexDouble> {
    val set = this.toMutableSet()
    set.removeAll(other)
    return set
}

/**
 * Returns a new [MutableSet] containing all distinct elements from the given array.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public fun ComplexFloatArray.toMutableSet(): MutableSet<ComplexFloat> {
    return toCollection(LinkedHashSet(mapCapacity(size)))
}

/**
 * Returns a new [MutableSet] containing all distinct elements from the given array.
 *
 * The returned set preserves the element iteration order of the original array.
 */
public fun ComplexDoubleArray.toMutableSet(): MutableSet<ComplexDouble> {
    return toCollection(LinkedHashSet(mapCapacity(size)))
}

/**
 * Returns a set containing all distinct elements from both collections.
 *
 * The returned set preserves the element iteration order of the original array.
 * Those elements of the [other] collection that are unique are iterated in the end
 * in the order of the [other] collection.
 *
 * To get a set containing all elements that are contained in both collections use [intersect].
 */
public infix fun ComplexFloatArray.union(other: Iterable<ComplexFloat>): Set<ComplexFloat> {
    val set = this.toMutableSet()
    set.addAll(other)
    return set
}

/**
 * Returns a set containing all distinct elements from both collections.
 *
 * The returned set preserves the element iteration order of the original array.
 * Those elements of the [other] collection that are unique are iterated in the end
 * in the order of the [other] collection.
 *
 * To get a set containing all elements that are contained in both collections use [intersect].
 */
public infix fun ComplexDoubleArray.union(other: Iterable<ComplexDouble>): Set<ComplexDouble> {
    val set = this.toMutableSet()
    set.addAll(other)
    return set
}

/**
 * Returns `true` if all elements match the given [predicate].
 */
public inline fun ComplexFloatArray.all(predicate: (ComplexFloat) -> Boolean): Boolean {
    for (element in this) if (!predicate(element)) return false
    return true
}

/**
 * Returns `true` if all elements match the given [predicate].
 */
public inline fun ComplexDoubleArray.all(predicate: (ComplexDouble) -> Boolean): Boolean {
    for (element in this) if (!predicate(element)) return false
    return true
}

/**
 * Returns `true` if array has at least one element.
 */
public fun ComplexFloatArray.any(): Boolean = !isEmpty()

/**
 * Returns `true` if array has at least one element.
 */
public fun ComplexDoubleArray.any(): Boolean = !isEmpty()

/**
 * Returns `true` if at least one element matches the given [predicate].
 */
public inline fun ComplexFloatArray.any(predicate: (ComplexFloat) -> Boolean): Boolean {
    for (element in this) if (predicate(element)) return true
    return false
}

/**
 * Returns `true` if at least one element matches the given [predicate].
 */
public inline fun ComplexDoubleArray.any(predicate: (ComplexDouble) -> Boolean): Boolean {
    for (element in this) if (predicate(element)) return true
    return false
}

/**
 * Returns the number of elements in this array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexFloatArray.count(): Int = size

/**
 * Returns the number of elements in this array.
 */
@Suppress("NOTHING_TO_INLINE")
public inline fun ComplexDoubleArray.count(): Int = size

/**
 * Returns the number of elements matching the given [predicate].
 */
public inline fun ComplexFloatArray.count(predicate: (ComplexFloat) -> Boolean): Int {
    var count = 0
    for (element in this) if (predicate(element)) ++count
    return count
}

/**
 * Returns the number of elements matching the given [predicate].
 */
public inline fun ComplexDoubleArray.count(predicate: (ComplexDouble) -> Boolean): Int {
    var count = 0
    for (element in this) if (predicate(element)) ++count
    return count
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.fold(initial: R, operation: (acc: R, ComplexFloat) -> R): R {
    var accumulator = initial
    for (element in this) accumulator = operation(accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.fold(initial: R, operation: (acc: R, ComplexDouble) -> R): R {
    var accumulator = initial
    for (element in this) accumulator = operation(accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.foldIndexed(initial: R, operation: (index: Int, acc: R, ComplexFloat) -> R): R {
    var index = 0
    var accumulator = initial
    for (element in this) accumulator = operation(index++, accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.foldIndexed(initial: R, operation: (index: Int, acc: R, ComplexDouble) -> R): R {
    var index = 0
    var accumulator = initial
    for (element in this) accumulator = operation(index++, accumulator, element)
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes an element and current accumulator value, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.foldRight(initial: R, operation: (ComplexFloat, acc: R) -> R): R {
    var index = lastIndex
    var accumulator = initial
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes an element and current accumulator value, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.foldRight(initial: R, operation: (ComplexDouble, acc: R) -> R): R {
    var index = lastIndex
    var accumulator = initial
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself
 * and current accumulator value, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.foldRightIndexed(initial: R, operation: (index: Int, ComplexFloat, acc: R) -> R): R {
    var index = lastIndex
    var accumulator = initial
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Accumulates value starting with [initial] value and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Returns the specified [initial] value if the array is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself
 * and current accumulator value, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.foldRightIndexed(initial: R, operation: (index: Int, ComplexDouble, acc: R) -> R): R {
    var index = lastIndex
    var accumulator = initial
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Performs the given [action] on each element.
 */
public inline fun ComplexFloatArray.forEach(action: (ComplexFloat) -> Unit): Unit {
    for (element in this) action(element)
}

/**
 * Performs the given [action] on each element.
 */
public inline fun ComplexDoubleArray.forEach(action: (ComplexDouble) -> Unit): Unit {
    for (element in this) action(element)
}

/**
 * Performs the given [action] on each element, providing sequential index with the element.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the action on the element.
 */
public inline fun ComplexFloatArray.forEachIndexed(action: (index: Int, ComplexFloat) -> Unit): Unit {
    var index = 0
    for (item in this) action(index++, item)
}

/**
 * Performs the given [action] on each element, providing sequential index with the element.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the action on the element.
 */
public inline fun ComplexDoubleArray.forEachIndexed(action: (index: Int, ComplexDouble) -> Unit): Unit {
    var index = 0
    for (item in this) action(index++, item)
}

public inline fun <R : Comparable<R>> ComplexFloatArray.maxBy(selector: (ComplexFloat) -> R): ComplexFloat =
    maxByOrNull(selector)!!

public inline fun <R : Comparable<R>> ComplexDoubleArray.maxBy(selector: (ComplexDouble) -> R): ComplexDouble =
    maxByOrNull(selector)!!

/**
 * Returns the first element yielding the largest value of the given function or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.maxByOrNull(selector: (ComplexFloat) -> R): ComplexFloat? {
    if (isEmpty()) return null
    var maxElem = this[0]
    val lastIndex = this.lastIndex
    if (lastIndex == 0) return maxElem
    var maxValue = selector(maxElem)
    for (i in 1..lastIndex) {
        val e = this[i]
        val v = selector(e)
        if (maxValue < v) {
            maxElem = e
            maxValue = v
        }
    }
    return maxElem
}

/**
 * Returns the first element yielding the largest value of the given function or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.maxByOrNull(selector: (ComplexDouble) -> R): ComplexDouble? {
    if (isEmpty()) return null
    var maxElem = this[0]
    val lastIndex = this.lastIndex
    if (lastIndex == 0) return maxElem
    var maxValue = selector(maxElem)
    for (i in 1..lastIndex) {
        val e = this[i]
        val v = selector(e)
        if (maxValue < v) {
            maxElem = e
            maxValue = v
        }
    }
    return maxElem
}

/**
 * Returns the largest value among all values produced by [selector] function
 * applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.maxOf(selector: (ComplexFloat) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (maxValue < v) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value among all values produced by [selector] function
 * applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.maxOf(selector: (ComplexDouble) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (maxValue < v) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value among all values produced by [selector] function
 * applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.maxOfOrNull(selector: (ComplexFloat) -> R): R? {
    if (isEmpty()) return null
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (maxValue < v) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value among all values produced by [selector] function
 * applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.maxOfOrNull(selector: (ComplexDouble) -> R): R? {
    if (isEmpty()) return null
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (maxValue < v) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R> ComplexFloatArray.maxOfWith(comparator: Comparator<in R>, selector: (ComplexFloat) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(maxValue, v) < 0) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R> ComplexDoubleArray.maxOfWith(comparator: Comparator<in R>, selector: (ComplexDouble) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(maxValue, v) < 0) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R> ComplexFloatArray.maxOfWithOrNull(comparator: Comparator<in R>, selector: (ComplexFloat) -> R): R? {
    if (isEmpty()) return null
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(maxValue, v) < 0) {
            maxValue = v
        }
    }
    return maxValue
}

/**
 * Returns the largest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R> ComplexDoubleArray.maxOfWithOrNull(comparator: Comparator<in R>, selector: (ComplexDouble) -> R): R? {
    if (isEmpty()) return null
    var maxValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(maxValue, v) < 0) {
            maxValue = v
        }
    }
    return maxValue
}

public inline fun <R : Comparable<R>> ComplexFloatArray.minBy(selector: (ComplexFloat) -> R): ComplexFloat =
    minByOrNull(selector)!!

public inline fun <R : Comparable<R>> ComplexDoubleArray.minBy(selector: (ComplexDouble) -> R): ComplexDouble =
    minByOrNull(selector)!!

/**
 * Returns the first element yielding the smallest value of the given function or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.minByOrNull(selector: (ComplexFloat) -> R): ComplexFloat? {
    if (isEmpty()) return null
    var minElem = this[0]
    val lastIndex = this.lastIndex
    if (lastIndex == 0) return minElem
    var minValue = selector(minElem)
    for (i in 1..lastIndex) {
        val e = this[i]
        val v = selector(e)
        if (minValue > v) {
            minElem = e
            minValue = v
        }
    }
    return minElem
}

/**
 * Returns the first element yielding the smallest value of the given function or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.minByOrNull(selector: (ComplexDouble) -> R): ComplexDouble? {
    if (isEmpty()) return null
    var minElem = this[0]
    val lastIndex = this.lastIndex
    if (lastIndex == 0) return minElem
    var minValue = selector(minElem)
    for (i in 1..lastIndex) {
        val e = this[i]
        val v = selector(e)
        if (minValue > v) {
            minElem = e
            minValue = v
        }
    }
    return minElem
}

/**
 * Returns the smallest value among all values produced by [selector] function
 * applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.minOf(selector: (ComplexFloat) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (minValue > v) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value among all values produced by [selector] function
 * applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.minOf(selector: (ComplexDouble) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (minValue > v) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value among all values produced by [selector] function
 * applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexFloatArray.minOfOrNull(selector: (ComplexFloat) -> R): R? {
    if (isEmpty()) return null
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (minValue > v) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value among all values produced by [selector] function
 * applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R : Comparable<R>> ComplexDoubleArray.minOfOrNull(selector: (ComplexDouble) -> R): R? {
    if (isEmpty()) return null
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (minValue > v) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R> ComplexFloatArray.minOfWith(comparator: Comparator<in R>, selector: (ComplexFloat) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(minValue, v) > 0) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array.
 *
 * @throws NoSuchElementException if the array is empty.
 */
public inline fun <R> ComplexDoubleArray.minOfWith(comparator: Comparator<in R>, selector: (ComplexDouble) -> R): R {
    if (isEmpty()) throw NoSuchElementException()
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(minValue, v) > 0) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R> ComplexFloatArray.minOfWithOrNull(comparator: Comparator<in R>, selector: (ComplexFloat) -> R): R? {
    if (isEmpty()) return null
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(minValue, v) > 0) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns the smallest value according to the provided [comparator]
 * among all values produced by [selector] function applied to each element in the array or `null` if there are no elements.
 */
public inline fun <R> ComplexDoubleArray.minOfWithOrNull(comparator: Comparator<in R>, selector: (ComplexDouble) -> R): R? {
    if (isEmpty()) return null
    var minValue = selector(this[0])
    for (i in 1..lastIndex) {
        val v = selector(this[i])
        if (comparator.compare(minValue, v) > 0) {
            minValue = v
        }
    }
    return minValue
}

/**
 * Returns `true` if the array has no elements.
 */
public fun ComplexFloatArray.none(): Boolean = isEmpty()

/**
 * Returns `true` if the array has no elements.
 */
public fun ComplexDoubleArray.none(): Boolean = isEmpty()

/**
 * Returns `true` if no elements match the given [predicate].
 */
public inline fun ComplexFloatArray.none(predicate: (ComplexFloat) -> Boolean): Boolean {
    for (element in this) if (predicate(element)) return false
    return true
}

/**
 * Returns `true` if no elements match the given [predicate].
 */
public inline fun ComplexDoubleArray.none(predicate: (ComplexDouble) -> Boolean): Boolean {
    for (element in this) if (predicate(element)) return false
    return true
}

/**
 * Performs the given [action] on each element and returns the array itself afterwards.
 */
public inline fun ComplexFloatArray.onEach(action: (ComplexFloat) -> Unit): ComplexFloatArray =
    apply { for (element in this) action(element) }

/**
 * Performs the given [action] on each element and returns the array itself afterwards.
 */
public inline fun ComplexDoubleArray.onEach(action: (ComplexDouble) -> Unit): ComplexDoubleArray =
    apply { for (element in this) action(element) }

/**
 * Performs the given [action] on each element, providing sequential index with the element,
 * and returns the array itself afterwards.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the action on the element.
 */
public inline fun ComplexFloatArray.onEachIndexed(action: (index: Int, ComplexFloat) -> Unit): ComplexFloatArray =
    apply { forEachIndexed(action) }

/**
 * Performs the given [action] on each element, providing sequential index with the element,
 * and returns the array itself afterwards.
 * @param [action] function that takes the index of an element and the element itself
 * and performs the action on the element.
 */
public inline fun ComplexDoubleArray.onEachIndexed(action: (index: Int, ComplexDouble) -> Unit): ComplexDoubleArray =
    apply { forEachIndexed(action) }

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes current accumulator value and an element,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduce(operation: (acc: ComplexFloat, ComplexFloat) -> ComplexFloat): ComplexFloat {
    if (isEmpty()) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes current accumulator value and an element,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduce(operation: (acc: ComplexDouble, ComplexDouble) -> ComplexDouble): ComplexDouble {
    if (isEmpty()) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceIndexedOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value and the element itself,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceIndexed(operation: (index: Int, acc: ComplexFloat, ComplexFloat) -> ComplexFloat): ComplexFloat {
    if (isEmpty()) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(index, accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceIndexedOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value and the element itself,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceIndexed(operation: (index: Int, acc: ComplexDouble, ComplexDouble) -> ComplexDouble): ComplexDouble {
    if (isEmpty()) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(index, accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value and the element itself,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceIndexedOrNull(operation: (index: Int, acc: ComplexFloat, ComplexFloat) -> ComplexFloat): ComplexFloat? {
    if (isEmpty()) return null
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(index, accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element with its index in the original array.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes the index of an element, current accumulator value and the element itself,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceIndexedOrNull(operation: (index: Int, acc: ComplexDouble, ComplexDouble) -> ComplexDouble): ComplexDouble? {
    if (isEmpty()) return null
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(index, accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes current accumulator value and an element,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceOrNull(operation: (acc: ComplexFloat, ComplexFloat) -> ComplexFloat): ComplexFloat? {
    if (isEmpty()) return null
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the first element and applying [operation] from left to right
 * to current accumulator value and each element.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes current accumulator value and an element,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceOrNull(operation: (acc: ComplexDouble, ComplexDouble) -> ComplexDouble): ComplexDouble? {
    if (isEmpty()) return null
    var accumulator = this[0]
    for (index in 1..lastIndex) {
        accumulator = operation(accumulator, this[index])
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceRightOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes an element and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceRight(operation: (ComplexFloat, acc: ComplexFloat) -> ComplexFloat): ComplexFloat {
    var index = lastIndex
    if (index < 0) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceRightOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes an element and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceRight(operation: (ComplexDouble, acc: ComplexDouble) -> ComplexDouble): ComplexDouble {
    var index = lastIndex
    if (index < 0) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceRightIndexedOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceRightIndexed(operation: (index: Int, ComplexFloat, acc: ComplexFloat) -> ComplexFloat): ComplexFloat {
    var index = lastIndex
    if (index < 0) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Throws an exception if this array is empty. If the array can be empty in an expected way,
 * please use [reduceRightIndexedOrNull] instead. It returns `null` when its receiver is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceRightIndexed(operation: (index: Int, ComplexDouble, acc: ComplexDouble) -> ComplexDouble): ComplexDouble {
    var index = lastIndex
    if (index < 0) throw UnsupportedOperationException("Empty array can't be reduced.")
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceRightIndexedOrNull(operation: (index: Int, ComplexFloat, acc: ComplexFloat) -> ComplexFloat): ComplexFloat? {
    var index = lastIndex
    if (index < 0) return null
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element with its index in the original array and current accumulator value.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes the index of an element, the element itself and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceRightIndexedOrNull(operation: (index: Int, ComplexDouble, acc: ComplexDouble) -> ComplexDouble): ComplexDouble? {
    var index = lastIndex
    if (index < 0) return null
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(index, get(index), accumulator)
        --index
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes an element and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.reduceRightOrNull(operation: (ComplexFloat, acc: ComplexFloat) -> ComplexFloat): ComplexFloat? {
    var index = lastIndex
    if (index < 0) return null
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Accumulates value starting with the last element and applying [operation] from right to left
 * to each element and current accumulator value.
 *
 * Returns `null` if the array is empty.
 *
 * @param [operation] function that takes an element and current accumulator value,
 * and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.reduceRightOrNull(operation: (ComplexDouble, acc: ComplexDouble) -> ComplexDouble): ComplexDouble? {
    var index = lastIndex
    if (index < 0) return null
    var accumulator = get(index--)
    while (index >= 0) {
        accumulator = operation(get(index--), accumulator)
    }
    return accumulator
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.runningFold(initial: R, operation: (acc: R, ComplexFloat) -> R): List<R> {
    if (isEmpty()) return listOf(initial)
    val result = ArrayList<R>(size + 1).apply { add(initial) }
    var accumulator = initial
    for (element in this) {
        accumulator = operation(accumulator, element)
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.runningFold(initial: R, operation: (acc: R, ComplexDouble) -> R): List<R> {
    if (isEmpty()) return listOf(initial)
    val result = ArrayList<R>(size + 1).apply { add(initial) }
    var accumulator = initial
    for (element in this) {
        accumulator = operation(accumulator, element)
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.runningFoldIndexed(initial: R, operation: (index: Int, acc: R, ComplexFloat) -> R): List<R> {
    if (isEmpty()) return listOf(initial)
    val result = ArrayList<R>(size + 1).apply { add(initial) }
    var accumulator = initial
    for (index in indices) {
        accumulator = operation(index, accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.runningFoldIndexed(initial: R, operation: (index: Int, acc: R, ComplexDouble) -> R): List<R> {
    if (isEmpty()) return listOf(initial)
    val result = ArrayList<R>(size + 1).apply { add(initial) }
    var accumulator = initial
    for (index in indices) {
        accumulator = operation(index, accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with the first element of this array.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.runningReduce(operation: (acc: ComplexFloat, ComplexFloat) -> ComplexFloat): List<ComplexFloat> {
    if (isEmpty()) return emptyList()
    var accumulator = this[0]
    val result = ArrayList<ComplexFloat>(size).apply { add(accumulator) }
    for (index in 1 until size) {
        accumulator = operation(accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with the first element of this array.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.runningReduce(operation: (acc: ComplexDouble, ComplexDouble) -> ComplexDouble): List<ComplexDouble> {
    if (isEmpty()) return emptyList()
    var accumulator = this[0]
    val result = ArrayList<ComplexDouble>(size).apply { add(accumulator) }
    for (index in 1 until size) {
        accumulator = operation(accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with the first element of this array.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun ComplexFloatArray.runningReduceIndexed(operation: (index: Int, acc: ComplexFloat, ComplexFloat) -> ComplexFloat): List<ComplexFloat> {
    if (isEmpty()) return emptyList()
    var accumulator = this[0]
    val result = ArrayList<ComplexFloat>(size).apply { add(accumulator) }
    for (index in 1 until size) {
        accumulator = operation(index, accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with the first element of this array.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun ComplexDoubleArray.runningReduceIndexed(operation: (index: Int, acc: ComplexDouble, ComplexDouble) -> ComplexDouble): List<ComplexDouble> {
    if (isEmpty()) return emptyList()
    var accumulator = this[0]
    val result = ArrayList<ComplexDouble>(size).apply { add(accumulator) }
    for (index in 1 until size) {
        accumulator = operation(index, accumulator, this[index])
        result.add(accumulator)
    }
    return result
}

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.scan(initial: R, operation: (acc: R, ComplexFloat) -> R): List<R> =
    runningFold(initial, operation)

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes current accumulator value and an element, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.scan(initial: R, operation: (acc: R, ComplexDouble) -> R): List<R> =
    runningFold(initial, operation)

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexFloatArray.scanIndexed(initial: R, operation: (index: Int, acc: R, ComplexFloat) -> R): List<R> =
    runningFoldIndexed(initial, operation)

/**
 * Returns a list containing successive accumulation values generated by applying [operation] from left to right
 * to each element, its index in the original array and current accumulator value that starts with [initial] value.
 *
 * Note that `acc` value passed to [operation] function should not be mutated;
 * otherwise it would affect the previous value in resulting list.
 *
 * @param [operation] function that takes the index of an element, current accumulator value
 * and the element itself, and calculates the next accumulator value.
 */
public inline fun <R> ComplexDoubleArray.scanIndexed(initial: R, operation: (index: Int, acc: R, ComplexDouble) -> R): List<R> =
    runningFoldIndexed(initial, operation)

/**
 * Splits the original array into pair of lists,
 * where *first* list contains elements for which [predicate] yielded `true`,
 * while *second* list contains elements for which [predicate] yielded `false`.
 */
public inline fun ComplexFloatArray.partition(predicate: (ComplexFloat) -> Boolean): Pair<List<ComplexFloat>, List<ComplexFloat>> {
    val first = ArrayList<ComplexFloat>()
    val second = ArrayList<ComplexFloat>()
    for (element in this) {
        if (predicate(element)) {
            first.add(element)
        } else {
            second.add(element)
        }
    }
    return Pair(first, second)
}

/**
 * Splits the original array into pair of lists,
 * where *first* list contains elements for which [predicate] yielded `true`,
 * while *second* list contains elements for which [predicate] yielded `false`.
 */
public inline fun ComplexDoubleArray.partition(predicate: (ComplexDouble) -> Boolean): Pair<List<ComplexDouble>, List<ComplexDouble>> {
    val first = ArrayList<ComplexDouble>()
    val second = ArrayList<ComplexDouble>()
    for (element in this) {
        if (predicate(element)) {
            first.add(element)
        } else {
            second.add(element)
        }
    }
    return Pair(first, second)
}

/**
 * Returns a list of pairs built from the elements of `this` array and the [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun <R> ComplexFloatArray.zip(other: Array<out R>): List<Pair<ComplexFloat, R>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of pairs built from the elements of `this` array and the [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun <R> ComplexDoubleArray.zip(other: Array<out R>): List<Pair<ComplexDouble, R>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of values built from the elements of `this` array and the [other] array with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest collection.
 */
public inline fun <R, V> ComplexFloatArray.zip(other: Array<out R>, transform: (a: ComplexFloat, b: R) -> V): List<V> {
    val size = minOf(size, other.size)
    val list = ArrayList<V>(size)
    for (i in 0 until size) {
        list.add(transform(this[i], other[i]))
    }
    return list
}

/**
 * Returns a list of values built from the elements of `this` array and the [other] array with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest collection.
 */
public inline fun <R, V> ComplexDoubleArray.zip(other: Array<out R>, transform: (a: ComplexDouble, b: R) -> V): List<V> {
    val size = minOf(size, other.size)
    val list = ArrayList<V>(size)
    for (i in 0 until size) {
        list.add(transform(this[i], other[i]))
    }
    return list
}

/**
 * Returns a list of pairs built from the elements of `this` collection and [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun <R> ComplexFloatArray.zip(other: Iterable<R>): List<Pair<ComplexFloat, R>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of pairs built from the elements of `this` collection and [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun <R> ComplexDoubleArray.zip(other: Iterable<R>): List<Pair<ComplexDouble, R>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of values built from the elements of `this` array and the [other] collection with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest collection.
 */
public inline fun <R, V> ComplexFloatArray.zip(other: Iterable<R>, transform: (a: ComplexFloat, b: R) -> V): List<V> {
    val arraySize = size
    val list = ArrayList<V>(minOf(if (other is Collection<*>) other.size else 10, arraySize))
    var i = 0
    for (element in other) {
        if (i >= arraySize) break
        list.add(transform(this[i++], element))
    }
    return list
}

/**
 * Returns a list of values built from the elements of `this` array and the [other] collection with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest collection.
 */
public inline fun <R, V> ComplexDoubleArray.zip(other: Iterable<R>, transform: (a: ComplexDouble, b: R) -> V): List<V> {
    val arraySize = size
    val list = ArrayList<V>(minOf(if (other is Collection<*>) other.size else 10, arraySize))
    var i = 0
    for (element in other) {
        if (i >= arraySize) break
        list.add(transform(this[i++], element))
    }
    return list
}

/**
 * Returns a list of pairs built from the elements of `this` array and the [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun ComplexFloatArray.zip(other: ComplexFloatArray): List<Pair<ComplexFloat, ComplexFloat>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of pairs built from the elements of `this` array and the [other] array with the same index.
 * The returned list has length of the shortest collection.
 */
public infix fun ComplexDoubleArray.zip(other: ComplexDoubleArray): List<Pair<ComplexDouble, ComplexDouble>> =
    zip(other) { t1, t2 -> t1 to t2 }

/**
 * Returns a list of values built from the elements of `this` array and the [other] array with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest array.
 */
public inline fun <V> ComplexFloatArray.zip(other: ComplexFloatArray, transform: (a: ComplexFloat, b: ComplexFloat) -> V): List<V> {
    val size = minOf(size, other.size)
    val list = ArrayList<V>(size)
    for (i in 0 until size) {
        list.add(transform(this[i], other[i]))
    }
    return list
}

/**
 * Returns a list of values built from the elements of `this` array and the [other] array with the same index
 * using the provided [transform] function applied to each pair of elements.
 * The returned list has length of the shortest array.
 */
public inline fun <V> ComplexDoubleArray.zip(other: ComplexDoubleArray, transform: (a: ComplexDouble, b: ComplexDouble) -> V): List<V> {
    val size = minOf(size, other.size)
    val list = ArrayList<V>(size)
    for (i in 0 until size) {
        list.add(transform(this[i], other[i]))
    }
    return list
}

/**
 * Appends the string from all the elements separated using [separator] and using the given [prefix] and [postfix] if supplied.
 *
 * If the collection could be huge, you can specify a non-negative value of [limit], in which case only the first [limit]
 * elements will be appended, followed by the [truncated] string (which defaults to "...").
 */
public fun <A : Appendable> ComplexFloatArray.joinTo(buffer: A, separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "", limit: Int = -1, truncated: CharSequence = "...", transform: ((ComplexFloat) -> CharSequence)? = null): A {
    buffer.append(prefix)
    var count = 0
    for (element in this) {
        if (++count > 1) buffer.append(separator)
        if (limit < 0 || count <= limit) {
            if (transform != null)
                buffer.append(transform(element))
            else
                buffer.append(element.toString())
        } else break
    }
    if (limit in 0 until count) buffer.append(truncated)
    buffer.append(postfix)
    return buffer
}

/**
 * Appends the string from all the elements separated using [separator] and using the given [prefix] and [postfix] if supplied.
 *
 * If the collection could be huge, you can specify a non-negative value of [limit], in which case only the first [limit]
 * elements will be appended, followed by the [truncated] string (which defaults to "...").
 */
public fun <A : Appendable> ComplexDoubleArray.joinTo(buffer: A, separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "", limit: Int = -1, truncated: CharSequence = "...", transform: ((ComplexDouble) -> CharSequence)? = null): A {
    buffer.append(prefix)
    var count = 0
    for (element in this) {
        if (++count > 1) buffer.append(separator)
        if (limit < 0 || count <= limit) {
            if (transform != null)
                buffer.append(transform(element))
            else
                buffer.append(element.toString())
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
public fun ComplexFloatArray.joinToString(separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "", limit: Int = -1, truncated: CharSequence = "...", transform: ((ComplexFloat) -> CharSequence)? = null): String =
    joinTo(StringBuilder(), separator, prefix, postfix, limit, truncated, transform).toString()

/**
 * Creates a string from all the elements separated using [separator] and using the given [prefix] and [postfix] if supplied.
 *
 * If the collection could be huge, you can specify a non-negative value of [limit], in which case only the first [limit]
 * elements will be appended, followed by the [truncated] string (which defaults to "...").
 */
public fun ComplexDoubleArray.joinToString(separator: CharSequence = ", ", prefix: CharSequence = "", postfix: CharSequence = "", limit: Int = -1, truncated: CharSequence = "...", transform: ((ComplexDouble) -> CharSequence)? = null): String =
    joinTo(StringBuilder(), separator, prefix, postfix, limit, truncated, transform).toString()

/**
 * Creates an [Iterable] instance that wraps the original array returning its elements when being iterated.
 */
public fun ComplexFloatArray.asIterable(): Iterable<ComplexFloat> {
    if (isEmpty()) return emptyList()
    return Iterable { this.iterator() }
}

/**
 * Creates an [Iterable] instance that wraps the original array returning its elements when being iterated.
 */
public fun ComplexDoubleArray.asIterable(): Iterable<ComplexDouble> {
    if (isEmpty()) return emptyList()
    return Iterable { this.iterator() }
}

/**
 * Creates a [Sequence] instance that wraps the original array returning its elements when being iterated.
 */
public fun ComplexFloatArray.asSequence(): Sequence<ComplexFloat> {
    if (isEmpty()) return emptySequence()
    return Sequence { this.iterator() }
}

/**
 * Creates a [Sequence] instance that wraps the original array returning its elements when being iterated.
 */
public fun ComplexDoubleArray.asSequence(): Sequence<ComplexDouble> {
    if (isEmpty()) return emptySequence()
    return Sequence { this.iterator() }
}

/**
 * Returns the sum of all elements in the array.
 */
public fun ComplexFloatArray.sum(): ComplexFloat {
    var sum = ComplexFloat(0f, 0f)
    for (element in this) {
        sum += element
    }
    return sum
}

/**
 * Returns the sum of all elements in the array.
 */
public fun ComplexDoubleArray.sum(): ComplexDouble {
    var sum = ComplexDouble(0.0, 0.0)
    for (element in this) {
        sum += element
    }
    return sum
}

private fun checkRangeIndexes(fromIndex: Int, toIndex: Int, size: Int) {
    if (fromIndex < 0 || toIndex > size)
        throw IndexOutOfBoundsException("fromIndex: $fromIndex, toIndex: $toIndex, size: $size")
    if (fromIndex > toIndex)
        throw IllegalArgumentException("fromIndex: $fromIndex > toIndex: $toIndex")
}
