package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray


/**
 * Returns a [SortedSet][java.util.SortedSet] of all elements.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.toSortedSet(): java.util.SortedSet<T> {
    return toCollection(java.util.TreeSet())
}

/**
 * Returns a [SortedSet][java.util.SortedSet] of all elements.
 *
 * Elements in the set returned are sorted according to the given [comparator].
 */
public fun <T, D : Dimension> MultiArray<T, D>.toSortedSet(comparator: Comparator<in T>): java.util.SortedSet<T> {
    return toCollection(java.util.TreeSet(comparator))
}
