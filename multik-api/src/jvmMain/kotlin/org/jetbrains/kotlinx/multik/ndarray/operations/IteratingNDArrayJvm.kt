/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray


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
