package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Row-major iterator for non-contiguous [NDArray] views.
 *
 * Traverses elements by computing flat indices from multi-dimensional coordinates using
 * [offset] and [strides]. Used when the array is not [consistent][NDArray.consistent]
 * (e.g. after slicing or transposing).
 *
 * @param T the element type.
 * @param data the underlying [MemoryView].
 * @param offset the starting offset in [data].
 * @param strides per-dimension strides.
 * @param shape per-dimension sizes.
 */
public class NDArrayIterator<T>(
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
