package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.d4array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.expandNDims
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertNull
import kotlin.test.assertSame

/**
 * Shape changes must never move data unless the requested layout genuinely cannot be strided over
 * the existing buffer, and must always produce the same elements in the same order either way.
 */
class ReshapeTest {

    // Views

    @Test
    fun testReshapeOfContiguousArrayIsAView() {
        val a = mk.d2array(2, 3) { it }
        val b = a.reshape(3, 2)

        assertSame(a.data, b.data)
        assertSame(a, b.base)
        assertContentEquals(intArrayOf(2, 1), b.strides)
        assertEquals(mk.ndarray(mk[mk[0, 1], mk[2, 3], mk[4, 5]]), b)
    }

    @Test
    fun testReshapeOfSliceIsAViewKeepingOffset() {
        val a = mk.d2array(4, 4) { it }
        val slice = a[1 until 3] as NDArray<Int, D2>
        val b = slice.reshape(2, 2, 2)

        assertSame(a.data, b.data)
        assertSame(a, b.base)
        assertEquals(slice.offset, b.offset)
        assertEquals(slice.toList(), b.toList())
    }

    @Test
    fun testReshapeWritesThroughToTheSource() {
        val a = mk.d2array(2, 3) { it }
        val b = a.reshape(6) as D1Array<Int>
        b[0] = 42

        assertEquals(42, a[0, 0])
    }

    @Test
    fun testChainedReshapeOfAViewStaysAView() {
        val a = mk.d3array(2, 3, 4) { it }
        val slice = a[0 until 1] as NDArray<Int, D3>
        val b = slice.reshape(12).reshape(3, 4).reshape(2, 2, 3)

        assertSame(a.data, b.data)
        assertEquals(slice.toList(), b.toList())
    }

    @Test
    fun testReshapeOfSteppedSliceIsAView() {
        // A stepped 1D slice is not `consistent`, but (3, 2) is still reachable with strides
        // (2 * step, step) over the same buffer.
        val a = mk.ndarrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        val stepped = a[sl.bounds..2] as NDArray<Int, D1>
        val b = stepped.reshape(3, 2)

        assertSame(a.data, b.data)
        assertContentEquals(intArrayOf(4, 2), b.strides)
        assertEquals(stepped.toList(), b.toList())
        assertEquals(listOf(0, 2, 4, 6, 8, 10), b.toList())
    }

    @Test
    fun testReshapeOfTransposedArrayIsAViewWhenAxesAreNotMixed() {
        // (3, 2) -> (3, 2, 1) only adds a size-one axis, which every layout can express.
        val t = mk.d2array(2, 3) { it }.transpose()
        val b = t.reshape(3, 2, 1)

        assertSame(t.data, b.data)
        assertEquals(t.toList(), b.toList())
    }

    // Copies

    @Test
    fun testReshapeOfTransposedArrayCopiesWhenAxesAreMixed() {
        val a = mk.d2array(2, 3) { it }
        val t = a.transpose()
        val b = t.reshape(6) as D1Array<Int>

        assertNotSame(a.data, b.data)
        assertNull(b.base)
        assertEquals(listOf(0, 3, 1, 4, 2, 5), b.toList())

        b[0] = 42
        assertEquals(0, a[0, 0])
    }

    @Test
    fun testReshapeOfNonContiguousSliceCopiesWhenAxesAreMerged() {
        val a = mk.d2array(3, 4) { it }
        val slice = a[0 until 3, 0 until 2] as NDArray<Int, D2>

        // Rows are 2 elements wide but 4 apart in the buffer, so merging them cannot be strided.
        val merged = slice.reshape(6) as D1Array<Int>
        assertNotSame(a.data, merged.data)
        assertNull(merged.base)
        assertEquals(listOf(0, 1, 4, 5, 8, 9), merged.toList())

        // Appending size-one axes merges nothing, so the same slice stays a view.
        val padded = slice.reshape(3, 2, 1, 1)
        assertSame(a.data, padded.data)
        assertEquals(slice.toList(), padded.toList())
    }

    // squeeze / unsqueeze

    @Test
    fun testSqueezeOfStridedSubArrayKeepsElements() {
        // Regression: squeeze used to recompute packed strides and silently read the wrong elements.
        val a = mk.d3array(2, 3, 4) { it }
        val slice = a[0 until 2, 1 until 2, 0 until 4] as NDArray<Int, D3>
        val squeezed = slice.squeeze()

        assertContentEquals(intArrayOf(2, 4), squeezed.shape)
        assertEquals(listOf(4, 5, 6, 7, 16, 17, 18, 19), squeezed.toList())
        assertSame(a.data, squeezed.data)
    }

    @Test
    fun testSqueezeOfSelectedAxisKeepsElements() {
        val a = mk.d4array(2, 1, 1, 3) { it }
        val transposed = a.transpose(0, 2, 1, 3)
        val squeezed = transposed.squeeze(1)

        assertContentEquals(intArrayOf(2, 1, 3), squeezed.shape)
        assertEquals(transposed.toList(), squeezed.toList())
    }

    @Test
    fun testUnsqueezeOfStridedSubArrayIsAView() {
        val a = mk.d2array(2, 3) { it }
        val slice = a[0 until 2, 1 until 3] as NDArray<Int, D2>

        for (axis in 0..2) {
            val unsqueezed = slice.unsqueeze(axis)
            assertSame(a.data, unsqueezed.data, "unsqueeze($axis) copied")
            assertEquals(slice.toList(), unsqueezed.toList(), "unsqueeze($axis) changed elements")
            assertEquals(1, unsqueezed.shape[axis])
        }
    }

    @Test
    fun testSqueezeUnsqueezeRoundTrip() {
        val a = mk.d3array(2, 3, 4) { it }
        val slice = a[0 until 2, 0 until 2, 1 until 3] as NDArray<Int, D3>
        val roundTrip = slice.unsqueeze(0, 2).squeeze(0, 2)

        assertContentEquals(slice.shape, roundTrip.shape)
        assertEquals(slice.toList(), roundTrip.toList())
    }

    @Test
    fun testSqueezeKeepsOneAxisWhenEveryAxisWouldBeDropped() {
        // Multik has no rank-0 arrays, so squeezing everything leaves shape [1] instead of throwing.
        assertContentEquals(intArrayOf(1), mk.zeros<Double>(1, 1).squeeze().shape)
        assertContentEquals(intArrayOf(1), mk.zeros<Double>(1, 1, 1).squeeze().shape)
        assertContentEquals(intArrayOf(1), mk.ndarrayOf(5).squeeze().shape)
        // Which is exactly what naming the axes explicitly already returned.
        assertContentEquals(intArrayOf(1), mk.zeros<Double>(1, 1).squeeze(0).shape)
        assertContentEquals(intArrayOf(1), mk.zeros<Double>(1, 1).squeeze(0, 1).shape)

        val a = mk.zeros<Double>(1, 1)
        assertSame(a.data, a.squeeze().data)
        assertEquals(listOf(0.0), a.squeeze().toList())
    }

    @Test
    fun testSqueezeRejectsInvalidAxes() {
        val a = mk.zeros<Double>(1, 3)
        assertFailsWith<IllegalArgumentException> { a.squeeze(1) }.let {
            assertContains(it.message.orEmpty(), "not 1")
        }
        assertFailsWith<IllegalArgumentException> { a.squeeze(7) }
        assertFailsWith<IllegalArgumentException> { a.squeeze(-1) }
    }

    @Test
    fun testUnsqueezeRejectsInvalidAxes() {
        val a = mk.ndarrayOf(1, 2, 3)
        // Axes address the resulting array, so 2 is valid for a single insertion but 5 is not.
        assertContentEquals(intArrayOf(3, 1), a.unsqueeze(1).shape)
        assertFailsWith<IllegalArgumentException> { a.unsqueeze(5) }
        assertFailsWith<IllegalArgumentException> { a.unsqueeze(-1) }
        assertFailsWith<IllegalArgumentException> { a.unsqueeze(1, 1) }
    }

    // expandDims

    @Test
    fun testExpandDimsInsertsAxisAtTheGivenPosition() {
        val d1 = mk.ndarrayOf(1, 2, 3)
        assertContentEquals(intArrayOf(3, 1), d1.expandDims(1).shape)
        assertContentEquals(intArrayOf(1, 3), d1.expandDims(0).shape)

        val d2 = mk.d2array(2, 3) { it }
        assertContentEquals(intArrayOf(2, 1, 3), d2.expandDims(1).shape)

        val d3 = mk.d3array(2, 3, 4) { it }
        assertContentEquals(intArrayOf(2, 3, 1, 4), d3.expandDims(2).shape)

        val d4 = mk.d4array(2, 3, 4, 5) { it }
        assertContentEquals(intArrayOf(2, 3, 1, 4, 5), d4.expandDims(2).shape)

        assertContentEquals(intArrayOf(1, 2, 1, 3), d2.expandNDims(0, 2).shape)
    }

    @Test
    fun testExpandDimsIsAView() {
        val a = mk.d2array(2, 3) { it }
        val slice = a[0 until 2, 1 until 3] as NDArray<Int, D2>
        val expanded = slice.expandDims(1)

        assertSame(a.data, expanded.data)
        assertEquals(slice.toList(), expanded.toList())
    }

    // Dimension bookkeeping

    @Test
    fun testReshapeReportsTheRequestedDimension() {
        // `unsqueeze` yields DN(2); reshaping back to a 2D shape must report D2, or the result
        // compares unequal to an identically shaped D2 array.
        val a = mk.ndarrayOf(1, 2, 3, 4, 5, 6).unsqueeze(0)
        assertEquals(mk.ndarray(mk[mk[1, 2, 3, 4, 5, 6]]), a.reshape(1, 6))
        assertEquals(mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]]), a.reshape(2, 3))
    }

    // Edge cases

    @Test
    fun testShapeChangesOfAnEmptyArray() {
        val a = mk.zeros<Double>(0, 3)

        val unsqueezed = a.unsqueeze(0)
        assertContentEquals(intArrayOf(1, 0, 3), unsqueezed.shape)
        assertEquals(0, unsqueezed.size)

        val squeezed = mk.zeros<Double>(0, 1, 3).squeeze()
        assertContentEquals(intArrayOf(0, 3), squeezed.shape)
    }

    // reshapeStrides

    @Test
    fun testReshapeStridesOfContiguousLayout() {
        assertContentEquals(intArrayOf(2, 1), reshapeStrides(intArrayOf(2, 3), intArrayOf(3, 1), intArrayOf(3, 2)))
        assertContentEquals(intArrayOf(1), reshapeStrides(intArrayOf(2, 3), intArrayOf(3, 1), intArrayOf(6)))
        assertContentEquals(
            intArrayOf(6, 3, 1),
            reshapeStrides(intArrayOf(2, 3), intArrayOf(3, 1), intArrayOf(1, 2, 3))
        )
    }

    @Test
    fun testReshapeStridesIgnoresSizeOneAxes() {
        // A size-one axis in the middle must not break up an otherwise contiguous group.
        assertContentEquals(
            intArrayOf(12, 1),
            reshapeStrides(intArrayOf(2, 1, 4), intArrayOf(12, 4, 1), intArrayOf(2, 4))
        )
    }

    @Test
    fun testReshapeStridesRejectsLayoutsThatNeedACopy() {
        // Transposed (3, 2): rows are 1 apart, columns 3 apart — the axes cannot be merged.
        assertNull(reshapeStrides(intArrayOf(3, 2), intArrayOf(1, 3), intArrayOf(6)))
        // A slice whose rows are further apart than their own width.
        assertNull(reshapeStrides(intArrayOf(3, 2), intArrayOf(4, 1), intArrayOf(6)))
    }

    @Test
    fun testReshapeStridesKeepsConsistentLayoutsPacked() {
        // A view of a contiguous array must stay recognisably contiguous after reshaping.
        for (newShape in listOf(intArrayOf(24), intArrayOf(4, 6), intArrayOf(2, 3, 4), intArrayOf(1, 24, 1))) {
            assertContentEquals(
                computeStrides(newShape),
                reshapeStrides(intArrayOf(2, 12), intArrayOf(12, 1), newShape),
                newShape.joinToString(prefix = "shape(", postfix = ")")
            )
        }
    }
}
