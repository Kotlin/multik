package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.test.Test
import kotlin.test.assertEquals

class IndexingAndSlicing {
    @Test
    fun indexing_basics() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a[2] // 3

        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        b[1, 2] // 6.0
        // SampleEnd
        assertEquals(3, a[2])
        assertEquals(6.0, b[1, 2])
    }

    @Test
    fun dimension_reduction() {
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // SampleStart
        val row = b[1] // D1 array: [4.0, 5.0, 6.0]
        // SampleEnd
    }

    @Test
    fun slicing_with_ranges() {
        // SampleStart
        val c = mk.ndarray(mk[10, 20, 30, 40, 50])

        c[1..3]   // [20, 30, 40]
        c[0..<3]  // [10, 20, 30]
        // SampleEnd
    }

    @Test
    fun step_slicing() {
        val c = mk.ndarray(mk[10, 20, 30, 40, 50])
        // SampleStart
        c[0..4..2] // [10, 30, 50]
        // SampleEnd
    }

    @Test
    fun multi_axis_slicing() {
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // SampleStart
        // take rows 0 and 1 from column 1
        b[0..<2, 1] // [2.1, 5.0]

        // same as selecting the entire second axis
        b[1]         // [4.0, 5.0, 6.0]
        b[1, 0..2..1] // [4.0, 5.0, 6.0]
        // SampleEnd
    }

    @Test
    fun open_ended_slices() {
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        val c = mk.ndarray(mk[10, 20, 30, 40, 50])
        // SampleStart
        val col = b[sl.bounds, 1] // full rows, column 1
        val head = c[sl.first..2] // [10, 20, 30]
        val tail = c[2..sl.last]  // [30, 40, 50]
        // SampleEnd
    }

    @Test
    fun keeping_dimensions() {
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // SampleStart
        val keepDim = b[1..1] // D2 array with shape (1, 3)
        // SampleEnd
    }

    @Test
    fun slicing_by_axis() {
        // SampleStart
        val d = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

        val byAxis = d.slice<Int, D2, D2>(0..1, axis = 1) // take columns 0..1
        // SampleEnd
    }

    @Test
    fun nd_slicing() {
        // SampleStart
        val e = mk.d3array(2, 3, 4) { it }
        val view = e.slice<Int, D3, D2>(mapOf(0 to 0..1.r, 2 to (0..2).toSlice()))
        // axis 0 is indexed, axis 2 is sliced, axis 1 is kept as full slice
        // SampleEnd
    }
}