package samples.docs

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.rangeTo
import kotlin.test.Test
import kotlin.test.assertEquals

class IndexingAndIterating {
    @Test
    fun simple_indexing() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a[2] // select the element at index 2

        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        b[1, 2] // select the element at row 1 column 2
        // SampleEnd
        assertEquals(3, a[2])
        assertEquals(6.0, b[1, 2])
    }

    @Test
    fun slice_1() {
        // SampleStart
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // select elements at rows 0 and 1 in column 1
        b[0..<2, 1] // [2.1, 5.0]
        // SampleEnd
        assertEquals(mk.ndarray(mk[2.1, 5.0]), b[0..<2, 1])
    }

    @Test
    fun slice_2() {
        // SampleStart
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // select row 1
        b[1] // [4.0, 5.0, 6.0]
        b[1, 0..2..1] // [4.0, 5.0, 6.0]
        // SampleEnd

        assertEquals(b[1], b[1, 0..2..1])
    }

    @Test
    fun iterating() {
        // SampleStart
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        for (el in b) {
            print("$el, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
        }
        // SampleEnd
    }

    @Test
    fun iterating_multiIndices() {
        // SampleStart
        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        for (index in b.multiIndices) {
            print("${b[index]}, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
        }
        // SampleEnd
    }
}