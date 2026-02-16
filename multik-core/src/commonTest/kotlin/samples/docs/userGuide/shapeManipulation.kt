package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class ShapeManipulation {
    @Test
    fun reshape_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])
        val b = a.reshape(2, 3)
        /*
        [[1, 2, 3],
         [4, 5, 6]]
         */

        val c = b.reshape(3, 2)
        /*
        [[1, 2],
         [3, 4],
         [5, 6]]
         */
        // SampleEnd
        assertEquals(mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]]), b)
        assertEquals(mk.ndarray(mk[mk[1, 2], mk[3, 4], mk[5, 6]]), c)
    }

    @Test
    fun flatten_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val flat = m.flatten()
        // [1, 2, 3, 4]
        // SampleEnd
        assertEquals(mk.ndarray(mk[1, 2, 3, 4]), flat)
    }

    @Test
    fun transpose_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4], mk[5, 6]])
        val t = m.transpose()
        /*
        [[1, 3, 5],
         [2, 4, 6]]
         */
        // SampleEnd
        assertEquals(mk.ndarray(mk[mk[1, 3, 5], mk[2, 4, 6]]), t)
    }

    @Test
    fun transpose_2_example() {
        // SampleStart
        val x = mk.d3array(2, 3, 4) { it }
        val y = x.transpose(1, 0, 2) // swap axes 0 and 1
        // SampleEnd
    }

    @Test
    fun squeeze_example() {
        // SampleStart
        val a = mk.zeros<Double>(1, 3, 1)
        val b = a.squeeze()
        // shape is (3)
        // SampleEnd
        assertContentEquals(intArrayOf(3), b.shape)
    }

    @Test
    fun unsqueeze_example() {
        // SampleStart
        val v = mk.ndarray(mk[1, 2, 3])
        val col = v.unsqueeze(1) // shape (3, 1)
        val row = v.unsqueeze(0) // shape (1, 3)
        // SampleEnd
        assertContentEquals(intArrayOf(3, 1), col.shape)
        assertContentEquals(intArrayOf(1, 3), row.shape)
    }

    @Test
    fun cat_example() {
        // SampleStart
        val left = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val right = mk.ndarray(mk[mk[5, 6], mk[7, 8]])

        val byRows = left.cat(right, axis = 0)
        /*
        [[1, 2],
         [3, 4],
         [5, 6],
         [7, 8]]
         */

        val byCols = left.cat(right, axis = 1)
        /*
        [[1, 2, 5, 6],
         [3, 4, 7, 8]]
         */
        // SampleEnd
        assertEquals(mk.ndarray(mk[mk[1, 2], mk[3, 4], mk[5, 6], mk[7, 8]]), byRows)
        assertEquals(mk.ndarray(mk[mk[1, 2, 5, 6], mk[3, 4, 7, 8]]), byCols)
    }

    @Test
    fun stack_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val b = mk.ndarray(mk[4, 5, 6])
        val stacked = mk.stack(a, b) // shape (2, 3)
        // SampleEnd
        assertEquals(mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]]), stacked)
        assertContentEquals(intArrayOf(2, 3), stacked.shape)
    }
}