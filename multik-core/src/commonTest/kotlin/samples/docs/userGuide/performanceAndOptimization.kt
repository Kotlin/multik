package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PerformanceAndOptimization {
    @Test
    fun deepCopy_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val v = a[0] // view

        assertFalse(v.consistent)

        val c = v.deepCopy()
        assertTrue(c.consistent)
        // SampleEnd
    }

    @Test
    fun inplace_example() {
        // SampleStart
        val x = mk.ndarray(mk[1.0, 2.0, 3.0])

        x += 1.0
        x *= 2.0
        // SampleEnd
        assertEquals(mk.ndarray(mk[4.0, 6.0, 8.0]), x)
    }
}