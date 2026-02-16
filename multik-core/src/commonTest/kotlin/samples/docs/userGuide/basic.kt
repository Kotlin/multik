package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.test.Test
import kotlin.test.assertEquals

class Basic {
    @Test
    fun creating_arrays() {
        // SampleStart
        val v = mk.ndarray(mk[1, 2, 3])
        val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        // SampleEnd
    }

    @Test
    fun inspecting_properties() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

        m.shape // (2, 2)
        m.size  // 4
        m.dim   // D2
        m.dtype // DataType.DoubleDataType
        // SampleEnd
    }

    @Test
    fun elementWise_arithmetic() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val b = mk.ndarray(mk[4, 5, 6])

        val sum = a + b   // [5, 7, 9]
        val scaled = a * 2 // [2, 4, 6]
        // SampleEnd
        assertEquals(mk.ndarray(mk[5, 7, 9]), sum)
        assertEquals(mk.ndarray(mk[2, 4, 6]), scaled)
    }

    @Test
    fun inPlace_operations() {
        // SampleStart
        val x = mk.ndarray(mk[1, 2, 3])

        x += 10
        // [11, 12, 13]
        // SampleEnd
        assertEquals(mk.ndarray(mk[11, 12, 13]), x)
    }

    @Test
    fun indexing_basics() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

        m[1, 0] // 3.0
        // SampleEnd
        assertEquals(3.0, m[1, 0])
    }
}