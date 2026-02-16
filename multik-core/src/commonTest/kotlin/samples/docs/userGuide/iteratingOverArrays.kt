package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.forEach
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachMultiIndexed
import kotlin.test.Test
import kotlin.test.assertEquals

class IteratingOverArrays {
    @Test
    fun elementWise_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        for (value in a) {
            print("$value ")
        }
        // 1 2 3
        // SampleEnd
    }

    @Test
    fun forEach_example() {
        val a = mk.ndarray(mk[1, 2, 3])
        // SampleStart
        a.forEach { value ->
            println(value)
        }
        // SampleEnd
    }

    @Test
    fun indices_example() {
        // SampleStart
        val v = mk.ndarray(mk[10, 20, 30])

        for (i in v.indices) {
            println("v[$i] = ${v[i]}")
        }

        v.forEachIndexed { i, value ->
            println("$i -> $value")
        }
        // SampleEnd
    }

    @Test
    fun multiIndices_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])

        for (index in m.multiIndices) {
            print("${m[index]} ")
        }
        // 1.5 2.1 3.0 4.0 5.0 6.0
        // SampleEnd
    }

    @Test
    fun forEachMultiIndexed_example() {
        val m = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        // SampleStart
        m.forEachMultiIndexed { index, value ->
            println("${index.joinToString()} -> $value")
        }
        // SampleEnd
    }

    @Test
    fun update_example() {
        // SampleStart
        val x = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        for (index in x.multiIndices) {
            x[index] = x[index] * 2
        }
        // [[2, 4], [6, 8]]
        // SampleEnd
        assertEquals(mk.ndarray(mk[mk[2, 4], mk[6, 8]]), x)
    }
}