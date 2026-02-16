package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.test.Test

class InputAndOutput {
    @Test
    fun lists_and_collections() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val list = a.toList()           // List<Int>
        val mutable = a.toMutableList() // MutableList<Int>
        val set = a.toSet()             // Set<Int>
        // SampleEnd
    }

    @Test
    fun nested_lists() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val list2 = m.toListD2() // List<List<Double>>
        // SampleEnd
    }

    @Test
    fun primitive_arrays() {
        // SampleStart
        val v = mk.ndarray(mk[1, 2, 3])
        val ints = v.toIntArray() // IntArray
        // SampleEnd
    }

    @Test
    fun nested_primitive_arrays() {
        // SampleStart
        val mat = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val arr2 = mat.toArray() // Array<DoubleArray>
        // SampleEnd
    }
}