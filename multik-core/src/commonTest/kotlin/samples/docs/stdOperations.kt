package samples.docs

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.filter
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import kotlin.test.Test
import kotlin.test.assertEquals

class StandardOperations {
    @Test
    fun small_example_collection_operations() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
        val b = a.filter { it > 2 }
        println(b)  // [3, 4, 5]
        val c = a.map { it * 2 }
        println(c)  // [2, 4, 6, 8, 10]
        val d = a.reduce { acc, value -> acc + value }
        println(d)  // 15
        // SampleEnd
        assertEquals(mk.ndarray(mk[3, 4, 5]), b)
        assertEquals(mk.ndarray(mk[2, 4, 6, 8, 10]), c)
        assertEquals(15, d)
    }
}