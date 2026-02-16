package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.test.Test
import kotlin.test.assertEquals

class AboutMultik {
    @Test
    fun tiny_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val b = mk.ndarray(mk[4, 5, 6])
        val c = a + b
        println(c) // [5, 7, 9]
        // SampleEnd
        assertEquals(mk.ndarray(mk[5, 7, 9]), c)
    }
}