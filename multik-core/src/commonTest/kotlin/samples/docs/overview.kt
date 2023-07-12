package samples.docs

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import kotlin.test.Test

class Overview {
    @Test
    fun create_kotlin_matrix() {
        // SampleStart
        val a = listOf(listOf(1, 2, 3), listOf(4, 5, 6))
        val b = listOf(listOf(7, 8, 9), listOf(10, 11, 12))
        val c = MutableList(2) { MutableList(3) { 0 } }
        for (i in a.indices) {
            for (j in a.first().indices) {
                c[i][j] = a[i][j] * b[i][j]
            }
        }
        println(c) //[[7, 16, 27], [40, 55, 72]]
        // SampleEnd
    }

    @Test
    fun create_multik_matrix() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
        val b = mk.ndarray(mk[mk[7, 8, 9], mk[10, 11, 12]])
        val c = a * b
        println(c)
        /*
        [[7, 16, 27],
         [40, 55, 72]]
         */
        // SampleEnd
    }
}