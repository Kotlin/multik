package samples.docs

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.test.Test
import kotlin.test.assertEquals

class ArithmeticOperations {
    @Test
    fun arith_with_scalars() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        println(3.3 + a)
        /*
        [[4.8, 5.4, 6.3],
        [7.3, 8.3, 9.3]]
         */

        println(a * 2.0)
        /*
        [[3.0, 4.2, 6.0],
        [8.0, 10.0, 12.0]]
         */
        // SampleEnd
        assertEquals(listOf(4.8, 5.4, 6.3, 7.3, 8.3, 9.3), (3.3 + a).toList())
        assertEquals(listOf(3.0, 4.2, 6.0, 8.0, 10.0, 12.0), (a * 2.0).toList())
    }

    @Test
    fun div_with_ndarrays() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        val b = mk.ndarray(mk[mk[1.0, 1.3, 3.0], mk[4.0, 9.5, 5.0]])
        a / b // division
        /*
        [[1.5, 1.6153846153846154, 1.0],
        [1.0, 0.5263157894736842, 1.2]]
        */
        // SampleEnd

        val actual = a / b
        assertEquals(true, intArrayOf(2, 3).contentEquals(actual.shape))
        assertEquals(listOf(1.5, 1.6153846153846154, 1.0, 1.0, 0.5263157894736842, 1.2), actual.toList())
    }

    @Test
    fun mul_with_ndarrays() {
        // SampleStart
        val a = mk.ndarray(mk[mk[0.5, 0.8, 0.0], mk[0.0, -4.5, 1.0]])
        val b = mk.ndarray(mk[mk[1.0, 1.3, 3.0], mk[4.0, 9.5, 5.0]])
        a * b // multiplication
        /*
        [[0.5, 1.04, 0.0],
        [0.0, -42.75, 5.0]]
        */
        // SampleEnd

        val actual = a * b
        assertEquals(true, intArrayOf(2, 3).contentEquals(actual.shape))
        assertEquals(listOf(0.5, 1.04, 0.0, 0.0, -42.75, 5.0), actual.toList())
    }

    @Test
    fun inplace_arith_ops() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val b = mk.ndarray(mk[mk[4, 0], mk[7, 5]])

        a += b
        println(a)
        /*
        [[5, 2],
        [10, 9]]
         */

        a *= 3
        println(a)
        /*
        [[15, 6],
        [30, 27]]
         */
        // SampleEnd

        assertEquals(listOf(15, 6, 30, 27), a.toList())
    }
}