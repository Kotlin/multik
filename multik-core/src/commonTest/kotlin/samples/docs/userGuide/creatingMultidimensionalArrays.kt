package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.test.Test

class CreatingMultidimensionalArrays {
    @Test
    fun literal_construction() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        // [1, 2, 3]

        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        /*
        [[1.5, 2.1, 3.0],
         [4.0, 5.0, 6.0]]
         */
        // SampleEnd
    }

    @Test
    fun explicit_dimension_type() {
        // SampleStart
        val c: NDArray<Double, D2> = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        // SampleEnd
    }

    @Test
    fun from_collections() {
        // SampleStart
        val fromList = mk.ndarray(listOf(10, 20, 30))
        // [10, 20, 30]

        val fromIterable = listOf(1.2, 3.4, 5.6).toNDArray()
        // [1.2, 3.4, 5.6]
        // SampleEnd
    }

    @Test
    fun from_primitive_array() {
        // SampleStart
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val shaped = mk.ndarray(data, 2, 3)
        /*
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
         */
        // SampleEnd
    }

    @Test
    fun factory_functions() {
        // SampleStart
        val zeros = mk.zeros<Double>(2, 3)
        /*
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]
         */

        val ones = mk.ones<Int>(3)
        // [1, 1, 1]

        val eye = mk.identity<Double>(3)
        /*
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
         */

        val diag = mk.diagonal(mk[2, 4, 8])
        /*
        [[2, 0, 0],
         [0, 4, 0],
         [0, 0, 8]]
         */
        // SampleEnd
    }

    @Test
    fun range_and_linspace() {
        // SampleStart
        val arange = mk.arange<Int>(0, 10, 2)
        // [0, 2, 4, 6, 8]

        val lin = mk.linspace<Double>(0.0, 1.0, 5)
        // [0.0, 0.25, 0.5, 0.75, 1.0]
        // SampleEnd
    }

    @Test
    fun lambda_creation() {
        // SampleStart
        val squares = mk.d2array(3, 3) { it * it }
        /*
        [[0, 1, 4],
         [9, 16, 25],
         [36, 49, 64]]
         */

        val grid = mk.d2arrayIndices(2, 3) { i, j -> i * 10 + j }
        /*
        [[0, 1, 2],
         [10, 11, 12]]
         */
        // SampleEnd
    }

    @Test
    fun controlling_dtype() {
        // SampleStart
        val x = mk.zeros<Float>(2, 2) // dtype is Float

        val y: NDArray<Int, D2> = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        // SampleEnd
    }
}