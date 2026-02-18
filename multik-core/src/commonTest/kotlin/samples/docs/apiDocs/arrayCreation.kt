package samples.docs.apiDocs

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import kotlin.test.Test

class ArrayCreation {
    @Test
    fun arange_example() {
        // SampleStart
        val a = mk.arange<Int>(5)              // [0, 1, 2, 3, 4]
        val b = mk.arange<Int>(2, 10, 3)       // [2, 5, 8]
        val c = mk.arange<Double>(0, 1, 0.3)   // [0.0, 0.3, 0.6, 0.9]
        // SampleEnd
    }

    @Test
    fun d1array_example() {
        // SampleStart
        val a = mk.d1array<Int>(5) { it * 2 }
        // [0, 2, 4, 6, 8]

        val b = mk.d1array<Double>(3) { it + 0.5 }
        // [0.5, 1.5, 2.5]
        // SampleEnd
    }

    @Test
    fun d2array_example() {
        // SampleStart
        val m = mk.d2array<Int>(2, 3) { it }
        // [[0, 1, 2],
        //  [3, 4, 5]]

        val identity = mk.d2array<Double>(3, 3) { if (it / 3 == it % 3) 1.0 else 0.0 }
        // SampleEnd
    }

    @Test
    fun d2arrayIndices_example() {
        // SampleStart
        val m = mk.d2arrayIndices<Int>(3, 3) { i, j -> i + j }
        // [[0, 1, 2],
        //  [1, 2, 3],
        //  [2, 3, 4]]

        val identity = mk.d2arrayIndices<Double>(4, 4) { i, j ->
            if (i == j) 1.0 else 0.0
        }
        // SampleEnd
    }

    @Test
    fun d3array_example() {
        // SampleStart
        val t = mk.d3array<Int>(2, 3, 4) { it }
        // shape (2, 3, 4), elements 0..23
        // SampleEnd
    }

    @Test
    fun d3arrayIndices_example() {
        // SampleStart
        val t = mk.d3arrayIndices<Int>(2, 3, 4) { i, j, k -> i * 100 + j * 10 + k }
        // t[1, 2, 3] == 123
        // SampleEnd
    }

    @Test
    fun d4array_example() {
        // SampleStart
        val a = mk.d4array<Double>(2, 3, 4, 5) { it.toDouble() }
        // shape (2, 3, 4, 5), 120 elements
        // SampleEnd
    }

    @Test
    fun d4arrayIndices_example() {
        // SampleStart
        val a = mk.d4arrayIndices<Int>(2, 2, 2, 2) { i, j, k, m ->
            i * 8 + j * 4 + k * 2 + m
        }
        // a[1, 1, 1, 1] == 15
        // SampleEnd
    }

    @Test
    fun dnarray_example() {
        // SampleStart
        // 5D array
        val a = mk.dnarray<Int>(2, 3, 4, 5, 6) { it }
        // shape (2, 3, 4, 5, 6), 720 elements

        // Using shape array with reified dimension
        val b = mk.dnarray<Double, D3>(intArrayOf(2, 3, 4)) { it.toDouble() }
        // shape (2, 3, 4)
        // SampleEnd
    }

    @Test
    fun identity_example() {
        // SampleStart
        val I = mk.identity<Double>(3)
        // [[1.0, 0.0, 0.0],
        //  [0.0, 1.0, 0.0],
        //  [0.0, 0.0, 1.0]]

        val intI = mk.identity<Int>(2, DataType.IntDataType)
        // [[1, 0],
        //  [0, 1]]
        // SampleEnd
    }

    @Test
    fun linspace_example() {
        // SampleStart
        val a = mk.linspace<Double>(0, 1, 5)
        // [0.0, 0.25, 0.5, 0.75, 1.0]

        val b = mk.linspace<Float>(0.0, 10.0, 3)
        // [0.0, 5.0, 10.0]
        // SampleEnd
    }

    @Test
    fun meshgrid_example() {
        // SampleStart
        val x = mk.ndarray(mk[1, 2, 3])
        val y = mk.ndarray(mk[4, 5])

        val (X, Y) = mk.meshgrid(x, y)

        // X:
        // [[1, 2, 3],
        //  [1, 2, 3]]

        // Y:
        // [[4, 4, 4],
        //  [5, 5, 5]]
        // SampleEnd
    }

    @Test
    fun meshgrid_grid_function_example() {
        // SampleStart
        val x = mk.linspace<Double>(0.0, 1.0, 10)
        val y = mk.linspace<Double>(0.0, 1.0, 10)
        val (X, Y) = mk.meshgrid(x, y)
        // Compute f(x, y) = x^2 + y^2 element-wise on the grid
        // SampleEnd
    }

    @Test
    fun ndarray_from_mk_list_example() {
        // SampleStart
        val v = mk.ndarray(mk[1, 2, 3])                         // D1Array<Int>
        val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])     // D2Array<Double>
        val t = mk.ndarray(mk[mk[mk[1, 2], mk[3, 4]]])         // D3Array<Int>
        // SampleEnd
    }

    @Test
    fun ndarray_from_collection_shape_example() {
        // SampleStart
        val a = mk.ndarray<Int, D2>(listOf(1, 2, 3, 4), intArrayOf(2, 2))
        // D2Array<Int>, shape (2, 2)
        // SampleEnd
    }

    @Test
    fun ndarray_from_collection_explicit_dim_example() {
        // SampleStart
        val b = mk.ndarray(listOf(1, 2, 3, 4), intArrayOf(2, 2), D2)
        // SampleEnd
    }

    @Test
    fun ndarray_from_collection_dims_example() {
        // SampleStart
        val m = mk.ndarray(listOf(1, 2, 3, 4, 5, 6), 2, 3)     // D2Array, shape (2, 3)
        val t = mk.ndarray(listOf(1, 2, 3, 4, 5, 6), 1, 2, 3)  // D3Array, shape (1, 2, 3)
        // SampleEnd
    }

    @Test
    fun ndarray_from_primitive_1d_example() {
        // SampleStart
        val a = mk.ndarray(intArrayOf(1, 2, 3))         // D1Array<Int>
        val b = mk.ndarray(doubleArrayOf(1.0, 2.0))     // D1Array<Double>
        // SampleEnd
    }

    @Test
    fun ndarray_from_primitive_shape_example() {
        // SampleStart
        val m = mk.ndarray(intArrayOf(1, 2, 3, 4), 2, 2)  // D2Array<Int>
        // SampleEnd
    }

    @Test
    fun ndarray_from_array_of_arrays_example() {
        // SampleStart
        val m = mk.ndarray(arrayOf(intArrayOf(1, 2), intArrayOf(3, 4)))  // D2Array<Int>
        // SampleEnd
    }

    @Test
    fun ndarrayOf_example() {
        // SampleStart
        val a = mk.ndarrayOf(1, 2, 3)          // D1Array<Int>: [1, 2, 3]
        val b = mk.ndarrayOf(1.0, 2.0, 3.0)   // D1Array<Double>: [1.0, 2.0, 3.0]
        // SampleEnd
    }

    @Test
    fun ones_example() {
        // SampleStart
        val v = mk.ones<Double>(5)           // [1.0, 1.0, 1.0, 1.0, 1.0]
        val m = mk.ones<Int>(2, 3)           // [[1, 1, 1], [1, 1, 1]]
        val t = mk.ones<Float>(2, 3, 4)     // shape (2, 3, 4), all 1.0f
        // SampleEnd
    }

    @Test
    fun rand_example() {
        // SampleStart
        val a = mk.rand<Double>(3, 3)                     // 3×3, values in [0.0, 1.0)
        val b = mk.rand<Int, D1>(from = 1, until = 100, 10)  // 10 ints in [1, 100)
        val c = mk.rand<Double, D2>(42, 0.0, 1.0, 3, 3)  // seeded 3×3
        // SampleEnd
    }

    @Test
    fun zeros_example() {
        // SampleStart
        val v = mk.zeros<Double>(5)           // [0.0, 0.0, 0.0, 0.0, 0.0]
        val m = mk.zeros<Int>(2, 3)           // [[0, 0, 0], [0, 0, 0]]
        val t = mk.zeros<Float>(2, 3, 4)     // shape (2, 3, 4), all 0.0f
        // SampleEnd
    }

    @Test
    fun toNDArray_iterable_example() {
        // SampleStart
        val a = listOf(1, 2, 3).toNDArray()          // D1Array<Int>
        val b = setOf(1.0, 2.0).toNDArray()          // D1Array<Double>
        // SampleEnd
    }

    @Test
    fun toNDArray_nested_list_example() {
        // SampleStart
        val m = listOf(listOf(1, 2), listOf(3, 4)).toNDArray()  // D2Array<Int>
        // SampleEnd
    }

    @Test
    fun toNDArray_array_of_arrays_example() {
        // SampleStart
        val m = arrayOf(intArrayOf(1, 2), intArrayOf(3, 4)).toNDArray()  // D2Array<Int>
        // SampleEnd
    }
}