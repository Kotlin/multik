package samples.docs

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class ArrayCreation {

    @Test
    fun simple_way_of_creation() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        println(a.dtype)
        // DataType(nativeCode=3, itemSize=4, class=class kotlin.Int)
        println(a.dim)
        // dimension: 1
        println(a)
        // [1, 2, 3]

        val b = mk.ndarray(mk[mk[1.5, 5.8], mk[9.1, 7.3]])
        println(b.dtype)
        // DataType(nativeCode=6, itemSize=8, class=class kotlin.Double)
        println(b.dim)
        // dimension: 2
        println(b)
        /*
         [[1.5, 5.8],
         [9.1, 7.3]]
         */
        // SampleEnd
        assertEquals(DataType.IntDataType, a.dtype)
        assertEquals(1, a.dim.d)
        assertEquals(DataType.DoubleDataType, b.dtype)
        assertEquals(2, b.dim.d)
    }

    @Test
    fun create_array_from_collections() {
        // SampleStart
        mk.ndarray(setOf(1, 2, 3)) // [1, 2, 3]
        listOf(8.4, 5.2, 9.3, 11.5).toNDArray() // [8.4, 5.2, 9.3, 11.5]
        // SampleEnd
        assertEquals(listOf(1, 2, 3), mk.ndarray(setOf(1, 2, 3)).toList())
        assertEquals(listOf(8.4, 5.2, 9.3, 11.5), listOf(8.4, 5.2, 9.3, 11.5).toNDArray().toList())
    }

    @Test
    fun create_array_from_primitive_with_shape() {
        // SampleStart
        mk.ndarray(floatArrayOf(34.2f, 13.4f, 4.8f, 8.8f, 3.3f, 7.1f), 2, 1, 3)
        /*
        [[[34.2, 13.4, 4.8]],

        [[8.8, 3.3, 7.1]]]
         */
        // SampleEnd

        val a = mk.ndarray(floatArrayOf(34.2f, 13.4f, 4.8f, 8.8f, 3.3f, 7.1f), 2, 1, 3)
        assertEquals(listOf(2, 1, 3), a.shape.toList())
        assertEquals(listOf(34.2f, 13.4f, 4.8f, 8.8f, 3.3f, 7.1f), a.toList())
    }

    @Test
    fun create_zeros_and_ones_arrays() {
        // SampleStart
        mk.zeros<Int>(7)
        // [0, 0, 0, 0, 0, 0, 0]

        mk.ones<Float>(3, 2)
        /*
        [[1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]]
         */
        // SampleEnd
        assertEquals(listOf(0, 0, 0, 0, 0, 0, 0), mk.zeros<Int>(7).toList())
        assertEquals(listOf(1f, 1f, 1f, 1f, 1f, 1f), mk.ones<Float>(3, 2).toList())
    }

    @Test
    fun creation_with_lambda() {
        // SampleStart
        mk.d3array(2, 2, 3) { it * it } // create an array of dimension 3
        /*
        [[[0, 1, 4],
        [9, 16, 25]],

        [[36, 49, 64],
        [81, 100, 121]]]
        */

        mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
        /*
        [[0.0+(0.0)i, 0.0+(1.0)i, 0.0+(2.0)i],
        [1.0+(0.0)i, 1.0+(1.0)i, 1.0+(2.0)i],
        [2.0+(0.0)i, 2.0+(1.0)i, 2.0+(2.0)i]]
         */
        // SampleEnd
        assertEquals(
            listOf(0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121),
            mk.d3array(2, 2, 3) { it * it }.toList()
        )

        assertEquals(listOf(
            0f + 0f.i, 0f + 1f.i, 0f + 2f.i,
            1f + 0f.i, 1f + 1f.i, 1f + 2f.i,
            2f + 0f.i, 2f + 1f.i, 2f + 2f.i
        ),
            mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }.toList()
        )
    }

    @Test
    fun creation_with_arange_and_linspace() {
        // SampleStart
        mk.arange<Int>(3, 10, 2)
        // [3, 5, 7, 9]
        mk.linspace<Double>(0.0, 10.0, 8)
        // [0.0, 1.4285714285714286, 2.857142857142857, 4.285714285714286, 5.714285714285714, 7.142857142857143, 8.571428571428571, 10.0]
        // SampleEnd

        assertEquals(4, mk.arange<Int>(3, 10, 2).size)
        assertEquals(8, mk.linspace<Double>(0.0, 10.0, 8).size)

    }
}