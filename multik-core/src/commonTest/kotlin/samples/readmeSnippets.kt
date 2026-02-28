package samples

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.diagonal
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linspace
import org.jetbrains.kotlinx.multik.api.math.cos
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.api.math.log
import org.jetbrains.kotlinx.multik.api.math.sin
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.rangeTo
import org.jetbrains.kotlinx.multik.ndarray.operations.abs
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.filter
import org.jetbrains.kotlinx.multik.ndarray.operations.groupNDArrayBy
import org.jetbrains.kotlinx.multik.ndarray.operations.inplace
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.math
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.sorted
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class ReadmeSnippets {

    private val a = mk.ndarray(mk[1, 2, 3])
    private val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
    private val c = mk.ndarray(mk[mk[mk[1.5f, 2f, 3f], mk[4f, 5f, 6f]], mk[mk[3f, 2f, 1f], mk[4f, 5f, 6f]]])
    private val d = mk.ndarray(doubleArrayOf(1.0, 1.3, 3.0, 4.0, 9.5, 5.0), 2, 3)
    private val e = mk.identity<Double>(3)
    private val diag = mk.diagonal(mk[2, 4, 8])


    @Test
    fun creatingArrays() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        /* [1, 2, 3] */

        val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
        /*
        [[1.5, 2.1, 3.0],
        [4.0, 5.0, 6.0]]
        */

        val c = mk.ndarray(mk[mk[mk[1.5f, 2f, 3f], mk[4f, 5f, 6f]], mk[mk[3f, 2f, 1f], mk[4f, 5f, 6f]]])
        /*
        [[[1.5, 2.0, 3.0],
        [4.0, 5.0, 6.0]],

        [[3.0, 2.0, 1.0],
        [4.0, 5.0, 6.0]]]
        */


        mk.zeros<Double>(3, 4) // create an array of zeros
        /*
        [[0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]]
        */
        mk.ndarray<Float, D2>(setOf(30f, 2f, 13f, 12f), intArrayOf(2, 2)) // create an array from a collection
        /*
        [[30.0, 2.0],
        [13.0, 12.0]]
        */

        val d = mk.ndarray(
            doubleArrayOf(1.0, 1.3, 3.0, 4.0, 9.5, 5.0),
            2, 3
        ) // create an array of shape(2, 3) from a primitive array
        /*
        [[1.0, 1.3, 3.0],
        [4.0, 9.5, 5.0]]
        */

        mk.d3array(2, 2, 3) { it * it } // create an array of 3 dimension
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

        mk.arange<Long>(10, 25, 5) // create an array with elements in the interval [10, 25) with step 5
        /* [10, 15, 20] */

        mk.linspace<Double>(0, 2, 9) // create an array of 9 elements in the interval [0, 2]
        /* [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] */

        val e = mk.identity<Double>(3) // create an identity array of shape (3, 3)
        /*
        [[1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]
        */

        val diag = mk.diagonal(mk[2, 4, 8]) // create a diagonal array
        /*
        [[2, 0, 0],
        [0, 4, 0],
        [0, 0, 8]]
         */
        // SampleEnd

        assertEquals("[1, 2, 3]", a.toString())

        val bStr = """
            [[1.5, 2.1, 3.0],
            [4.0, 5.0, 6.0]]
        """.trimIndent()
        assertEquals(bStr, b.toString())

        val cStr = """
            [[[1.5, 2.0, 3.0],
            [4.0, 5.0, 6.0]],

            [[3.0, 2.0, 1.0],
            [4.0, 5.0, 6.0]]]
        """.trimIndent()
        assertEquals(cStr, c.toString())
        assertEquals(true, mk.zeros<Double>(3, 4).all { it == 0.0 })
        assertContentEquals(intArrayOf(2, 3), d.shape)
        assertEquals(D3, mk.d3array(2, 2, 3) { it * it }.dim)
        assertEquals(9, mk.linspace<Double>(0, 2, 9).size)
        assertEquals(listOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), e.toList())
        assertEquals(listOf(2, 0, 0, 0, 4, 0, 0, 0, 8), diag.toList())
    }

    @Test
    fun arrayProperties() {
        // SampleStart
        a.shape // Array dimensions
        a.size // Size of array
        a.dim // object Dimension
        a.dim.d // number of array dimensions
        a.dtype // Data type of array elements
        // SampleEnd
        assertContentEquals(intArrayOf(3), a.shape)
        assertEquals(3, a.size)
        assertEquals(D1, a.dim)
        assertEquals(1, a.dim.d)
        assertEquals(DataType.IntDataType, a.dtype)
    }

    @Test
    fun arithmeticOperations() {
        // SampleStart
        val f = b - d // subtraction
        /*
        [[0.5, 0.8, 0.0],
        [0.0, -4.5, 1.0]]
        */

        d + f // addition
        /*
        [[1.5, 2.1, 3.0],
        [4.0, 5.0, 6.0]]
        */

        b / d // division
        /*
        [[1.5, 1.6153846153846154, 1.0],
        [1.0, 0.5263157894736842, 1.2]]
        */

        f * d // multiplication
        /*
        [[0.5, 1.04, 0.0],
        [0.0, -42.75, 5.0]]
        */
        // SampleEnd
        val sub = """
            [[0.5, 0.8, 0.0],
            [0.0, -4.5, 1.0]]
        """.trimIndent()
        val add = """
            [[1.5, 2.1, 3.0],
            [4.0, 5.0, 6.0]]
        """.trimIndent()
        val div = """
            [[1.5, 1.6153846153846154, 1.0],
            [1.0, 0.5263157894736842, 1.2]]
        """.trimIndent()
        val mul = """
            [[0.5, 1.04, 0.0],
            [0.0, -42.75, 5.0]]
        """.trimIndent()

        assertEquals(sub, f.toString())
        assertEquals(add, (d + f).toString())
        assertEquals(div, (b / d).toString())
        assertEquals(mul, (f * d).toString())
    }

    @Test
    @Ignore
    fun mathFunctions() {
        // SampleStart
        a.sin() // element-wise sin, equivalent to mk.math.sin(a)
        a.cos() // element-wise cos, equivalent to mk.math.cos(a)
        b.log() // element-wise natural logarithm, equivalent to mk.math.log(b)
        b.exp() // element-wise exp, equivalent to mk.math.exp(b)
        d dot e // dot product, equivalent to mk.linalg.dot(d, e)

        mk.math.sum(c) // array-wise sum
        mk.math.min(c) // array-wise minimum elements
        mk.math.maxD3(c, axis = 0) // maximum value of an array along axis 0
        mk.math.cumSum(b, axis = 1) // cumulative sum of the elements
        mk.stat.mean(a) // mean
        mk.stat.median(b) // median
        // SampleEnd
    }

    @Test
    fun copyingArrays() {
        // SampleStart
        val f = a.copy() // create a copy of the array and its data
        val h = b.deepCopy() // create a copy of the array and copy the meaningful data
        // SampleEnd
        assertEquals(a.toString(), f.toString())
        assertEquals(b.toString(), h.toString())
    }

    @Test
    fun collectionOperations() {
        // SampleStart
        c.filter { it < 3 } // select all elements less than 3
        b.map { (it * it).toInt() } // return squares
        c.groupNDArrayBy { it % 2 } // group elements by condition
        c.sorted() // sort elements
        // SampleEnd
        assertEquals("[1.5, 2.0, 2.0, 1.0]", c.filter { it < 3 }.toString())
        assertEquals(listOf(2, 4, 9, 16, 25, 36), b.map { (it * it).toInt() }.toList())
        assertEquals(3, c.groupNDArrayBy { it % 2 }.keys.size)
        assertEquals(listOf(1f, 1.5f, 2f, 2f, 3f, 3f, 4f, 4f, 5f, 5f, 6f, 6f), c.sorted().toList())
    }

    @Test
    fun indexingSlicingIterating() {
        // SampleStart
        a[2] // select the element at the 2 index
        b[1, 2] // select the element at row 1 column 2
        b[1] // select row 1
        b[0..1, 1] // select elements at rows 0 to 1 in column 1
        b[0, 0..2..1] // select elements at row 0 in columns 0 to 2 with step 1

        for (el in b) {
            print("$el, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
        }

        // for n-dimensional
        val q = b.asDNArray()
        for (index in q.multiIndices) {
            print("${q[index]}, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
        }
        // SampleEnd
    }

    @Test
    fun inplaceOperations() {
        // SampleStart
        val a = mk.linspace<Float>(0, 1, 10)
        /*
        a = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556,
        0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0]
        */
        val b = mk.linspace<Float>(8, 9, 10)
        /*
        b = [8.0, 8.11111111111111, 8.222222222222221, 8.333333333333334, 8.444444444444445, 8.555555555555555,
        8.666666666666666, 8.777777777777779, 8.88888888888889, 9.0]
        */

        a.inplace {
            math {
                (this - b) * b
                abs()
            }
        }
        // a = [64.0, 64.88888, 65.77778, 66.66666, 67.55556, 68.44444, 69.333336, 70.22222, 71.111115, 72.0]
        // SampleEnd
        assertEquals(
            listOf(64f, 64.88888f, 65.77778f, 66.66666f, 67.55556f, 68.44444f, 69.333336f, 70.22222f, 71.111115f, 72f),
            a.toList()
        )
    }
}
