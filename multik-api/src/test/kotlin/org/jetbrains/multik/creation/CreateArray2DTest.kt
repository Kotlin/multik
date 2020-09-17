package org.jetbrains.multik.creation

import org.jetbrains.multik.api.d2array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.ndarray.data.D2
import org.jetbrains.multik.ndarray.data.Ndarray
import org.jetbrains.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray2DTest {

    private val dim = 2

    @Test
    fun createByteArray2DTest() {
        val inputArray = ByteArray(10) { it.toByte() }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createShortArray2DTest() {
        val inputArray = ShortArray(10) { it.toShort() }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createIntArray2DTest() {
        val inputArray = IntArray(10) { it }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createLongArray2DTest() {
        val inputArray = LongArray(10) { it.toLong() }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createFloatArray2DTest() {
        val inputArray = FloatArray(10) { it.toFloat() }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDoubleArray2DTest() {
        val inputArray = DoubleArray(10) { it.toDouble() }
        val a = mk.ndarray(inputArray, 2, 5)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(10) { it }
        val a = mk.d2array(dim, 5) { it }
        assertEquals(mk.ndarray(inputArr, 2, 5), a)

        val b = mk.d2array(2, 5) { it * it }
        assertEquals(mk.ndarray(inputArr.map { it * it }.toIntArray(), 2, 5), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3)
        val a: Ndarray<Int, D2> = mk.ndarray(setOf(0, 2, 3, 32, 5, 7), shapeCol)
        val b: Ndarray<Int, D2> = mk.ndarray(arrayListOf(0, 2, 3, 32, 5, 7), shapeCol)
        assertEquals(a, b)
    }
}
