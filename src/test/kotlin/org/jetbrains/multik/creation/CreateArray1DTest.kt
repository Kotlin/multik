package org.jetbrains.multik.creation

import org.jetbrains.multik.api.*
import org.jetbrains.multik.core.D1
import org.jetbrains.multik.core.DataType
import org.jetbrains.multik.core.Ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray1DTest {

    @Test
    fun createByteArrayTest() {
        val inputArr = ByteArray(10) { it.toByte() }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createShortArrayTest() {
        val inputArr = ByteArray(10) { it.toByte() }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createIntArrayTest() {
        val inputArr = IntArray(10) { it }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createLongArrayTest() {
        val inputArr = ByteArray(10) { it.toByte() }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createFloatArrayTest() {
        val inputArr = ByteArray(10) { it.toByte() }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createDoubleArrayTest() {
        val inputArr = ByteArray(10) { it.toByte() }
        val a = mk.ndarray<D1>(inputArr, shape = intArrayOf(10))

        // check dimension
        assertEquals(1, a.dim.d)

        //check data
        assertEquals(inputArr.asList(), a.getData())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(10) { it }
        val a = mk.d1array(10) { it }
        assertEquals(mk.ndarray(inputArr, intArrayOf(10)), a)

        val b = mk.d1array(10) { it * it }
        assertEquals(mk.ndarray(inputArr.map { it * it }.toIntArray(), intArrayOf(10)), b)
    }

    @Test
    fun createSimpleArray1DTest() {
        val inputArr = FloatArray(7) { it.toFloat() }
        inputArr[inputArr.lastIndex] = 33f
        val a = mk.ndarrayOf(1f, 2f, 3f, 4f, 5f, 6f, 33f)
        assertEquals(mk.ndarray(inputArr, intArrayOf(7)), a)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val a: Ndarray<Int, D1> = mk.ndarray(setOf(1, 2, 3), intArrayOf(3))
        val b: Ndarray<Int, D1> = mk.ndarray(arrayListOf(1, 2, 3), intArrayOf(3))
        assertEquals(a, b)
    }

    @Test
    fun createArrayFromIntRangeTest() {
        val a = (0..9).toNdarray()
        val b = mk.d1array(10) { it }
        assertEquals(a, b)
    }

    @Test
    fun createArrayRangeTest() {
        val testArray1 = mk.arange<Int>(10, 2)
        assertEquals(mk.ndarrayOf(0, 2, 4, 6, 8), testArray1)

        val testArray2 = mk.arange<Double>(10, 1.5)
        assertEquals(mk.ndarrayOf(0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0), testArray2)

        val testArray3 = mk.arange<Int>(5, 20, 3.5)
        assertEquals(DataType.IntDataType, testArray3.dtype)
        assertEquals(mk.ndarrayOf(5, 8, 12, 15, 19), testArray3)

        val testArray4 = mk.arange<Float>(5, 21, 4)
        assertEquals(DataType.FloatDataType, testArray4.dtype)
        assertEquals(mk.ndarrayOf(5f, 9f, 13f, 17f), testArray4)
    }

    @Test
    fun createArrayLinspaceTest() {
        val a = mk.linspace(0, 33, num = 5)
        assertEquals(mk.ndarrayOf(0.0, 8.25, 16.5, 24.75, 33.0), a)
    }
}
