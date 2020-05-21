package org.jetbrains.multik.creation

import org.jetbrains.multik.api.d3array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.core.D3
import org.jetbrains.multik.core.Ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray3DTest {

    private val shape = intArrayOf(2, 5, 3)
    private val dim = 3

    @Test
    fun createByteArray3DTest() {
        val inputArray = ByteArray(30) { it.toByte() }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createShortArray3DTest() {
        val inputArray = ShortArray(30) { it.toShort() }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createIntArray3DTest() {
        val inputArray = IntArray(30) { it }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createLongArray3DTest() {
        val inputArray = LongArray(30) { it.toLong() }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createFloatArray3DTest() {
        val inputArray = FloatArray(30) { it.toFloat() }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createDoubleArray3DTest() {
        val inputArray = DoubleArray(30) { it.toDouble() }
        val a = mk.ndarray<D3>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(30) { it }
        val a = mk.d3array(2, 5, 3) { it }
        assertEquals(mk.ndarray(inputArr, shape), a)

        val b = mk.d3array(2, 5, 3) { it * it }
        assertEquals(mk.ndarray(inputArr.map { it * it }, shape), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3, 3)
        val l = Array(18) { it }.toList()
        val s = HashSet(l)
        val a: Ndarray<Int, D3> = mk.ndarray(s, shapeCol)
        val b: Ndarray<Int, D3> = mk.ndarray(l, shapeCol)
        assertEquals(a, b)
    }
}
