package org.jetbrains.multik.creation

import org.jetbrains.multik.api.d4array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.core.D4
import org.jetbrains.multik.core.Ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray4DTest {
    private val shape = intArrayOf(2, 5, 3, 2)
    private val dim = 4

    @Test
    fun createByteArray4DTest() {
        val inputArray = ByteArray(60) { it.toByte() }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createShortArray4DTest() {
        val inputArray = ShortArray(60) { it.toShort() }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createIntArray4DTest() {
        val inputArray = IntArray(60) { it }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createLongArray4DTest() {
        val inputArray = LongArray(60) { it.toLong() }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createFloatArray4DTest() {
        val inputArray = FloatArray(60) { it.toFloat() }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createDoubleArray4DTest() {
        val inputArray = DoubleArray(60) { it.toDouble() }
        val a = mk.ndarray<D4>(inputArray, shape)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.getData())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(60) { it }
        val a = mk.d4array(2, 5, 3, 2) { it }
        assertEquals(mk.ndarray(inputArr, shape), a)

        val b = mk.d4array(2, 5, 3, 2) { it * it }
        assertEquals(mk.ndarray(inputArr.map { it * it }, shape), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3, 3, 2)
        val l = Array(36) { it }.toList()
        val s = HashSet(l)
        val a: Ndarray<Int, D4> = mk.ndarray(s, shapeCol)
        val b: Ndarray<Int, D4> = mk.ndarray(l, shapeCol)
        assertEquals(a, b)
    }
}