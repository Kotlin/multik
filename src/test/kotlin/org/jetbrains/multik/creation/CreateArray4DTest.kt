package org.jetbrains.multik.creation

import org.jetbrains.multik.api.d4array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.api.ndarray
import org.jetbrains.multik.core.D4
import org.jetbrains.multik.core.Ndarray
import org.jetbrains.multik.core.toList
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class CreateArray4DTest {

    @BeforeTest
    fun loadLibrary() {
        System.load("/Users/pavel.gorgulov/Projects/main_project/multik/src/jni_multik/cmake-build-debug/libjni_multik.dylib")
    }

    private val shape = intArrayOf(2, 5, 3, 2)
    private val dim = 4

    @Test
    fun createByteArray4DTest() {
        val inputArray = ByteArray(60) { it.toByte() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createShortArray4DTest() {
        val inputArray = ShortArray(60) { it.toShort() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createIntArray4DTest() {
        val inputArray = IntArray(60) { it }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createLongArray4DTest() {
        val inputArray = LongArray(60) { it.toLong() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createFloatArray4DTest() {
        val inputArray = FloatArray(60) { it.toFloat() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDoubleArray4DTest() {
        val inputArray = DoubleArray(60) { it.toDouble() }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)

        assertEquals(dim, a.dim.d)

        assertEquals(inputArray.asList(), a.toList())
    }

    @Test
    fun createDslArrayTest() {
        val inputArr = IntArray(60) { it }
        val a = mk.d4array(2, 5, 3, 2) { it }
        assertEquals(mk.ndarray(inputArr, 2, 5, 3, 2), a)

        val b = mk.d4array(2, 5, 3, 2) { it * it }
        assertEquals(mk.ndarray<Int, D4>(inputArr.map { it * it }, shape), b)
    }

    @Test
    fun createArrayFromCollectionTest() {
        val shapeCol = intArrayOf(2, 3, 3, 2)
        val l = Array(36) { it }.toList()
        val size_ = HashSet(l)
        val a: Ndarray<Int, D4> = mk.ndarray(size_, shapeCol)
        val b: Ndarray<Int, D4> = mk.ndarray(l, shapeCol)
        assertEquals(a, b)
    }
}