/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.random.Random
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CreateNDArrayTests {

    private val random = Random(42)

    /*___________________________Byte_______________________________________*/

    /**
     * This method checks if a byte array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledByteArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Byte>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0.toByte() } }
    }

    /**
     * Creates a byte array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createByteArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Byte>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1.toByte() } }
    }


    /**
     * Creates an n-dimensional array from a set of bytes
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromByteSet() {
        val set = (1..36).map { it.toByte() }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Byte, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive ByteArray
     * and checks if the array's ByteArray representation matches the input ByteArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveByteArray() {
        val array = ByteArray(36) { random.nextInt(101).toByte() }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getByteArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createByteNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<Byte>(2, 3, 1, 2) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByteNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Byte>(2, 2, 1, 2, 2) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /*___________________________Short_______________________________________*/

    /**
     * This method checks if a short array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledShortArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Short>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0.toShort() } }
    }

    /**
     * Creates a short array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createShortArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Short>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1.toShort() } }
    }

    /**
     * Creates an n-dimensional array from a set of shorts
     * and checks if the array's set representation matches the input set.
     */
    @Test
    @Ignore
    fun createNDimensionalArrayFromShortSet() {
        val set = (1..36).map { it.toShort() }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Short, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive ShortArray
     * and checks if the array's ShortArray representation matches the input ShortArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveShortArray() {
        val array = ShortArray(36) { random.nextInt(101).toShort() }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getShortArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShortNDArrayWithInitializationFunctionWith4D() {
        val a = mk.d4array<Short>(2, 3, 1, 2) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShortNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Short>(2, 2, 1, 2, 2) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /*___________________________Int_______________________________________*/


    /**
     * This method checks if an integer array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledIntArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Int>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0 } }
    }

    /**
     * Creates an integer array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createIntArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Int>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1 } }
    }


    /**
     * Creates an n-dimensional array from a set of integers
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromIntSet() {
        val set = (1..36).toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Int, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive IntArray
     * and checks if the array's IntArray representation matches the input IntArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveIntArray() {
        val array = IntArray(36) { random.nextInt(101) }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getIntArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createIntNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<Int>(2, 3, 1, 2) { (it + 3) }
        val expected = intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createIntNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Int>(2, 2, 1, 2, 2) { it + 3 }
        val expected = intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /*___________________________Long_______________________________________*/


    /**
     * This method checks if a long array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledLongArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Long>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0L } }
    }

    /**
     * Creates a long array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createLongArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Long>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1L } }
    }

    /**
     * Creates an n-dimensional array from a set of longs
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromLongSet() {
        val set = (1..36).map { it.toLong() }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Long, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive LongArray
     * and checks if the array's LongArray representation matches the input LongArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveLongArray() {
        val array = LongArray(36) { random.nextLong(101) }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getLongArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createLongNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<Long>(2, 3, 1, 2) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLongNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Long>(2, 2, 1, 2, 2) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /*___________________________Float_______________________________________*/

    /**
     * This method checks if a float array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledFloatArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Float>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0f } }
    }

    /**
     * Creates a float array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createFloatArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Float>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1f } }
    }

    /**
     * Creates an n-dimensional array from a set of floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromFloatSet() {
        val set = (1..36).map { it.toFloat() }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Float, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive FloatArray
     * and checks if the array's FloatArray representation matches the input FloatArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveFloatArray() {
        val array = FloatArray(36) { random.nextFloat() }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getFloatArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createFloatNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<Float>(2, 3, 1, 2) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloatNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Float>(2, 2, 1, 2, 2) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /*___________________________Double_______________________________________*/


    /**
     * This method checks if a double array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledDoubleArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<Double>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 0.0 } }
    }

    /**
     * Creates a double array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createDoubleArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<Double>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == 1.0 } }
    }

    /**
     * Creates an n-dimensional array from a set of doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromDoubleSet() {
        val set = (1..36).map { it.toDouble() }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<Double, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive DoubleArray
     * and checks if the array's DoubleArray representation matches the input DoubleArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveDoubleArray() {
        val array = DoubleArray(36) { random.nextDouble(101.0) }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getDoubleArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createDoubleNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<Double>(2, 3, 1, 2) { it + 3.0 }
        val expected = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDoubleNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<Double>(2, 2, 1, 2, 2) { it + 3.0 }
        val expected =
            doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }


    /*___________________________ComplexFloat_______________________________________*/

    /**
     * This method checks if a ComplexFloat array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexFloatArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<ComplexFloat>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == ComplexFloat.zero } }
    }

    /**
     * Creates a ComplexFloat array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexFloatArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<ComplexFloat>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == ComplexFloat.one } }
    }

    /**
     * Creates an n-dimensional array from a set of complex floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromComplexFloatSet() {
        val set = (1..36).map { ComplexFloat(it, it + 1) }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<ComplexFloat, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive ComplexFloatArray
     * and checks if the array's ComplexFloatArray representation matches the input ComplexFloatArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveComplexFloatArray() {
        val array = ComplexFloatArray(36) { ComplexFloat(random.nextFloat(), random.nextFloat()) }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getComplexFloatArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createComplexFloatNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<ComplexFloat>(2, 3, 1, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
        val expected = complexFloatArrayOf(
            3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i,
            6.21f + 2.168f.i, 7.21f + 3.168f.i, 8.21f + 4.168f.i,
            9.21f + 5.168f.i, 10.21f + 6.168f.i, 11.21f + 7.168f.i,
            12.21f + 8.168f.i, 13.21f + 9.168f.i, 14.21f + 10.168f.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloatNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<ComplexFloat>(2, 2, 1, 2, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
        val expected = complexFloatArrayOf(
            3.21f - 0.832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i, 6.21f + 2.168f.i,
            7.21f + 3.168f.i, 8.21f + 4.168f.i, 9.21f + 5.168f.i, 10.21f + 6.168f.i,
            11.21f + 7.168f.i, 12.21f + 8.168f.i, 13.21f + 9.168f.i, 14.21f + 10.168f.i,
            15.21f + 11.168f.i, 16.21f + 12.168f.i, 17.21f + 13.168f.i, 18.21f + 14.168f.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }


    /*___________________________ComplexDouble_______________________________________*/

    /**
     * This method checks if a ComplexDouble array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexDoubleArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.zeros<ComplexDouble>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == ComplexDouble.zero } }
    }

    /**
     * Creates a ComplexDouble array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexDoubleArrayFilledWithOnes() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val dim4 = 2
        val dim5 = 3
        val a = mk.ones<ComplexDouble>(dim1, dim2, dim3, dim4, dim5)

        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == ComplexDouble.one } }
    }

    /**
     * Creates an n-dimensional array from a set of complex doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createNDimensionalArrayFromComplexDoubleSet() {
        val set = (1..36).map { ComplexDouble(it, it + 1) }.toSet()
        val shape = intArrayOf(2, 3, 1, 2, 3)
        val a = mk.ndarray<ComplexDouble, DN>(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates an n-dimensional array from a primitive ComplexDoubleArray
     * and checks if the array's ComplexDoubleArray representation matches the input ComplexDoubleArray.
     */
    @Test
    fun createNDimensionalArrayFromPrimitiveComplexDoubleArray() {
        val array = ComplexDoubleArray(36) { ComplexDouble(random.nextDouble(101.0), random.nextDouble(10.0)) }
        val a = mk.ndarray(array, 2, 3, 1, 2, 3)

        assertEquals(array.size, a.size)
        a.data.getComplexDoubleArray() shouldBe array
    }


    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    @Ignore
    fun createComplexDoubleNDArrayWithInitializationFunctionWith4D() {
        val a = mk.dnarray<ComplexDouble>(2, 3, 1, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
        val expected = complexDoubleArrayOf(
            3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i,
            6.21 + 2.168.i, 7.21 + 3.168.i, 8.21 + 4.168.i,
            9.21 + 5.168.i, 10.21 + 6.168.i, 11.21 + 7.168.i,
            12.21 + 8.168.i, 13.21 + 9.168.i, 14.21 + 10.168.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }

    /**
     * Creates an n-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDoubleNDArrayWithInitializationFunctionWith5D() {
        val a = mk.dnarray<ComplexDouble>(2, 2, 1, 2, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
        val expected = complexDoubleArrayOf(
            3.21 - 0.832.i, 4.21 + 0.168.i, 5.21 + 1.168.i, 6.21 + 2.168.i,
            7.21 + 3.168.i, 8.21 + 4.168.i, 9.21 + 5.168.i, 10.21 + 6.168.i,
            11.21 + 7.168.i, 12.21 + 8.168.i, 13.21 + 9.168.i, 14.21 + 10.168.i,
            15.21 + 11.168.i, 16.21 + 12.168.i, 17.21 + 13.168.i, 18.21 + 14.168.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }
}