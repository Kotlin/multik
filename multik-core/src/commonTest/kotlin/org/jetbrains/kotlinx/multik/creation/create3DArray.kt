/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD3
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Create3DArrayTests {

    /*___________________________Byte_______________________________________*/

    /**
     * This method checks if a byte array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledByteArray() {
        val dim1 = 5
        val dim2 = 7
        val dim3 = 3
        val a = mk.zeros<Byte>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Byte>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1.toByte() } }
    }

    /**
     * Creates a three-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromByteList() {
        val list =
            listOf(listOf(listOf<Byte>(1, 3), listOf<Byte>(7, 8)), listOf(listOf<Byte>(4, 7), listOf<Byte>(9, 2)))
        val a: D3Array<Byte> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of bytes
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromByteSet() {
        val set = setOf<Byte>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Byte> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive ByteArray
     * and checks if the array's ByteArray representation matches the input ByteArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveByteArray() {
        val array = byteArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getByteArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte3DArrayWithInitializationFunction() {
        val a = mk.d3array<Byte>(2, 3, 2) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices(2, 3, 2) { i, j, k -> (i * j + k).toByte() }
        val expected = byteArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedByte3DArray() {
        val list = listOf(
            listOf(
                listOf<Byte>(1, 3),
                listOf<Byte>(8)
            ),
            listOf(
                listOf<Byte>(4, 7)
            )
        )

        val expected = listOf(
            listOf(listOf<Byte>(1, 3), listOf<Byte>(8, 0)),
            listOf(listOf<Byte>(4, 7), listOf<Byte>(0, 0))
        )

        val a: D3Array<Byte> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<Short>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Short>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1.toShort() } }
    }


    /**
     * Creates a three-dimensional array from a list of short lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromShortList() {
        val list =
            listOf(listOf(listOf<Short>(1, 3), listOf<Short>(7, 8)), listOf(listOf<Short>(4, 7), listOf<Short>(9, 2)))
        val a: D3Array<Short> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of shorts
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromShortSet() {
        val set = setOf<Short>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Short> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive ShortArray
     * and checks if the array's ShortArray representation matches the input ShortArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveShortArray() {
        val array = shortArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getShortArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort3DArrayWithInitializationFunction() {
        val a = mk.d3array<Short>(2, 3, 2) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices(2, 3, 2) { i, j, k -> (i * j + k).toShort() }
        val expected = shortArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedShort3DArray() {
        val list = listOf(
            listOf(
                listOf<Short>(1, 3),
                listOf<Short>(8)
            ),
            listOf(
                listOf<Short>(4, 7)
            )
        )

        val expected = listOf(
            listOf(listOf<Short>(1, 3), listOf<Short>(8, 0)),
            listOf(listOf<Short>(4, 7), listOf<Short>(0, 0))
        )

        val a: D3Array<Short> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<Int>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Int>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1 } }
    }


    /**
     * Creates a three-dimensional array from a list of integer lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromIntList() {
        val list = listOf(listOf(listOf(1, 3), listOf(7, 8)), listOf(listOf(4, 7), listOf(9, 2)))
        val a: D3Array<Int> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of integers
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromIntSet() {
        val set = setOf(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Int> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive IntArray
     * and checks if the array's IntArray representation matches the input IntArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveIntArray() {
        val array = intArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getIntArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt3DArrayWithInitializationFunction() {
        val a = mk.d3array<Int>(2, 3, 2) { (it + 3) }
        val expected = intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices(2, 3, 2) { i, j, k -> i * j + k }
        val expected = intArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedInt3DArray() {
        val list = listOf(
            listOf(
                listOf(1, 3),
                listOf(8)
            ),
            listOf(
                listOf(4, 7)
            )
        )

        val expected = listOf(
            listOf(listOf(1, 3), listOf(8, 0)),
            listOf(listOf(4, 7), listOf(0, 0))
        )

        val a: D3Array<Int> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<Long>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Long>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1L } }
    }

    /**
     * Creates a three-dimensional array from a list of long lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromLongList() {
        val list = listOf(listOf(listOf(1L, 3L), listOf(7L, 8L)), listOf(listOf(4L, 7L), listOf(9L, 2L)))
        val a: D3Array<Long> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of longs
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromLongSet() {
        val set = setOf(1L, 3L, 8L, 4L, 9L, 2L, 7L, 5L, 11L, 13L, -8L, -1L)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Long> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive LongArray
     * and checks if the array's LongArray representation matches the input LongArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveLongArray() {
        val array = longArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getLongArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong3DArrayWithInitializationFunction() {
        val a = mk.d3array<Long>(2, 3, 2) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices<Long>(2, 3, 2) { i, j, k -> i * j + k.toLong() }
        val expected = longArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedLong3DArray() {
        val list = listOf(
            listOf(
                listOf(1L, 3L),
                listOf(8L)
            ),
            listOf(
                listOf(4L, 7L)
            )
        )

        val expected = listOf(
            listOf(listOf(1L, 3L), listOf(8L, 0L)),
            listOf(listOf(4L, 7L), listOf(0L, 0L))
        )

        val a: D3Array<Long> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<Float>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Float>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1f } }
    }

    /**
     * Creates a three-dimensional array from a list of float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromFloatList() {
        val list = listOf(listOf(listOf(1f, 3f), listOf(7f, 8f)), listOf(listOf(4f, 7f), listOf(9f, 2f)))
        val a: D3Array<Float> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromFloatSet() {
        val set = setOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 5f, 11f, 13f, -8f, -1f)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Float> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive FloatArray
     * and checks if the array's FloatArray representation matches the input FloatArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveFloatArray() {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 3f, 4f, 3f, 8f, 1f)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getFloatArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat3DArrayWithInitializationFunction() {
        val a = mk.d3array<Float>(2, 3, 2) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices<Float>(2, 3, 2) { i, j, k -> i * j + k.toFloat() }
        val expected = floatArrayOf(0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f, 1f, 2f, 2f, 3f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedFloat3DArray() {
        val list = listOf(
            listOf(
                listOf(1f, 3f),
                listOf(8f)
            ),
            listOf(
                listOf(4f, 7f),
            )
        )

        val expected = listOf(
            listOf(listOf(1f, 3f), listOf(8f, 0f)),
            listOf(listOf(4f, 7f), listOf(0f, 0f))
        )

        val a: D3Array<Float> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<Double>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<Double>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == 1.0 } }
    }

    /**
     * Creates a three-dimensional array from a list of double lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromDoubleList() {
        val list = listOf(listOf(listOf(1.0, 3.0), listOf(7.0, 8.0)), listOf(listOf(4.0, 7.0), listOf(9.0, 2.0)))
        val a: D3Array<Double> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromDoubleSet() {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 5.0, 11.0, 13.0, -8.0, -1.0)
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<Double> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive DoubleArray
     * and checks if the array's DoubleArray representation matches the input DoubleArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveDoubleArray() {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 3.0, 4.0, 3.0, 8.0, 1.0)
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getDoubleArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble3DArrayWithInitializationFunction() {
        val a = mk.d3array<Double>(2, 3, 2) { it + 3.0 }
        val expected = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices<Double>(2, 3, 2) { i, j, k -> i * j + k.toDouble() }
        val expected = doubleArrayOf(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a three-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedDouble3DArray() {
        val list = listOf(
            listOf(
                listOf(1.0, 3.0),
                listOf(8.0)
            ),
            listOf(
                listOf(4.0, 7.0),
            )
        )

        val expected = listOf(
            listOf(listOf(1.0, 3.0), listOf(8.0, 0.0)),
            listOf(listOf(4.0, 7.0), listOf(0.0, 0.0))
        )

        val a: D3Array<Double> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD3())
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
        val a = mk.zeros<ComplexFloat>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<ComplexFloat>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == ComplexFloat.one } }
    }

    /**
     * Creates a three-dimensional array from a list of complex float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromComplexFloatList() {
        val list = listOf(
            listOf(listOf(1f + 0f.i, 3f + 0f.i), listOf(7f + 0f.i, 8f + 0f.i)),
            listOf(listOf(4f + 0f.i, 7f + 0f.i), listOf(9f + 0f.i, 2f + 0f.i))
        )
        val a: D3Array<ComplexFloat> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }

    /**
     * Creates a three-dimensional array from a set of complex floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromComplexFloatSet() {
        val set = setOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 5f + 0f.i, 11f + 0f.i, 13f + 0f.i, -8f + 0f.i, -1f + 0f.i
        )
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<ComplexFloat> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive ComplexFloatArray
     * and checks if the array's ComplexFloatArray representation matches the input ComplexFloatArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveComplexFloatArrayArray() {
        val array = complexFloatArrayOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 3f + 0f.i, 4f + 0f.i, 3f + 0f.i, 8f + 0f.i, 1f + 0f.i
        )
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getComplexFloatArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat3DArrayWithInitializationFunction() {
        val a = mk.d3array<ComplexFloat>(2, 3, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
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
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices<ComplexFloat>(2, 3, 2) { i, j, k -> i * j + k + ComplexFloat(7) }
        val expected = complexFloatArrayOf(
            7f + 0f.i, 8f + 0f.i, 7f + 0f.i, 8f + 0f.i, 7f + 0f.i, 8f + 0f.i,
            7f + 0f.i, 8f + 0f.i, 8f + 0f.i, 9f + 0f.i, 9f + 0f.i, 10f + 0f.i
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
        val a = mk.zeros<ComplexDouble>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
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
        val a = mk.ones<ComplexDouble>(dim1, dim2, dim3)

        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == ComplexDouble.one } }
    }

    /**
     * Creates a three-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createThreeDimensionalArrayFromComplexDoubleList() {
        val list = listOf(
            listOf(listOf(1.0 + .0.i, 3.0 + .0.i), listOf(7.0 + .0.i, 8 + .0.i)),
            listOf(listOf(4.0 + .0.i, 7.0 + .0.i), listOf(9.0 + .0.i, 2.0 + .0.i))
        )
        val a: D3Array<ComplexDouble> = mk.ndarray(list)
        assertEquals(list, a.toListD3())
    }


    /**
     * Creates a three-dimensional array from a set of complex doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createThreeDimensionalArrayFromComplexDoubleSet() {
        val set = setOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 5.0 + .0.i, 11.0 + .0.i, 13.0 + .0.i, -8.0 + .0.i, -1.0 + .0.i
        )
        val shape = intArrayOf(2, 3, 2)
        val a: D3Array<ComplexDouble> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a three-dimensional array from a primitive ComplexDoubleArray
     * and checks if the array's ComplexDoubleArray representation matches the input ComplexDoubleArray.
     */
    @Test
    fun createThreeDimensionalArrayFromPrimitiveComplexDoubleArrayArray() {
        val array = complexDoubleArrayOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 3.0 + .0.i, 4.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 1.0 + .0.i
        )
        val a = mk.ndarray(array, 2, 3, 2)

        assertEquals(array.size, a.size)
        a.data.getComplexDoubleArray() shouldBe array
    }


    /**
     * Creates a three-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble3DArrayWithInitializationFunction() {
        val a = mk.d3array<ComplexDouble>(2, 3, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
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
     * Creates a three-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble3DArrayWithInitAndIndices() {
        val a = mk.d3arrayIndices<ComplexDouble>(2, 3, 2) { i, j, k -> i * j + k + ComplexDouble(7) }
        val expected = complexDoubleArrayOf(
            7.0 + .0.i, 8.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i,
            7.0 + .0.i, 8.0 + .0.i, 8.0 + .0.i, 9.0 + .0.i, 9.0 + .0.i, 10.0 + .0.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }
}
