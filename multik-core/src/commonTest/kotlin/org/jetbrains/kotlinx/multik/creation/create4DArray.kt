/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.D4Array
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD4
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Create4DArrayTests {

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
        val a = mk.zeros<Byte>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Byte>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1.toByte() } }
    }

    /**
     * Creates a four-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromByteList() {
        val list =
            listOf(
                listOf(
                    listOf(listOf<Byte>(1, 3), listOf<Byte>(7, 8)),
                    listOf(listOf<Byte>(4, 7), listOf<Byte>(9, 2))
                )
            )
        val a: D4Array<Byte> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of bytes
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromByteSet() {
        val set = setOf<Byte>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Byte> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive ByteArray
     * and checks if the array's ByteArray representation matches the input ByteArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveByteArray() {
        val array = byteArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getByteArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte4DArrayWithInitializationFunction() {
        val a = mk.d4array<Byte>(2, 3, 1, 2) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices(2, 3, 1, 2) { i, j, k, l -> (i * j + k - l).toByte() }
        val expected = byteArrayOf(0, -1, 0, -1, 0, -1, 0, -1, 1, 0, 2, 1)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedByte4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf<Byte>(1, 3),
                    listOf<Byte>(8)
                ),
                listOf(
                    listOf<Byte>(4, 7)
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf<Byte>(1, 3), listOf<Byte>(8, 0)),
                listOf(listOf<Byte>(4, 7), listOf<Byte>(0, 0))
            )
        )

        val a: D4Array<Byte> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<Short>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Short>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1.toShort() } }
    }


    /**
     * Creates a four-dimensional array from a list of short lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromShortList() {
        val list =
            listOf(
                listOf(
                    listOf(listOf<Short>(1, 3), listOf<Short>(7, 8)),
                    listOf(listOf<Short>(4, 7), listOf<Short>(9, 2))
                )
            )
        val a: D4Array<Short> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of shorts
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromShortSet() {
        val set = setOf<Short>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Short> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive ShortArray
     * and checks if the array's ShortArray representation matches the input ShortArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveShortArray() {
        val array = shortArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getShortArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort4DArrayWithInitializationFunction() {
        val a = mk.d4array<Short>(2, 3, 1, 2) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices(2, 3, 1, 2) { i, j, k, l -> (i * j + k - l).toShort() }
        val expected = shortArrayOf(0, -1, 0, -1, 0, -1, 0, -1, 1, 0, 2, 1)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedShort4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf<Short>(1, 3),
                    listOf<Short>(8)
                ),
                listOf(
                    listOf<Short>(4, 7)
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf<Short>(1, 3), listOf<Short>(8, 0)),
                listOf(listOf<Short>(4, 7), listOf<Short>(0, 0))
            )
        )

        val a: D4Array<Short> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<Int>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Int>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1 } }
    }


    /**
     * Creates a four-dimensional array from a list of integer lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromIntList() {
        val list = listOf(listOf(listOf(listOf(1, 3), listOf(7, 8)), listOf(listOf(4, 7), listOf(9, 2))))
        val a: D4Array<Int> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of integers
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromIntSet() {
        val set = setOf(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Int> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive IntArray
     * and checks if the array's IntArray representation matches the input IntArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveIntArray() {
        val array = intArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getIntArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt4DArrayWithInitializationFunction() {
        val a = mk.d4array<Int>(2, 3, 1, 2) { (it + 3) }
        val expected = intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices(2, 3, 1, 2) { i, j, k, l -> i * j + k - l }
        val expected = intArrayOf(0, -1, 0, -1, 0, -1, 0, -1, 1, 0, 2, 1)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedInt4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf(1, 3),
                    listOf(8)
                ),
                listOf(
                    listOf(4, 7)
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf(1, 3), listOf(8, 0)),
                listOf(listOf(4, 7), listOf(0, 0))
            )
        )

        val a: D4Array<Int> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<Long>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Long>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1L } }
    }

    /**
     * Creates a four-dimensional array from a list of long lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromLongList() {
        val list = listOf(listOf(listOf(listOf(1L, 3L), listOf(7L, 8L)), listOf(listOf(4L, 7L), listOf(9L, 2L))))
        val a: D4Array<Long> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of longs
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromLongSet() {
        val set = setOf(1L, 3L, 8L, 4L, 9L, 2L, 7L, 5L, 11L, 13L, -8L, -1L)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Long> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive LongArray
     * and checks if the array's LongArray representation matches the input LongArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveLongArray() {
        val array = longArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getLongArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong4DArrayWithInitializationFunction() {
        val a = mk.d4array<Long>(2, 3, 1, 2) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices<Long>(2, 3, 1, 2) { i, j, k, l -> i * j + k.toLong() - l }
        val expected = longArrayOf(0, -1, 0, -1, 0, -1, 0, -1, 1, 0, 2, 1)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedLong4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf(1L, 3L),
                    listOf(8L)
                ),
                listOf(
                    listOf(4L, 7L)
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf(1L, 3L), listOf(8L, 0L)),
                listOf(listOf(4L, 7L), listOf(0L, 0L))
            )
        )

        val a: D4Array<Long> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<Float>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Float>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1f } }
    }

    /**
     * Creates a four-dimensional array from a list of float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromFloatList() {
        val list = listOf(listOf(listOf(listOf(1f, 3f), listOf(7f, 8f)), listOf(listOf(4f, 7f), listOf(9f, 2f))))
        val a: D4Array<Float> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromFloatSet() {
        val set = setOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 5f, 11f, 13f, -8f, -1f)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Float> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive FloatArray
     * and checks if the array's FloatArray representation matches the input FloatArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveFloatArray() {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 3f, 4f, 3f, 8f, 1f)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getFloatArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat4DArrayWithInitializationFunction() {
        val a = mk.d4array<Float>(2, 3, 1, 2) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices<Float>(2, 3, 1, 2) { i, j, k, l -> i * j + k.toFloat() - l }
        val expected = floatArrayOf(0f, -1f, 0f, -1f, 0f, -1f, 0f, -1f, 1f, 0f, 2f, 1f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedFloat4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf(1f, 3f),
                    listOf(8f)
                ),
                listOf(
                    listOf(4f, 7f),
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf(1f, 3f), listOf(8f, 0f)),
                listOf(listOf(4f, 7f), listOf(0f, 0f))
            )
        )

        val a: D4Array<Float> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<Double>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<Double>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == 1.0 } }
    }

    /**
     * Creates a four-dimensional array from a list of double lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromDoubleList() {
        val list =
            listOf(listOf(listOf(listOf(1.0, 3.0), listOf(7.0, 8.0)), listOf(listOf(4.0, 7.0), listOf(9.0, 2.0))))
        val a: D4Array<Double> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromDoubleSet() {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 5.0, 11.0, 13.0, -8.0, -1.0)
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<Double> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive DoubleArray
     * and checks if the array's DoubleArray representation matches the input DoubleArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveDoubleArray() {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 3.0, 4.0, 3.0, 8.0, 1.0)
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getDoubleArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble4DArrayWithInitializationFunction() {
        val a = mk.d4array<Double>(2, 3, 1, 2) { it + 3.0 }
        val expected = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices<Double>(2, 3, 1, 2) { i, j, k, l -> i * j + k.toDouble() - l }
        val expected = doubleArrayOf(0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0, 2.0, 1.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function 'createAlignedNDArray' that creates a four-dimensional array from a list of number lists.
     * The test asserts that:
     * - The output array's size matches the size of the longest list in the input
     * and all lists are filled to match this length.
     * - The lists shorter than the longest one are filled with the specified filling value.
     */
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAlignedDouble4DArray() {
        val list = listOf(
            listOf(
                listOf(
                    listOf(1.0, 3.0),
                    listOf(8.0)
                ),
                listOf(
                    listOf(4.0, 7.0),
                )
            )
        )

        val expected = listOf(
            listOf(
                listOf(listOf(1.0, 3.0), listOf(8.0, 0.0)),
                listOf(listOf(4.0, 7.0), listOf(0.0, 0.0))
            )
        )

        val a: D4Array<Double> = mk.createAlignedNDArray(list, filling = 0.0)

        assertEquals(expected, a.toListD4())
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
        val a = mk.zeros<ComplexFloat>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<ComplexFloat>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == ComplexFloat.one } }
    }

    /**
     * Creates a four-dimensional array from a list of complex float lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromComplexFloatList() {
        val list = listOf(
            listOf(
                listOf(listOf(1f + 0f.i, 3f + 0f.i), listOf(7f + 0f.i, 8f + 0f.i)),
                listOf(listOf(4f + 0f.i, 7f + 0f.i), listOf(9f + 0f.i, 2f + 0f.i))
            )
        )
        val a: D4Array<ComplexFloat> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }

    /**
     * Creates a four-dimensional array from a set of complex floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromComplexFloatSet() {
        val set = setOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 5f + 0f.i, 11f + 0f.i, 13f + 0f.i, -8f + 0f.i, -1f + 0f.i
        )
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<ComplexFloat> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive ComplexFloatArray
     * and checks if the array's ComplexFloatArray representation matches the input ComplexFloatArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveComplexFloatArray() {
        val array = complexFloatArrayOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 3f + 0f.i, 4f + 0f.i, 3f + 0f.i, 8f + 0f.i, 1f + 0f.i
        )
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getComplexFloatArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat4DArrayWithInitializationFunction() {
        val a = mk.d4array<ComplexFloat>(2, 3, 1, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
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
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices<ComplexFloat>(2, 3, 1, 2) { i, j, k, l -> i * j + k - l + ComplexFloat(7) }
        val expected = complexFloatArrayOf(
            7f + 0f.i, 6f + 0f.i, 7f + 0f.i, 6f + 0f.i, 7f + 0f.i, 6f + 0f.i,
            7f + 0f.i, 6f + 0f.i, 8f + 0f.i, 7f + 0f.i, 9f + 0f.i, 8f + 0f.i
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
        val a = mk.zeros<ComplexDouble>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
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
        val a = mk.ones<ComplexDouble>(dim1, dim2, dim3, dim4)

        assertEquals(dim1 * dim2 * dim3 * dim4, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4, a.data.size)
        assertTrue { a.all { it == ComplexDouble.one } }
    }

    /**
     * Creates a four-dimensional array from a list of byte lists
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createFourDimensionalArrayFromComplexDoubleList() {
        val list = listOf(
            listOf(
                listOf(listOf(1.0 + .0.i, 3.0 + .0.i), listOf(7.0 + .0.i, 8 + .0.i)),
                listOf(listOf(4.0 + .0.i, 7.0 + .0.i), listOf(9.0 + .0.i, 2.0 + .0.i))
            )
        )
        val a: D4Array<ComplexDouble> = mk.ndarray(list)
        assertEquals(list, a.toListD4())
    }


    /**
     * Creates a four-dimensional array from a set of complex doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createFourDimensionalArrayFromComplexDoubleSet() {
        val set = setOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 5.0 + .0.i, 11.0 + .0.i, 13.0 + .0.i, -8.0 + .0.i, -1.0 + .0.i
        )
        val shape = intArrayOf(2, 3, 1, 2)
        val a: D4Array<ComplexDouble> = mk.ndarray(set, shape = shape)

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a four-dimensional array from a primitive ComplexDoubleArray
     * and checks if the array's ComplexDoubleArray representation matches the input ComplexDoubleArray.
     */
    @Test
    fun createFourDimensionalArrayFromPrimitiveComplexDoubleArray() {
        val array = complexDoubleArrayOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 3.0 + .0.i, 4.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 1.0 + .0.i
        )
        val a = mk.ndarray(array, 2, 3, 1, 2)

        assertEquals(array.size, a.size)
        a.data.getComplexDoubleArray() shouldBe array
    }


    /**
     * Creates a four-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble4DArrayWithInitializationFunction() {
        val a = mk.d4array<ComplexDouble>(2, 3, 1, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
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
     * Creates a four-dimensional array with a given size using an initialization function and indices.
     * Checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble4DArrayWithInitAndIndices() {
        val a = mk.d4arrayIndices<ComplexDouble>(2, 3, 1, 2) { i, j, k, l -> i * j + k - l + ComplexDouble(7) }
        val expected = complexDoubleArrayOf(
            7.0 + .0.i, 6.0 + .0.i, 7.0 + .0.i, 6.0 + .0.i, 7.0 + .0.i, 6.0 + .0.i,
            7.0 + .0.i, 6.0 + .0.i, 8.0 + .0.i, 7.0 + .0.i, 9.0 + .0.i, 8.0 + .0.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }
}