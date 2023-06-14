/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Create1DArrayTests {

    /*___________________________Byte_______________________________________*/

    /**
     * This method checks if a byte array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledByteArray() {
        val size = 10
        val a = mk.zeros<Byte>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0.toByte() } }
    }

    /**
     * Creates a byte array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createByteArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Byte>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1.toByte() } }
    }


    /**
     * Creates a one-dimensional array from a list of bytes
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromByteList() {
        val list = listOf<Byte>(1, 3, 8, 4, 9)
        val a: D1Array<Byte> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of bytes
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromByteSet() {
        val set = setOf<Byte>(1, 3, 8, 4, 9)
        val a: D1Array<Byte> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive ByteArray
     * and checks if the array's ByteArray representation matches the input ByteArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveByteArray() {
        val array = byteArrayOf(1, 3, 8, 4, 9)
        val a: D1Array<Byte> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getByteArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte1DArrayWithInitializationFunction() {
        val a = mk.d1array<Byte>(5) { (it + 3).toByte() }
        val expected = byteArrayOf(3, 4, 5, 6, 7)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from bytes
     * and checks if the array's ByteArray representation matches the expected output.
     */
    @Test
    fun createByte1DArrayWithNdarrayOf() {
        val a: D1Array<Byte> = mk.ndarrayOf(1.toByte(), 3.toByte(), 8.toByte(), 4.toByte(), 9.toByte())
        val expected = byteArrayOf(1, 3, 8, 4, 9)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Byte type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeByteArrayWithStep() {
        val a = mk.arange<Byte>(3, 10, step = 2)
        val expected = byteArrayOf(3, 5, 7, 9)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Byte type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeByteArrayWithNonIntegerStep() {
        val a = mk.arange<Byte>(3, 10, step = 2.5)
        val expected = byteArrayOf(3, 5, 8)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Byte type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeByteArrayWithDefaultStart() {
        val a = mk.arange<Byte>(10, step = 2)
        val expected = byteArrayOf(0, 2, 4, 6, 8)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Byte type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeByteArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Byte>(10, step = 2.3)
        val expected = byteArrayOf(0, 2, 4, 6, 9)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Byte type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceByteArray() {
        val a = mk.linspace<Byte>(3, 10, num = 15)
        val expected = byteArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Byte type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceByteArrayWithNonIntegerBounds() {
        val a = mk.linspace<Byte>(1.7, 13.8, num = 15)
        val expected = byteArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)

        assertEquals(expected.size, a.size)
        a.data.getByteArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Byte> to a one-dimensional NDArray.
     * The test creates a list of bytes, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToByteArrayNDArray() {
        val expected = listOf<Byte>(3, 8, 13, 2, 0)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }

    /*___________________________Short_______________________________________*/

    /**
     * This method checks if a short array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledShortArray() {
        val size = 10
        val a = mk.zeros<Short>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0.toShort() } }
    }

    /**
     * Creates a short array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createShortArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Short>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1.toShort() } }
    }


    /**
     * Creates a one-dimensional array from a list of shorts
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromShortList() {
        val list = listOf<Short>(1, 3, 8, 4, 9)
        val a: D1Array<Short> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of shorts
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromShortSet() {
        val set = setOf<Short>(1, 3, 8, 4, 9)
        val a: D1Array<Short> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive ShortArray
     * and checks if the array's ShortArray representation matches the input ShortArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveShortArray() {
        val array = shortArrayOf(1, 3, 8, 4, 9)
        val a: D1Array<Short> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getShortArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort1DArrayWithInitializationFunction() {
        val a = mk.d1array<Short>(5) { (it + 3).toShort() }
        val expected = shortArrayOf(3, 4, 5, 6, 7)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from shorts
     * and checks if the array's ShortArray representation matches the expected output.
     */
    @Test
    fun createShort1DArrayWithNdarrayOf() {
        val a: D1Array<Short> = mk.ndarrayOf(1.toShort(), 3.toShort(), 8.toShort(), 4.toShort(), 9.toShort())
        val expected = shortArrayOf(1, 3, 8, 4, 9)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Short type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeShortArrayWithStep() {
        val a = mk.arange<Short>(3, 10, step = 2)
        val expected = shortArrayOf(3, 5, 7, 9)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Short type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeShortArrayWithNonIntegerStep() {
        val a = mk.arange<Short>(3, 10, step = 2.5)
        val expected = shortArrayOf(3, 5, 8)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Short type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeShortArrayWithDefaultStart() {
        val a = mk.arange<Short>(10, step = 2)
        val expected = shortArrayOf(0, 2, 4, 6, 8)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Short type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeShortArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Short>(10, step = 2.3)
        val expected = shortArrayOf(0, 2, 4, 6, 9)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Short type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceShortArray() {
        val a = mk.linspace<Short>(3, 10, num = 15)
        val expected = shortArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Short type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceShortArrayWithNonIntegerBounds() {
        val a = mk.linspace<Short>(1.7, 13.8, num = 15)
        val expected = shortArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)

        assertEquals(expected.size, a.size)
        a.data.getShortArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Short> to a one-dimensional NDArray.
     * The test creates a list of Shorts, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToShortArrayNDArray() {
        val expected = listOf<Short>(3, 8, 13, 2, 0)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }


    /*___________________________Int_______________________________________*/


    /**
     * This method checks if an integer array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledIntArray() {
        val size = 10
        val a = mk.zeros<Int>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0 } }
    }

    /**
     * Creates an integer array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createIntArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Int>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1 } }
    }


    /**
     * Creates a one-dimensional array from a list of integers
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromIntList() {
        val list = listOf(1, 3, 8, 4, 9)
        val a: D1Array<Int> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of integers
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromIntSet() {
        val set = setOf(1, 3, 8, 4, 9)
        val a: D1Array<Int> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive IntArray
     * and checks if the array's IntArray representation matches the input IntArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveIntArray() {
        val array = intArrayOf(1, 3, 8, 4, 9)
        val a: D1Array<Int> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getIntArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt1DArrayWithInitializationFunction() {
        val a = mk.d1array<Int>(5) { (it + 3) }
        val expected = intArrayOf(3, 4, 5, 6, 7)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from integers
     * and checks if the array's IntArray representation matches the expected output.
     */
    @Test
    fun createInt1DArrayWithNdarrayOf() {
        val a: D1Array<Int> = mk.ndarrayOf(1, 3, 8, 4, 9)
        val expected = intArrayOf(1, 3, 8, 4, 9)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Int type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeIntArrayWithStep() {
        val a = mk.arange<Int>(3, 10, step = 2)
        val expected = intArrayOf(3, 5, 7, 9)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Int type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeIntArrayWithNonIntegerStep() {
        val a = mk.arange<Int>(3, 10, step = 2.5)
        val expected = intArrayOf(3, 5, 8)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Int type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeIntArrayWithDefaultStart() {
        val a = mk.arange<Int>(10, step = 2)
        val expected = intArrayOf(0, 2, 4, 6, 8)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Int type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeIntArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Int>(10, step = 2.3)
        val expected = intArrayOf(0, 2, 4, 6, 9)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Int type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceIntArray() {
        val a = mk.linspace<Int>(3, 10, num = 15)
        val expected = intArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Int type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceIntArrayWithNonIntegerBounds() {
        val a = mk.linspace<Int>(1.7, 13.8, num = 15)
        val expected = intArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)

        assertEquals(expected.size, a.size)
        a.data.getIntArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Int> to a one-dimensional NDArray.
     * The test creates a list of Ints, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToIntArrayNDArray() {
        val expected = listOf(3, 8, 13, 2, 0)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }

    /*___________________________Long_______________________________________*/


    /**
     * This method checks if a long array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledLongArray() {
        val size = 10
        val a = mk.zeros<Long>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0L } }
    }

    /**
     * Creates a long array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createLongArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Long>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1L } }
    }


    /**
     * Creates a one-dimensional array from a list of longs
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromLongList() {
        val list = listOf<Long>(1, 3, 8, 4, 9)
        val a: D1Array<Long> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of longs
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromLongSet() {
        val set = setOf<Long>(1, 3, 8, 4, 9)
        val a: D1Array<Long> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive LongArray
     * and checks if the array's LongArray representation matches the input LongArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveLongArray() {
        val array = longArrayOf(1, 3, 8, 4, 9)
        val a: D1Array<Long> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getLongArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong1DArrayWithInitializationFunction() {
        val a = mk.d1array<Long>(5) { it + 3L }
        val expected = longArrayOf(3, 4, 5, 6, 7)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from longs
     * and checks if the array's LongArray representation matches the expected output.
     */
    @Test
    fun createLong1DArrayWithNdarrayOf() {
        val a: D1Array<Long> = mk.ndarrayOf(1L, 3L, 8L, 4L, 9L)
        val expected = longArrayOf(1, 3, 8, 4, 9)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Long type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeLongArrayWithStep() {
        val a = mk.arange<Long>(3, 10, step = 2)
        val expected = longArrayOf(3, 5, 7, 9)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Long type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeLongArrayWithNonIntegerStep() {
        val a = mk.arange<Long>(3, 10, step = 2.5)
        val expected = longArrayOf(3, 5, 8)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Long type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeLongArrayWithDefaultStart() {
        val a = mk.arange<Long>(10, step = 2)
        val expected = longArrayOf(0, 2, 4, 6, 8)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Long type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeLongArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Long>(10, step = 2.3)
        val expected = longArrayOf(0, 2, 4, 6, 9)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Long type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceLongArray() {
        val a = mk.linspace<Long>(3, 10, num = 15)
        val expected = longArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Long type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceLongArrayWithNonIntegerBounds() {
        val a = mk.linspace<Long>(1.7, 13.8, num = 15)
        val expected = longArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)

        assertEquals(expected.size, a.size)
        a.data.getLongArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Long> to a one-dimensional NDArray.
     * The test creates a list of Longs, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToLongArrayNDArray() {
        val expected = listOf<Long>(3, 8, 13, 2, 0)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }


    /*___________________________Float_______________________________________*/

    /**
     * This method checks if a float array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledFloatArray() {
        val size = 10
        val a = mk.zeros<Float>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0f } }
    }

    /**
     * Creates a float array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createFloatArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Float>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1f } }
    }


    /**
     * Creates a one-dimensional array from a list of floats
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromFloatList() {
        val list = listOf(1f, 3f, 8f, 4f, 9f)
        val a: D1Array<Float> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromFloatSet() {
        val set = setOf(1f, 3f, 8f, 4f, 9f)
        val a: D1Array<Float> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive FloatArray
     * and checks if the array's FloatArray representation matches the input FloatArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveFloatArray() {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f)
        val a: D1Array<Float> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getFloatArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat1DArrayWithInitializationFunction() {
        val a = mk.d1array<Float>(5) { it + 3f }
        val expected = floatArrayOf(3f, 4f, 5f, 6f, 7f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from floats
     * and checks if the array's FloatArray representation matches the expected output.
     */
    @Test
    fun createFloat1DArrayWithNdarrayOf() {
        val a: D1Array<Float> = mk.ndarrayOf(1f, 3f, 8f, 4f, 9f)
        val expected = floatArrayOf(1f, 3f, 8f, 4f, 9f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Float type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeFloatArrayWithStep() {
        val a = mk.arange<Float>(3, 10, step = 2)
        val expected = floatArrayOf(3f, 5f, 7f, 9f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Float type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeFloatArrayWithNonIntegerStep() {
        val a = mk.arange<Float>(3, 10, step = 2.5)
        val expected = floatArrayOf(3f, 5.5f, 8f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Float type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeFloatArrayWithDefaultStart() {
        val a = mk.arange<Float>(10, step = 2)
        val expected = floatArrayOf(0f, 2f, 4f, 6f, 8f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Float type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeFloatArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Float>(10, step = 2.3)
        val expected = floatArrayOf(0f, 2.3f, 4.6f, 6.9f, 9.2f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Float type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceFloatArray() {
        val a = mk.linspace<Float>(3, 10, num = 15)
        val expected = floatArrayOf(3f, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 7.5f, 8f, 8.5f, 9f, 9.5f, 10f)

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Float type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceFloatArrayWithNonIntegerBounds() {
        val a = mk.linspace<Float>(1.7, 13.8, num = 15).map { round(it * 1e3f) / 1e3f }
        val expected = floatArrayOf(
            1.7f, 2.564f, 3.429f, 4.293f, 5.157f, 6.021f, 6.886f, 7.75f,
            8.614f, 9.479f, 10.343f, 11.207f, 12.071f, 12.936f, 13.8f
        )

        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Float> to a one-dimensional NDArray.
     * The test creates a list of Floats, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToFloatArrayNDArray() {
        val expected = listOf(3f, 8f, 13f, 2f, 0f)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }

    /*___________________________Double_______________________________________*/


    /**
     * This method checks if a double array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledDoubleArray() {
        val size = 10
        val a = mk.zeros<Double>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 0.0 } }
    }

    /**
     * Creates a double array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createDoubleArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<Double>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == 1.0 } }
    }


    /**
     * Creates a one-dimensional array from a list of doubles
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromDoubleList() {
        val list = listOf(1.0, 3.0, 8.0, 4.0, 9.0)
        val a: D1Array<Double> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromDoubleSet() {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0)
        val a: D1Array<Double> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive DoubleArray
     * and checks if the array's DoubleArray representation matches the input DoubleArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveDoubleArray() {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0)
        val a: D1Array<Double> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getDoubleArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble1DArrayWithInitializationFunction() {
        val a = mk.d1array<Double>(5) { it + 3.0 }
        val expected = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from Doubles
     * and checks if the array's DoubleArray representation matches the expected output.
     */
    @Test
    fun createDouble1DArrayWithNdarrayOf() {
        val a: D1Array<Double> = mk.ndarrayOf(1.0, 3.0, 8.0, 4.0, 9.0)
        val expected = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Double type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeDoubleArrayWithStep() {
        val a = mk.arange<Double>(3, 10, step = 2)
        val expected = doubleArrayOf(3.0, 5.0, 7.0, 9.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Double type.
     * The function is supposed to generate an array starting from 3, ending before 10, with a non-integer step of 2.5.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeDoubleArrayWithNonIntegerStep() {
        val a = mk.arange<Double>(3, 10, step = 2.5)
        val expected = doubleArrayOf(3.0, 5.5, 8.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Double type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a step of 2.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeDoubleArrayWithDefaultStart() {
        val a = mk.arange<Double>(10, step = 2)
        val expected = doubleArrayOf(0.0, 2.0, 4.0, 6.0, 8.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `arange` for the Double type.
     * The function is supposed to generate an array starting from the default start value of 0,
     * ending before 10, with a non-integer step of 2.3.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateArangeDoubleArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Double>(10, step = 2.3).map { round(it * 1e3) / 1e3 }
        val expected = doubleArrayOf(0.0, 2.3, 4.6, 6.9, 9.2)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Double type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 3 to 10 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceDoubleArray() {
        val a = mk.linspace<Double>(3, 10, num = 15)
        val expected = doubleArrayOf(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `linspace` for the Double type.
     * The function is supposed to generate an array of 15 elements evenly spaced from 1.7 to 13.8 inclusive.
     * The created array is then checked against an expected array for both size and element equality.
     */
    @Test
    fun generateLinspaceDoubleArrayWithNonIntegerBounds() {
        val a = mk.linspace<Double>(1.7, 13.8, num = 15).map { round(it * 1e5) / 1e5 }
        val expected = doubleArrayOf(
            1.7, 2.56429, 3.42857, 4.29286, 5.15714, 6.02143, 6.88571, 7.75,
            8.61429, 9.47857, 10.34286, 11.20714, 12.07143, 12.93571, 13.8
        )

        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<Double> to a one-dimensional NDArray.
     * The test creates a list of Doubles, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToDoubleArrayNDArray() {
        val expected = listOf(3f, 8f, 13f, 2f, 0f)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }

    /*___________________________ComplexFloat_______________________________________*/

    private val complexFloatList: List<ComplexFloat> = listOf(
        ComplexFloat(1.032f, 21.4214f),
        ComplexFloat(3.2323f, 4.35903f),
        ComplexFloat(.3498f, 5.49230f),
        ComplexFloat(4.43285f, 8.5382f),
        ComplexFloat(.34829f, 3.2389f)
    )

    /**
     * This method checks if a ComplexFloat array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexFloatArray() {
        val size = 10
        val a = mk.zeros<ComplexFloat>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == ComplexFloat.zero } }
    }

    /**
     * Creates a ComplexFloat array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexFloatArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<ComplexFloat>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == ComplexFloat.one } }
    }


    /**
     * Creates a one-dimensional array from a list of complex floats
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromComplexFloatList() {
        val list = complexFloatList
        val a: D1Array<ComplexFloat> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of complex floats
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromComplexFloatSet() {
        val set = complexFloatList.toSet()
        val a: D1Array<ComplexFloat> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive ComplexFloatArray
     * and checks if the array's ComplexFloatArray representation matches the input ComplexFloatArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveComplexFloatArrayArray() {
        val array = complexFloatList.toComplexFloatArray()
        val a: D1Array<ComplexFloat> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getComplexFloatArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat1DArrayWithInitializationFunction() {
        val a = mk.d1array<ComplexFloat>(5) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) }
        val expected =
            complexFloatArrayOf(3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i, 6.21f + 2.168f.i, 7.21f + 3.168f.i)

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from floats
     * and checks if the array's ComplexFloatArray representation matches the expected output.
     */
    @Test
    fun createComplexFloat1DArrayWithNdarrayOf() {
        val a: D1Array<ComplexFloat> = mk.ndarrayOf(
            1.32f + 2.328f.i, 4.231f + 5.83f.i, 6.02f + 7.437f.i,
            8.372f + 9.139f.i, 10.0f + 11.7453f.i
        )
        val expected = complexFloatArrayOf(
            1.32f + 2.328f.i, 4.231f + 5.83f.i, 6.02f + 7.437f.i,
            8.372f + 9.139f.i, 10.0f + 11.7453f.i
        )

        assertEquals(expected.size, a.size)
        a.data.getComplexFloatArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<ComplexFloat> to a one-dimensional NDArray.
     * The test creates a list of ComplexFloats, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToComplexFloatArrayNDArray() {
        val expected = listOf(
            ComplexFloat(1.32f, 2.328f),
            ComplexFloat(4.231f, 5.83f),
            ComplexFloat(6.02f, 7.437f),
            ComplexFloat(8.372f, 9.139f),
            ComplexFloat(10f, 11.7453f)
        )
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }

    /*___________________________ComplexDouble_______________________________________*/

    private val complexDoubleList: List<ComplexDouble> = listOf(
        ComplexDouble(1.032, 21.4214),
        ComplexDouble(3.2323, 4.35903),
        ComplexDouble(.3498, 5.49230),
        ComplexDouble(4.43285, 8.5382),
        ComplexDouble(.34829, 3.2389)
    )

    /**
     * This method checks if a ComplexDouble array of a given size is correctly created with all elements set to zero.
     */
    @Test
    fun createZeroFilledComplexDoubleArray() {
        val size = 10
        val a = mk.zeros<ComplexDouble>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == ComplexDouble.zero } }
    }

    /**
     * Creates a ComplexDouble array filled with ones of a given size and checks if all elements are set to one.
     */
    @Test
    fun createComplexDoubleArrayFilledWithOnes() {
        val size = 10
        val a = mk.ones<ComplexDouble>(size)

        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == ComplexDouble.one } }
    }


    /**
     * Creates a one-dimensional array from a list of complex doubles
     * and checks if the array's list representation matches the input list.
     */
    @Test
    fun createOneDimensionalArrayFromComplexDoubleList() {
        val list = complexDoubleList
        val a: D1Array<ComplexDouble> = mk.ndarray(list)
        assertEquals(list, a.toList())
    }


    /**
     * Creates a one-dimensional array from a set of complex doubles
     * and checks if the array's set representation matches the input set.
     */
    @Test
    fun createOneDimensionalArrayFromComplexDoubleSet() {
        val set = complexDoubleList.toSet()
        val a: D1Array<ComplexDouble> = mk.ndarray(set, shape = intArrayOf(5))

        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }


    /**
     * Creates a one-dimensional array from a primitive ComplexDoubleArray
     * and checks if the array's ComplexDoubleArray representation matches the input ComplexDoubleArray.
     */
    @Test
    fun createOneDimensionalArrayFromPrimitiveComplexDoubleArrayArray() {
        val array = complexDoubleList.toComplexDoubleArray()
        val a: D1Array<ComplexDouble> = mk.ndarray(array)

        assertEquals(array.size, a.size)
        a.data.getComplexDoubleArray() shouldBe array
    }


    /**
     * Creates a one-dimensional array with a given size using an initialization function
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble1DArrayWithInitializationFunction() {
        val a = mk.d1array<ComplexDouble>(5) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) }
        val expected =
            complexDoubleArrayOf(3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i, 6.21 + 2.168.i, 7.21 + 3.168.i)

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }

    /**
     * Creates a one-dimensional array with a given size from doubles
     * and checks if the array's ComplexDoubleArray representation matches the expected output.
     */
    @Test
    fun createComplexDouble1DArrayWithNdarrayOf() {
        val a: D1Array<ComplexDouble> =
            mk.ndarrayOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i)
        val expected =
            complexDoubleArrayOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i)

        assertEquals(expected.size, a.size)
        a.data.getComplexDoubleArray() shouldBe expected
    }

    /**
     * Tests the function `toNDArray` which converts an Iterable<ComplexDouble> to a one-dimensional NDArray.
     * The test creates a list of ComplexDoubles, converts it to an NDArray,
     * and then checks that the size and elements of the NDArray match those of the original list.
     */
    @Test
    fun convertIterableToComplexDoubleArrayNDArray() {
        val expected = listOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i)
        val a = expected.toNDArray()

        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }
}
