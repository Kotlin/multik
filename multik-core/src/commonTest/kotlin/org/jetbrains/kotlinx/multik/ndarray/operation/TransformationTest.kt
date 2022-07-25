package org.jetbrains.kotlinx.multik.ndarray.operation

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.clip
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class TransformationTest {

    @Test
    fun clip1DIntDataType() {
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
        val result = a.clip(2, 4)
        assertContentEquals(listOf(2, 2, 3, 4, 4), result.toList())
    }

    @Test
    fun clip1DShortDataType() {
        val a = mk.ndarray<Short>(mk[1, 2, 3, 4, 5])
        val result = a.clip(2, 4)
        assertContentEquals(listOf<Short>(2, 2, 3, 4, 4), result.toList())
    }

    @Test
    fun clip1DLongDataType() {
        val a = mk.ndarray<Long>(mk[1, 2, 3, 4, 5])
        val result = a.clip(2, 4)
        assertContentEquals(listOf<Long>(2, 2, 3, 4, 4), result.toList())
    }

    @Test
    fun clip2dFloatDataType() {
        val absoluteTolerance = 0.01f
        val a = mk.ndarray(mk[ mk[1f, 2f, 3f, 4f, 5f], mk[6f, 7f, 8f, 9f, 10f]])
        val min = 3.5f
        val max = 7.1f
        val expected = listOf(3.5f, 3.5f, 3.5f, 4f, 5f, 6f, 7f, 7.1f, 7.1f, 7.1f)
        val result = a.clip(min, max).toList()
        for (i in expected.indices){ // run assert with absolute tolerance because of floating number is not stable when we test for js
            assertEquals(expected[i], result[i], absoluteTolerance = absoluteTolerance)
        }
    }

    @Test
    fun `clip_3d_byte_data_type_and_min_is_max`() {
        val inputArray = ByteArray(60) { it.toByte() }
        val a = mk.ndarray(inputArray, 2, 5, 6)
        val min = 1.toByte()
        val expected = ByteArray(60) { min }
        assertContentEquals(expected.toList(), a.clip(min, min).toList())
    }

    @Test
    fun clip4dDoubleDataType(){
        val inputArray = DoubleArray(60) { Random.nextDouble() }
        val min = 7.0
        val max = 42.0
        val expected = inputArray.copyOf().map { if (it < min) min else if (it > max) max else it }
        val a = mk.ndarray(inputArray, 2, 5, 3, 2)
        assertContentEquals(expected.toList(), a.clip(min,max).toList())
    }
}
