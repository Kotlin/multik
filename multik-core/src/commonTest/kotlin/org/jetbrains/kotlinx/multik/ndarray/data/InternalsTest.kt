package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

class InternalsTest {

    @Test
    fun `require_equal_shape_throws_exception_for_unequal_shape`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2, 4)
        expectUnEqualShape(left, right)
    }

    @Test
    fun `require_equal_shape_throws_exception_for_different_no_of_dim`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2)
        expectUnEqualShape(left, right)
    }

    @Test
    fun `require_equal_shape_succeeds_for_arrays_with_equal_shapes`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2, 3)
        requireEqualShape(left.shape, right.shape)
    }

    @Test
    fun `require_equal_shape_succeeds_empty_arrays`() {
        val left = mk.zeros<Double>(0)
        val right = mk.zeros<Double>(0)
        assertTrue(left.isEmpty())
        assertTrue(right.isEmpty())
        requireEqualShape(left.shape, right.shape)
    }

    private fun expectUnEqualShape(left: NDArray<Double, *>, right: NDArray<Double, *>) {
        try {
            requireEqualShape(left.shape, right.shape)
            fail("Exception expected")
        } catch (e: IllegalArgumentException) { }
    }
}