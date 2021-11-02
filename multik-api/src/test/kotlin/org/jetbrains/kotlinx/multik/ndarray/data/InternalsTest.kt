package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

class InternalsTest {

    @Test
    fun `require equal shape throws exception for unequal shape`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2, 4)
        expectUnEqualShape(left, right)
    }

    @Test
    fun `require equal shape throws exception for different no of dim`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2)
        expectUnEqualShape(left, right)
    }

    @Test
    fun `require equal shape succeeds for arrays with equal shapes`() {
        val left = mk.zeros<Double>(0, 1, 2, 3)
        val right = mk.zeros<Double>(0, 1, 2, 3)
        requireEqualShape(left.shape, right.shape)
    }

    @Test
    fun `require equal shape succeeds empty arrays`() {
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