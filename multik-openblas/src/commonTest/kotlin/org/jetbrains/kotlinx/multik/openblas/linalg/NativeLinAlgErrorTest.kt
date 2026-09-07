package org.jetbrains.kotlinx.multik.openblas.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.openblas.NativeTestBase
import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * The exception types a failing decomposition throws are part of the engine contract: `multik-default`
 * chooses the engine at runtime, so `NativeEngine` and `KEEngine` must agree. The mirror of this class
 * is `KELinAlgErrorTest` in multik-kotlin.
 */
class NativeLinAlgErrorTest : NativeTestBase() {

    private val singular = mk.ndarray(mk[mk[1.0, 2.0], mk[2.0, 4.0]])

    @Test
    fun testSolveOfASingularMatrixThrowsArithmeticException() {
        val b = mk.ndarray(mk[mk[1.0], mk[1.0]])
        assertFailsWith<ArithmeticException> { NativeLinAlgEx.solve(singular, b) }
    }

    @Test
    fun testInvOfASingularMatrixThrowsArithmeticException() {
        assertFailsWith<ArithmeticException> { NativeLinAlgEx.inv(singular) }
    }

    @Test
    fun testMismatchedShapesThrowIllegalArgumentException() {
        val b = mk.ndarray(mk[1.0, 2.0, 3.0])
        assertFailsWith<IllegalArgumentException> { NativeLinAlgEx.solve(singular, b) }
        assertFailsWith<IllegalArgumentException> {
            NativeLinAlgEx.solve(mk.ndarray(mk[mk[1.0, 2.0, 3.0], mk[4.0, 5.0, 6.0]]), b)
        }
    }
}
