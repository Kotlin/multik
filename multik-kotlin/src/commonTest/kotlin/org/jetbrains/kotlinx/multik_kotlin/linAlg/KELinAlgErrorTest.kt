package org.jetbrains.kotlinx.multik_kotlin.linAlg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.kotlin.linalg.KELinAlgEx
import kotlin.test.Test
import kotlin.test.assertFailsWith

/**
 * The exception types a failing decomposition throws are part of the engine contract: `multik-default`
 * chooses the engine at runtime, so `KEEngine` and `NativeEngine` must agree. The mirror of this class
 * is `NativeLinAlgErrorTest` in multik-openblas.
 */
class KELinAlgErrorTest {

    private val singular = mk.ndarray(mk[mk[1.0, 2.0], mk[2.0, 4.0]])

    @Test
    fun testSolveOfASingularMatrixThrowsArithmeticException() {
        val b = mk.ndarray(mk[mk[1.0], mk[1.0]])
        assertFailsWith<ArithmeticException> { KELinAlgEx.solve(singular, b) }
    }

    @Test
    fun testInvOfASingularMatrixThrowsArithmeticException() {
        assertFailsWith<ArithmeticException> { KELinAlgEx.inv(singular) }
    }

    @Test
    fun testMismatchedShapesThrowIllegalArgumentException() {
        val b = mk.ndarray(mk[1.0, 2.0, 3.0])
        assertFailsWith<IllegalArgumentException> { KELinAlgEx.solve(singular, b) }
        assertFailsWith<IllegalArgumentException> {
            KELinAlgEx.solve(mk.ndarray(mk[mk[1.0, 2.0, 3.0], mk[4.0, 5.0, 6.0]]), b)
        }
    }
}
