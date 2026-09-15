package org.jetbrains.kotlinx.multik.ndarray.operation

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.d4array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.bounds
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.data.sl
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The scalar-on-left operators build their result in a single pass straight out of the source's
 * backing array, so they must produce exactly the same elements for a contiguous array and for a
 * view over someone else's buffer, must never write into that buffer, and must always hand back a
 * detached contiguous array.
 */
class ScalarArithmeticTest {

    // Values, all dtypes

    @Test
    fun testScalarOpsOnByte() {
        val a = mk.ndarrayOf(1.toByte(), 2.toByte(), 4.toByte())
        assertContentEquals(listOf<Byte>(11, 12, 14), (10.toByte() + a).toList())
        assertContentEquals(listOf<Byte>(9, 8, 6), (10.toByte() - a).toList())
        assertContentEquals(listOf<Byte>(10, 20, 40), (10.toByte() * a).toList())
        assertContentEquals(listOf<Byte>(10, 5, 2), (10.toByte() / a).toList())
    }

    @Test
    fun testScalarOpsOnShort() {
        val a = mk.ndarrayOf(1.toShort(), 2.toShort(), 4.toShort())
        assertContentEquals(listOf<Short>(11, 12, 14), (10.toShort() + a).toList())
        assertContentEquals(listOf<Short>(9, 8, 6), (10.toShort() - a).toList())
        assertContentEquals(listOf<Short>(10, 20, 40), (10.toShort() * a).toList())
        assertContentEquals(listOf<Short>(10, 5, 2), (10.toShort() / a).toList())
    }

    @Test
    fun testScalarOpsOnInt() {
        val a = mk.ndarrayOf(1, 2, 4)
        assertContentEquals(listOf(11, 12, 14), (10 + a).toList())
        assertContentEquals(listOf(9, 8, 6), (10 - a).toList())
        assertContentEquals(listOf(10, 20, 40), (10 * a).toList())
        assertContentEquals(listOf(10, 5, 2), (10 / a).toList())
    }

    @Test
    fun testScalarOpsOnLong() {
        val a = mk.ndarrayOf(1L, 2L, 4L)
        assertContentEquals(listOf(11L, 12L, 14L), (10L + a).toList())
        assertContentEquals(listOf(9L, 8L, 6L), (10L - a).toList())
        assertContentEquals(listOf(10L, 20L, 40L), (10L * a).toList())
        assertContentEquals(listOf(10L, 5L, 2L), (10L / a).toList())
    }

    @Test
    fun testScalarOpsOnFloat() {
        val a = mk.ndarrayOf(1f, 2f, 4f)
        assertContentEquals(listOf(11f, 12f, 14f), (10f + a).toList())
        assertContentEquals(listOf(9f, 8f, 6f), (10f - a).toList())
        assertContentEquals(listOf(10f, 20f, 40f), (10f * a).toList())
        assertContentEquals(listOf(10f, 5f, 2.5f), (10f / a).toList())
    }

    @Test
    fun testScalarOpsOnDouble() {
        val a = mk.ndarrayOf(1.0, 2.0, 4.0)
        assertContentEquals(listOf(11.0, 12.0, 14.0), (10.0 + a).toList())
        assertContentEquals(listOf(9.0, 8.0, 6.0), (10.0 - a).toList())
        assertContentEquals(listOf(10.0, 20.0, 40.0), (10.0 * a).toList())
        assertContentEquals(listOf(10.0, 5.0, 2.5), (10.0 / a).toList())
    }

    @Test
    fun testScalarOpsOnComplexFloat() {
        val a = mk.ndarray(listOf(ComplexFloat(1f, 1f), ComplexFloat(2f, 0f)))
        val s = ComplexFloat(2f, 0f)
        assertContentEquals(listOf(ComplexFloat(3f, 1f), ComplexFloat(4f, 0f)), (s + a).toList())
        assertContentEquals(listOf(ComplexFloat(1f, -1f), ComplexFloat(0f, 0f)), (s - a).toList())
        assertContentEquals(listOf(ComplexFloat(2f, 2f), ComplexFloat(4f, 0f)), (s * a).toList())
        assertContentEquals(listOf(ComplexFloat(1f, -1f), ComplexFloat(1f, 0f)), (s / a).toList())
    }

    @Test
    fun testScalarOpsOnComplexDouble() {
        val a = mk.ndarray(listOf(ComplexDouble(1.0, 1.0), ComplexDouble(2.0, 0.0)))
        val s = ComplexDouble(2.0, 0.0)
        assertContentEquals(listOf(ComplexDouble(3.0, 1.0), ComplexDouble(4.0, 0.0)), (s + a).toList())
        assertContentEquals(listOf(ComplexDouble(1.0, -1.0), ComplexDouble(0.0, 0.0)), (s - a).toList())
        assertContentEquals(listOf(ComplexDouble(2.0, 2.0), ComplexDouble(4.0, 0.0)), (s * a).toList())
        assertContentEquals(listOf(ComplexDouble(1.0, -1.0), ComplexDouble(1.0, 0.0)), (s / a).toList())
    }

    // Views: the result must not depend on the source's layout

    private fun assertMatchesCompactedSource(view: MultiArray<Int, *>) {
        assertFalse(view.consistent, "the fixture stopped being a view; the test would be vacuous")
        val elements = view.toList()
        assertContentEquals(elements.map { 10 + it }, (10 + view).toList())
        assertContentEquals(elements.map { 10 - it }, (10 - view).toList())
        assertContentEquals(elements.map { 10 * it }, (10 * view).toList())
        assertContentEquals(elements.map { 10 / it }, (10 / view).toList())
        assertContentEquals(view.shape, (10 - view).shape)
    }

    @Test
    fun testScalarOpsOnTransposedView() {
        assertMatchesCompactedSource(mk.d2array(2, 3) { it + 1 }.transpose())
        assertMatchesCompactedSource(mk.d3array(2, 3, 4) { it + 1 }.transpose())
        assertMatchesCompactedSource(mk.d4array(2, 3, 2, 2) { it + 1 }.transpose(1, 0, 3, 2))
    }

    @Test
    fun testScalarOpsOnOffsetSlice() {
        assertMatchesCompactedSource(mk.d2array(3, 3) { it + 1 }[1 until 3, 0 until 2])
        assertMatchesCompactedSource(mk.d3array(3, 3, 3) { it + 1 }[1 until 3, 1 until 3, 0 until 2])
    }

    @Test
    fun testScalarOpsOnSteppedSlice() {
        assertMatchesCompactedSource(mk.ndarrayOf(1, 2, 3, 4, 5)[sl.bounds..2])
        assertMatchesCompactedSource(mk.d2array(4, 4) { it + 1 }[sl.bounds..2, sl.bounds..3])
    }

    @Test
    fun testScalarOpsOnSingletonAxis() {
        assertMatchesCompactedSource(mk.d2array(1, 4) { it + 1 }.transpose())
    }

    // Detachment: the source buffer must be neither read stale nor written

    @Test
    fun testResultIsDetachedFromContiguousSource() {
        val a = mk.d2array(2, 3) { it }
        val r = 10 + a

        assertContentEquals(listOf(0, 1, 2, 3, 4, 5), a.toList())

        r[0, 0] = 999
        assertEquals(0, a[0, 0])

        a[0, 1] = 777
        assertEquals(11, r[0, 1])
    }

    @Test
    fun testResultIsDetachedFromViewAndItsBase() {
        val base = mk.d2array(3, 3) { it }
        val view = base[1 until 3, 0 until 2]
        assertFalse(view.consistent)
        val r = 10 - view

        assertContentEquals(listOf(0, 1, 2, 3, 4, 5, 6, 7, 8), base.toList())

        r[0, 0] = 999
        assertEquals(3, base[1, 0])

        base[1, 0] = 777
        assertContentEquals(listOf(999, 6, 4, 3), r.toList())
    }

    // Layout of the result

    @Test
    fun testResultIsContiguousAndUnbased() {
        val contiguous = mk.d2array(2, 3) { it }
        val view = contiguous.transpose()
        for (r in listOf(10 + contiguous, 10 - view, 10 * view, 10 / (1 + view))) {
            assertTrue(r.consistent)
            assertEquals(0, r.offset)
            assertNull(r.base)
        }
    }

    @Test
    fun testScalarOpsPreserveDimension() {
        val d1 = mk.ndarrayOf(1, 2, 3)
        val d2 = mk.d2array(2, 3) { it }
        val d3 = mk.d3array(2, 3, 4) { it }
        val d4 = mk.d4array(2, 3, 2, 2) { it }
        assertContentEquals(intArrayOf(3), (1 + d1).shape)
        assertContentEquals(intArrayOf(2, 3), (1 + d2).shape)
        assertContentEquals(intArrayOf(2, 3, 4), (1 + d3).shape)
        assertContentEquals(intArrayOf(2, 3, 2, 2), (1 + d4).shape)
        assertEquals(D2, (1 + d2).dim)
    }

    // Edge cases

    @Test
    fun testScalarOpsOnEmptyArray() {
        val empty = mk.ndarrayOf(1, 2, 3, 4)[3 until 1]
        assertTrue(empty.isEmpty())
        val r = 10 + empty
        assertTrue(r.isEmpty())
        assertContentEquals(empty.shape, r.shape)
        assertContentEquals(emptyList(), r.toList())
    }

    @Test
    fun testByteAndShortOpsTruncate() {
        val bytes = mk.ndarrayOf(100.toByte(), 100.toByte())
        assertContentEquals(listOf<Byte>(-56, -56), (100.toByte() + bytes).toList())

        val shorts = mk.ndarrayOf(30000.toShort(), 30000.toShort())
        assertContentEquals(listOf<Short>(-5536, -5536), (30000.toShort() + shorts).toList())
    }

    @Test
    fun testComplexDivisionByZeroThrows() {
        val zeros = mk.ndarray(listOf(ComplexFloat.zero))
        assertFailsWith<ArithmeticException> { ComplexFloat.one / zeros }
    }
}
