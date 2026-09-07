package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.api.zeros
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFailsWith

/**
 * Pins the error-handling policy: an invalid argument is an [IllegalArgumentException], a missing
 * implementation is an [UnsupportedOperationException], and every message names the value that
 * caused the failure so the caller can act on it without reading the source.
 */
class ErrorHandlingTest {

    @Test
    fun testInvalidShapeArgumentsAreIllegalArgument() {
        val a = mk.ndarrayOf(1, 2, 3, 4, 5, 6)

        assertContains(assertFailsWith<IllegalArgumentException> { a.reshape(4, 2) }.message.orEmpty(), "(4, 2)")
        assertContains(assertFailsWith<IllegalArgumentException> { a.reshape(0) }.message.orEmpty(), "0")
        assertFailsWith<IllegalArgumentException> { a.reshape(2, 3, 4) }
    }

    @Test
    fun testInvalidAxisArgumentsAreIllegalArgument() {
        val a = mk.zeros<Double>(1, 3)

        assertContains(assertFailsWith<IllegalArgumentException> { a.squeeze(1) }.message.orEmpty(), "(1, 3)")
        assertFailsWith<IllegalArgumentException> { a.squeeze(7) }
        assertFailsWith<IllegalArgumentException> { a.unsqueeze(9) }
        assertFailsWith<IllegalArgumentException> { a.transpose(0, 0) }
    }

    @Test
    fun testUnknownDataTypeCodeIsIllegalArgument() {
        assertContains(assertFailsWith<IllegalArgumentException> { DataType.of(0) }.message.orEmpty(), "0")
        assertContains(assertFailsWith<IllegalArgumentException> { DataType.of(9) }.message.orEmpty(), "9")
        assertFailsWith<IllegalArgumentException> { DataType.of<String?>(null) }
    }

    @Test
    fun testConvertingToANonNumericDataTypeIsIllegalArgument() {
        val e = assertFailsWith<IllegalArgumentException> {
            (5 as Number).toPrimitiveType<Int>(DataType.ComplexFloatDataType)
        }
        assertContains(e.message.orEmpty(), "ComplexFloatDataType")
    }

    @Test
    fun testReadingAMemoryViewAsTheWrongArrayTypeIsUnsupported() {
        val data = mk.ndarrayOf(1, 2, 3).data

        val e = assertFailsWith<UnsupportedOperationException> { data.getDoubleArray() }
        assertContains(e.message.orEmpty(), "IntDataType")
        assertFailsWith<UnsupportedOperationException> { data.getComplexFloatArray() }
        assertFailsWith<UnsupportedOperationException> { data.getByteArray() }
    }

    @Test
    fun testUnsupportedRangeTypeIsIllegalArgument() {
        val range = object : ClosedRange<Int> {
            override val start: Int get() = 0
            override val endInclusive: Int get() = 2
        }
        assertFailsWith<IllegalArgumentException> { range.toSlice() }
    }
}
