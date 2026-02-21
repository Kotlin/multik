package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import kotlin.test.Test
import kotlin.test.assertEquals

class MemoryViewOperationsTest {

    // ---- helpers ----

    private fun <T> testAllScalarOperations(
        createView: (List<Int>) -> MemoryView<T>,
        fromInt: (Int) -> T
    ) {
        val initial = listOf(10, 20, 30)
        val scalar = 5

        // +=
        createView(initial).let { view ->
            view += fromInt(scalar)
            assertEquals(fromInt(15), view[0], "+= scalar [0]")
            assertEquals(fromInt(25), view[1], "+= scalar [1]")
            assertEquals(fromInt(35), view[2], "+= scalar [2]")
        }

        // -=
        createView(initial).let { view ->
            view -= fromInt(scalar)
            assertEquals(fromInt(5), view[0], "-= scalar [0]")
            assertEquals(fromInt(15), view[1], "-= scalar [1]")
            assertEquals(fromInt(25), view[2], "-= scalar [2]")
        }

        // *=
        createView(initial).let { view ->
            view *= fromInt(scalar)
            assertEquals(fromInt(50), view[0], "*= scalar [0]")
            assertEquals(fromInt(100), view[1], "*= scalar [1]")
            assertEquals(fromInt(150), view[2], "*= scalar [2]")
        }

        // /=
        createView(initial).let { view ->
            view /= fromInt(scalar)
            assertEquals(fromInt(2), view[0], "/= scalar [0]")
            assertEquals(fromInt(4), view[1], "/= scalar [1]")
            assertEquals(fromInt(6), view[2], "/= scalar [2]")
        }
    }

    private fun <T> testAllArrayOperations(
        createView: (List<Int>) -> MemoryView<T>,
        fromInt: (Int) -> T
    ) {
        val initial = listOf(10, 20, 30)
        val operand = listOf(2, 4, 6)

        // +=
        createView(initial).let { view ->
            view += createView(operand)
            assertEquals(fromInt(12), view[0], "+= array [0]")
            assertEquals(fromInt(24), view[1], "+= array [1]")
            assertEquals(fromInt(36), view[2], "+= array [2]")
        }

        // -=
        createView(initial).let { view ->
            view -= createView(operand)
            assertEquals(fromInt(8), view[0], "-= array [0]")
            assertEquals(fromInt(16), view[1], "-= array [1]")
            assertEquals(fromInt(24), view[2], "-= array [2]")
        }

        // *=
        createView(initial).let { view ->
            view *= createView(operand)
            assertEquals(fromInt(20), view[0], "*= array [0]")
            assertEquals(fromInt(80), view[1], "*= array [1]")
            assertEquals(fromInt(180), view[2], "*= array [2]")
        }

        // /=
        createView(initial).let { view ->
            view /= createView(operand)
            assertEquals(fromInt(5), view[0], "/= array [0]")
            assertEquals(fromInt(5), view[1], "/= array [1]")
            assertEquals(fromInt(5), view[2], "/= array [2]")
        }
    }

    // ---- Byte ----

    @Test
    fun byteScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewByteArray(ByteArray(it.size) { i -> it[i].toByte() }) },
        fromInt = { it.toByte() }
    )

    @Test
    fun byteArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewByteArray(ByteArray(it.size) { i -> it[i].toByte() }) },
        fromInt = { it.toByte() }
    )

    // ---- Short ----

    @Test
    fun shortScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewShortArray(ShortArray(it.size) { i -> it[i].toShort() }) },
        fromInt = { it.toShort() }
    )

    @Test
    fun shortArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewShortArray(ShortArray(it.size) { i -> it[i].toShort() }) },
        fromInt = { it.toShort() }
    )

    // ---- Int ----

    @Test
    fun intScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewIntArray(IntArray(it.size) { i -> it[i] }) },
        fromInt = { it }
    )

    @Test
    fun intArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewIntArray(IntArray(it.size) { i -> it[i] }) },
        fromInt = { it }
    )

    // ---- Long ----

    @Test
    fun longScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewLongArray(LongArray(it.size) { i -> it[i].toLong() }) },
        fromInt = { it.toLong() }
    )

    @Test
    fun longArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewLongArray(LongArray(it.size) { i -> it[i].toLong() }) },
        fromInt = { it.toLong() }
    )

    // ---- Float ----

    @Test
    fun floatScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewFloatArray(FloatArray(it.size) { i -> it[i].toFloat() }) },
        fromInt = { it.toFloat() }
    )

    @Test
    fun floatArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewFloatArray(FloatArray(it.size) { i -> it[i].toFloat() }) },
        fromInt = { it.toFloat() }
    )

    // ---- Double ----

    @Test
    fun doubleScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewDoubleArray(DoubleArray(it.size) { i -> it[i].toDouble() }) },
        fromInt = { it.toDouble() }
    )

    @Test
    fun doubleArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewDoubleArray(DoubleArray(it.size) { i -> it[i].toDouble() }) },
        fromInt = { it.toDouble() }
    )

    // ---- ComplexFloat ----

    @Test
    fun complexFloatScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewComplexFloatArray(ComplexFloatArray(it.size) { i -> ComplexFloat(it[i], 0) }) },
        fromInt = { ComplexFloat(it, 0) }
    )

    @Test
    fun complexFloatArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewComplexFloatArray(ComplexFloatArray(it.size) { i -> ComplexFloat(it[i], 0) }) },
        fromInt = { ComplexFloat(it, 0) }
    )

    // ---- ComplexDouble ----

    @Test
    fun complexDoubleScalarOperations() = testAllScalarOperations(
        createView = { MemoryViewComplexDoubleArray(ComplexDoubleArray(it.size) { i -> ComplexDouble(it[i]) }) },
        fromInt = { ComplexDouble(it) }
    )

    @Test
    fun complexDoubleArrayOperations() = testAllArrayOperations(
        createView = { MemoryViewComplexDoubleArray(ComplexDoubleArray(it.size) { i -> ComplexDouble(it[i]) }) },
        fromInt = { ComplexDouble(it) }
    )
}
