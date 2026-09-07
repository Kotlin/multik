package org.jetbrains.kotlinx.multik.io

import org.jetbrains.kotlinx.multik.api.io.read
import org.jetbrains.kotlinx.multik.api.io.write
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.complexFloatArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import kotlin.io.path.Path
import kotlin.io.path.deleteIfExists
import kotlin.io.path.exists
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class IOErrorsTest {

    @Test
    fun testWritingComplexArraysToNpyIsRejected() {
        val path = Path("src/jvmTest/resources/data/npy/testComplexReject.npy")

        // Both complex types must be rejected: the guard used to name ComplexFloat twice, so
        // ComplexDouble slipped through into `writeNPY`.
        val complexFloat = mk.ndarray(complexFloatArrayOf(1f + 2f.i, 3f + 4f.i))
        val complexDouble = mk.ndarray(complexDoubleArrayOf(1.0 + 2.0.i, 3.0 + 4.0.i))

        assertContains(
            assertFailsWith<IllegalArgumentException> { mk.write(path, complexFloat) }.message.orEmpty(),
            "ComplexFloatDataType"
        )
        assertContains(
            assertFailsWith<IllegalArgumentException> { mk.write(path, complexDouble) }.message.orEmpty(),
            "ComplexDoubleDataType"
        )
        path.deleteIfExists()
    }

    @Test
    fun testReadingNpyWithAComplexDataTypeIsRejected() {
        val e = assertFailsWith<IllegalArgumentException> {
            mk.read<Any, D1>(Path(testNpy("a1d")), DataType.ComplexDoubleDataType, D1)
        }
        assertContains(e.message.orEmpty(), "ComplexDoubleDataType")
    }

    @Test
    fun testWritingA2dArrayToCsvIsAllowed() {
        // `read` accepts D1 and D2 and `writeCSV` handles both, but `write` used to reject D2.
        val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val path = Path("src/jvmTest/resources/data/csv/testWrite2dArray.csv")

        mk.write(path, a)
        assertTrue(path.exists())
        assertEquals(a, mk.read<Double, D2>(path))
        path.deleteIfExists()
    }

    @Test
    fun testWritingA3dArrayToCsvIsRejected() {
        val path = Path("src/jvmTest/resources/data/csv/testWrite3dArray.csv")
        val e = assertFailsWith<IllegalArgumentException> { mk.write(path, mk.zeros<Double>(2, 2, 2)) }
        assertContains(e.message.orEmpty(), "dimension 3")
        path.deleteIfExists()
    }

    @Test
    fun testUnknownFormatIsRejected() {
        val path = Path("src/jvmTest/resources/data/csv/testUnknownFormat.parquet")

        val write = assertFailsWith<IllegalArgumentException> { mk.write(path, mk.zeros<Double>(2)) }
        assertContains(write.message.orEmpty(), "parquet")

        path.writeText("1.0\n2.0\n")
        val read = assertFailsWith<IllegalArgumentException> { mk.read<Double, D1>(path) }
        assertContains(read.message.orEmpty(), "parquet")
        assertContains(read.message.orEmpty(), "npy")
        path.deleteIfExists()
    }
}
