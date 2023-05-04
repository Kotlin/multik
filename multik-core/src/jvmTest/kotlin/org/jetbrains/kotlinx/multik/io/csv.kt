package org.jetbrains.kotlinx.multik.io

import org.jetbrains.kotlinx.multik.api.io.read
import org.jetbrains.kotlinx.multik.api.io.write
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import kotlin.io.path.Path
import kotlin.io.path.deleteExisting
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue


class CSVTest {

    @Test
    fun `read 1-d array`() {
        println(mk.read<Double, D1>(testCsv("a1d")))
    }

    @Test
    fun `read 2-d array`() {
        println(mk.read<Double, D2>(testCsv("a2d")))
    }

    @Test
    fun `read array with complex numbers`() {
        TODO()
    }

    @Test
    fun `write simple array`() {
        val a = mk.ndarray(intArrayOf(1, 2, 3, 7))
        val path = Path("src/jvmTest/resources/data/csv/testWrite1dArray.csv")
        mk.write(path, a)
        assertTrue(path.exists())
        assertEquals(a, mk.read(path))
        path.deleteExisting()
    }

    @Test
    fun `write array with complex numbers`() {
        val a = mk.ndarray(complexDoubleArrayOf(1.0 + 2.0.i, 3.0 + 4.0.i, 5.0 + 6.0.i))
        val path = Path("src/jvmTest/resources/data/csv/testWrite1dArray.csv")
        mk.write(path, a)
        assertTrue(path.exists())
        assertEquals(a, mk.read(path))
        path.deleteExisting()
    }
}