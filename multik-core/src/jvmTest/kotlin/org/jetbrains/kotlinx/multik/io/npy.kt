package org.jetbrains.kotlinx.multik.io

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.io.readNPY
import org.jetbrains.kotlinx.multik.api.io.readNPZ
import org.jetbrains.kotlinx.multik.api.io.write
import org.jetbrains.kotlinx.multik.api.io.writeNPZ
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D4
import kotlin.io.path.Path
import kotlin.io.path.deleteExisting
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NPYTest {

    private val oneDimensionArray = testNpy("a1d")
    private val fourDimensionArray = testNpy("a4d")
    private val npzArrays = testNpz("arrays")

    @Test
    fun `read 1-d array`() {
        val actual = Multik.readNPY<Long, D1>(oneDimensionArray)
        val expected = mk.ndarray(mk[1L, 2L, 3L])
        assertEquals(expected, actual)
    }

    @Test
    fun `read 4-d array`() {
        val actual = Multik.readNPY<Double, D4>(fourDimensionArray)
        val expected = mk.ndarray(
            mk[mk[mk[mk[1.0, 1.2571428571428571, 1.5142857142857142],
                mk[1.7714285714285714, 2.0285714285714285, 2.2857142857142856]],

                mk[mk[2.5428571428571427, 2.8, 3.057142857142857],
                    mk[3.314285714285714, 3.571428571428571, 3.8285714285714283]],

                mk[mk[4.085714285714285, 4.3428571428571425, 4.6],
                    mk[4.857142857142857, 5.114285714285714, 5.371428571428571]]],


                mk[mk[mk[5.628571428571428, 5.885714285714285, 6.142857142857142],
                    mk[6.3999999999999995, 6.657142857142857, 6.914285714285714]],

                    mk[mk[7.171428571428571, 7.428571428571428, 7.685714285714285],
                        mk[7.942857142857142, 8.2, 8.457142857142856]],

                    mk[mk[8.714285714285714, 8.971428571428572, 9.228571428571428],
                        mk[9.485714285714284, 9.742857142857142, 10.0]]]]
        )
        assertEquals(expected, actual)
    }

    @Test
    fun `read arrays from npz`() {
        val actual = Multik.readNPZ(npzArrays)
        val expected = listOf(
            mk.ndarray(mk[mk[1L, 2L, 3L], mk[4L, 5L, 6L]]),
            mk.ndarray(
                mk[mk[mk[1.0, 1.5294117647058822, 2.0588235294117645],
                    mk[2.588235294117647, 3.1176470588235294, 3.6470588235294117],
                    mk[4.176470588235294, 4.705882352941177, 5.235294117647059]],

                    mk[mk[5.764705882352941, 6.294117647058823, 6.823529411764706],
                        mk[7.352941176470589, 7.882352941176471, 8.411764705882353],
                        mk[8.941176470588236, 9.470588235294118, 10.0]]]
            )
        )
        assertEquals(expected, actual)
    }

    @Test
    fun `write 1-d array npy file`() {
        val a = mk.ndarray(intArrayOf(1, 2, 3, 11))
        val path = Path("src/jvmTest/resources/data/npy/testWrite1dArray.npy")
        mk.write(path, a)
        assertTrue(path.exists())
        assertEquals(a, mk.readNPY(path))
        path.deleteExisting()
    }

    @Test
    fun `write arrays to npz file`() {
        val a = mk.ndarray(intArrayOf(1, 2, 3, 4, 5, 6))
        val b = mk.ndarray(mk[mk[1.0, 2.0, 3.5], mk[7.1, 8.72, 0.329]])
        val path = Path("src/jvmTest/resources/data/npy/testNpzArrays.npz")
        mk.writeNPZ(path, a, b)
        assertTrue(path.exists())
        val (testA, testB) = mk.readNPZ(path)
        assertEquals(a, testA.asD1Array())
        assertEquals(b, testB.asD2Array())
        path.deleteExisting()
    }
}