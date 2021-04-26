package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SliceTest {

    @Test
    fun testSlice1D() {
        val a = mk.ndarrayOf(1, 2, 3, 4)


        assertEquals(mk.ndarrayOf(2, 3), a[1..3])
        assertTrue(a[3..1].isEmpty())
        assertEquals(mk.ndarrayOf(1, 3), a[sl.bounds..2])
    }

    @Test
    fun testSlice2D() {
        val a = mk.d2array(3, 3) { it }

        assertEquals(mk.ndarray(mk[mk[3, 4, 5], mk[6, 7, 8]]), a[1..3])
        assertEquals(mk.ndarray(mk[mk[2], mk[5]]), a[0..2, 2..3])
        assertEquals(mk.ndarrayOf(3), a[1, 0..1])
        assertEquals(mk.ndarrayOf(5), a[1..2, 2])
    }

    @Test
    fun testSlice3D() {
        val a = mk.d3array(3, 3, 3) { it }

        assertEquals(mk.d3array(2, 3, 3) { it + 9 }, a[1..3])
        assertEquals(
            mk.ndarray(mk[mk[mk[3, 4, 5], mk[6, 7, 8]], mk[mk[12, 13, 14], mk[15, 16, 17]]]),
            a[0..2, 1..3, sl.bounds]
        )
        assertEquals(mk.ndarrayOf(15, 16), a[1, 2, 0..2])
        assertEquals(mk.ndarrayOf(11, 14), a[1, 0..2, 2])
        assertEquals(mk.ndarrayOf(5, 14), a[0..2, 1, 2])
        assertEquals(mk.ndarray(mk[mk[4, 7], mk[13, 16]]), a[0..2, 1..3, 1])
        assertEquals(mk.ndarray(mk[mk[4, 5], mk[13, 14]]), a[0..2, 1, 1..3])
        assertEquals(mk.ndarray(mk[mk[10, 11], mk[13, 14]]), a[1, 0..2, 1..3])
    }

    @Test
    fun testSlice4D() {
        val a = mk.d4array(2, 2, 2, 2) { it }
        assertEquals(a, a[sl.bounds])
        assertEquals(mk.ndarray(mk[mk[mk[mk[5]]]]), a[0..1, 1..2, 0..1, 1..2])
        assertEquals(mk.ndarray(mk[mk[mk[5]]]), a[0..1, 1, 0..1, 1..2..1])
        assertEquals(mk.ndarray(mk[mk[7]]), a[0..1, 1, 1, 1..2])
        assertEquals(mk.ndarrayOf(6), a[0..1, 1, 1, 0])
    }
}