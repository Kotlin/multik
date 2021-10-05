/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.iterable

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IterableNDArrayTest {

    @Test
    fun `test of function associate`() {
        val charCodesNDArray = mk.ndarray(mk[72, 69, 76, 76, 79])

        val actual = mutableMapOf<Int, Char>()
        charCodesNDArray.associateTo(actual) { it to it.toChar() }
        val expected = mapOf(72 to 'H', 69 to 'E', 76 to 'L', 79 to 'O')
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function associateBy`() {
        val charCodesNDArray = mk.ndarray(mk[72, 69, 76, 76, 79])

        val actual = mutableMapOf<Char, Int>()
        charCodesNDArray.associateByTo(actual) { it.toChar() }
        val expected = mapOf('H' to 72, 'E' to 69, 'L' to 76, 'O' to 79)
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function associateBy with transform`() {
        val charCodesNDArray = mk.ndarray(mk[65, 65, 66, 67, 68, 69])

        val actual = mutableMapOf<Char, Char>()
        charCodesNDArray.associateByTo(actual, { it.toChar() }, { (it + 32).toChar() })
        val expected = mapOf('A' to 'a', 'B' to 'b', 'C' to 'c', 'D' to 'd', 'E' to 'e')
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function associateWith`() {
        val numbers = mk.ndarray(mk[1, 2, 3, 4])

        val actual = mutableMapOf<Int, Int>()
        numbers.associateWithTo(actual) { it * it }
        val expected = mapOf(1 to 1, 2 to 4, 3 to 9, 4 to 16)
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function average`() {
        val array = intArrayOf(12, 49, 23, 4, 35, 60, 33)

        val ndarray = mk.ndarray(array)

        val actual = ndarray.average()
        val expected = array.average()
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function chunked`() {
        val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        val actual = a.chunked(3)
        val expected = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6], mk[7, 8, 9], mk[10, 0, 0]])
        assertEquals(expected, actual)
    }

    @Test
    fun `test of function contains`() {
        val ndarray = mk.d2array(5, 5) { it - 3f }
        assertTrue(-1f in ndarray)
        assertFalse(25f in ndarray)
    }

    @Test
    fun `test of function count`() {
        val ndarray = mk.ndarray(mk[1, 1, 2, 3, 4, 5, 2, 10])
        assertEquals(1, ndarray.count { it == 3 })
        assertEquals(4, ndarray.count { it % 2 == 0 })
    }

    @Test
    fun `test distinct`() {
        val data = mk.ndarrayOf(1, 2, 3, 1, 2, 3)
        assertEquals(mk.ndarrayOf(1, 2, 3), data.distinct())
    }

    @Test
    fun `test distinctBy`() {
        val data = mk.ndarrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).distinctBy {
            if (it <= 3.0)
                it * it
            else {
                it
            }
        }
        assertEquals(mk.ndarrayOf(1.0, 2.0, 3.0, 5.0, 6.0), data)
    }

    @Test
    fun `test drop`() {
        val data = mk.arange<Float>(10)
        assertEquals(mk.arange(start = 5, stop = 10), data.drop(5))
        assertEquals(mk.arange(start = 0, 8), data.drop(-2))
    }

    @Test
    fun `test dropWhile`() {
        val data = mk.arange<Long>(50)
        assertEquals(mk.arange(45, 50, 1), data.dropWhile { it < 45 })
    }

    @Test
    fun `test filter`() {
        val data = mk.arange<Int>(10, 30, 1)
        val actual = data.filter { it in 23..27 }
        assertEquals(mk.arange(23, 28, 1), actual)
    }

    @Test
    fun `test filterIndexed`() {
        val data = mk.arange<Float>(10)
        data[0] = 10f
        assertEquals(mk.arange(6, 10, 1), data.filterIndexed { index, fl -> (index != 0) && (fl > 5) })
    }

    @Test
    fun `test filterNot`() {
        val list = listOf(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
        val data = mk.ndarray(list)
        val actual = data.filterNot { it in 23..27 }
        val expectedList = list.filterNot { it in 23..27 }
        assertEquals(mk.ndarray(expectedList), actual)
    }

    @Test
    fun `test find`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        assertEquals(list.find { it % 2 != 0 }, ndarray.find { it % 2 != 0 })
        assertEquals(list.findLast { it % 2 == 0 }, ndarray.findLast { it % 2 == 0 })
    }

    @Test
    fun `test first and firstOrNull with predicate`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        println(list.first { it % 2 != 0 })
        assertEquals(list.first { it % 2 != 0 }, ndarray.first { it % 2 != 0 })
        assertEquals(list.firstOrNull { it % 10 == 0 }, ndarray.firstOrNull { it % 10 == 0 })
    }

    @Test
    fun `test flatMap`() {
        val list = listOf(0, 1, 2, 3)
        val ndarray = mk.ndarray(list, 2, 2)
        assertEquals(
            mk.ndarray(list.flatMap { listOf(it, it + 1, it + 2) }),
            ndarray.flatMap { listOf(it, it + 1, it + 2) })
    }

    @Test
    fun `test flatMapIndexed`() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(list)
        assertEquals(
            mk.ndarray(list.flatMapIndexed { i, e -> listOf(e, i) }),
            ndarray.flatMapIndexed { i: Int, e -> listOf(e, i) })
    }

    @Test
    fun `test fold`() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(list)
        assertEquals(list.fold(3, Int::times), ndarray.fold(3, Int::times))
    }

    @Test
    fun `test foldIndexed`() {
        val list = listOf(1, 2, 3, 4, 5)
        val ndarray = mk.ndarray(list)
        val actual = ndarray.foldIndexed(Pair(1, 1)) { index, acc: Pair<Int, Int>, i: Int ->
            Pair(
                acc.first + index,
                acc.second * i
            )
        }
        val expected = list.foldIndexed(Pair(1, 1)) { index, acc: Pair<Int, Int>, i: Int ->
            Pair(
                acc.first + index,
                acc.second * i
            )
        }
        assertEquals(expected, actual)
    }

    @Test
    fun `test groupNDArrayBy`() {
        val data = mk.d3array(2, 2, 2) { it }
        val expected1 = mapOf(0 to mk.ndarrayOf(0, 2, 4, 6), 1 to mk.ndarrayOf(1, 3, 5, 7))
        assertEquals(expected1, data.groupNDArrayBy { it % 2 })

        val expected2 = mapOf(0 to mk.ndarrayOf(0f, 2f, 4f, 6f), 1 to mk.ndarrayOf(1f, 3f, 5f, 7f))
        assertEquals(expected2, data.groupNDArrayBy({ it % 2 }, { it.toFloat() }))
    }

    @Test
    fun `test intersect`() {
        val list = listOf(1, 3, 4, 5, 6, 10)
        val ndarray = mk.ndarray(list)
        val list2 = listOf(2, 3, 5, 7, 6, 11)
        val expected = list intersect list2
        val actual = ndarray intersect list2
        assertEquals(expected, actual)
    }

    @Test
    fun `test last`() {
        val ndarray = mk.ndarray(mk[mk[2, 3, -17], mk[10, 23, 33]])
        assertEquals(33, ndarray.last())
    }

    @Test
    fun `test last with predicate`() {
        val list = listOf(1, 2, 3, -12, 42, 33, 89)
        val ndarray = mk.ndarray(list)
        println(list.last { it % 2 == 0 })
        println(ndarray.last { it % 2 == 0 })
    }

    @Test
    fun `test map for scalar ndarray`() {
        val a = mk.ndarray(mk[mk[mk[3.2]]])
        assertEquals(mk.ndarray(mk[mk[mk[3]]]), a.map { it.toInt() })
    }

    @Test
    fun `test map`() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(1, 4, 9, 16), data.map { it * it })
    }

    @Test
    fun `test mapIndexed`() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(0, 2, 6, 12), data.mapIndexed { idx: Int, value -> value * idx })
        val ndarray = mk.ndarrayOf(1, 2, 3, 4).reshape(2, 2)
        ndarray.mapMultiIndexed { idx: IntArray, value -> value * (idx[0] xor idx[1]) }
    }

    @Test
    fun `test max`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.maxOrNull(), ndarray.max())
    }

    @Test
    fun `test maxBy`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.maxByOrNull { -it }, ndarray.maxBy { -it })
    }

    @Test
    fun `test min`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.minOrNull(), ndarray.min())

    }

    @Test
    fun `test minBy`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.minByOrNull { -it }, ndarray.minBy { -it })
    }

    @Test
    fun `test partition`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        val (h, t) = ndarray.partition { it % 2 == 0 }
        val (lH, lT) = list.partition { it % 2 == 0 }
        assertEquals(mk.ndarray(lH), h)
        assertEquals(mk.ndarray(lT), t)
    }

    @Test
    fun `test sort`() {
        //TODO(assert)
        val intArray = intArrayOf(42, 42, 23, 1, 23, 4, 10, 14, 3, 7, 25, 16, 2, 1, 37)
        val ndarray = mk.ndarray(intArray, 3, 5)
        val sortedNDArray = ndarray.sorted()
        sortedNDArray[2, 2] = 1000

    }

    @Test
    fun `test reduce`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        val expected = list.reduce { acc, i -> acc + i / 2 }
        val actual = ndarray.reduce { acc, i -> acc + i / 2 }
        assertEquals(expected, actual)
    }

    @Test
    fun `test reversed`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(list, 2, 4)
        val expected = mk.ndarray(list.reversed(), 2, 4)
        assertEquals(expected, ndarray.reversed())
    }

    @Test
    fun `test scan`() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4, 5, 6])
        println(ndarray.scan(10) { acc: Int, i: Int -> acc + i })
    }

    @Test
    fun `test minimum`() {
        val ndarray1 = mk.ndarray(mk[2, 3, 4])
        val ndarray2 = mk.ndarray(mk[1, 5, 2])
        assertEquals(mk.ndarray(mk[1, 3, 2]), ndarray1.minimum(ndarray2))
    }

    @Test
    fun `test maximum`() {
        val ndarray1 = mk.ndarray(mk[2, 3, 4])
        val ndarray2 = mk.ndarray(mk[1, 5, 2])
        assertEquals(mk.ndarray(mk[2, 5, 4]), ndarray1.maximum(ndarray2))
    }
}
