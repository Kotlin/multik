package org.jetbrains.multik.iterable

import org.jetbrains.multik.api.*
import org.jetbrains.multik.core.*
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NdarrayKtTest {

    @BeforeTest
    fun loadLibrary() {
        System.load("/Users/pavel.gorgulov/Projects/main_project/multik/src/jni_multik/cmake-build-debug/libjni_multik.dylib")
    }

    @Test
    fun distinctTest() {
        val data = mk.ndarrayOf(1, 2, 3, 1, 2, 3)
        assertEquals(mk.ndarrayOf(1, 2, 3), data.distinct())
    }

    @Test
    fun distinctByTest() {
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
    fun dropTest() {
        val data = mk.arange<Float>(10)
        assertEquals(mk.arange(start = 5, stop = 10), data.drop(5))
    }

    @Test
    fun dropWhileTest() {
        val data = mk.arange<Long>(50)
        assertEquals(mk.arange(45, 50, 1), data.dropWhile { it < 45 })
    }

    @Test
    fun filterTest() {
        val data = mk.arange<Int>(10, 30, 1)
        val actual = data.filter { it in 23..27 }
        assertEquals(mk.arange(23, 28, 1), actual)
    }

    @Test
    fun filterIndexedTest() {
        val data = mk.arange<Float>(10)
        data[0] = 10f
        assertEquals(mk.arange(6, 10, 1), data.filterIndexed { index, fl -> (index != 0) && (fl > 5) })
    }

    @Test
    fun filterIndexedToTest() {
        val data = mk.arange<Float>(10)
        data[0] = 10f
        val destination = mk.empty<Float, D1>(4)
        data.filterIndexedTo(destination) { index, fl -> (index != 0) && (fl > 5) }
        assertEquals(mk.arange(6, 10, 1), destination)
    }


    @Test
    fun filterToTest() {
        val data = mk.arange<Int>(10, 30, 1)
        val destination = mk.empty<Int, D1>(5)
        data.filterTo(destination) { it in 23..27 }
        assertEquals(mk.arange(23, 28, 1), destination)
    }

    @Test
    fun groupingNdarrayByTest() {
        val data = mk.d3array(2, 2, 2) { it }
        val expected1 = mapOf(0 to mk.ndarrayOf(0, 2, 4, 6), 1 to mk.ndarrayOf(1, 3, 5, 7))
        assertEquals(expected1, data.groupNdarrayBy { it % 2 })

        val expected2 = mapOf(0 to mk.ndarrayOf(0f, 2f, 4f, 6f), 1 to mk.ndarrayOf(1f, 3f, 5f, 7f))
        assertEquals(expected2, data.groupNdarrayBy({ it % 2 }, { it.toFloat() }))
    }

    @Test
    fun mapTest() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(1, 4, 9, 16), data.map { it * it })
    }

    @Test
    fun mapIndexedTest() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(0, 2, 6, 12), data.mapIndexed { idx, value -> value * idx })
    }
}
