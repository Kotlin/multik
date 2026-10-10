package org.jetbrains.kotlinx.multik.iterable

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.associateByTo
import org.jetbrains.kotlinx.multik.ndarray.operations.associateTo
import org.jetbrains.kotlinx.multik.ndarray.operations.associateWithTo
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.chunked
import org.jetbrains.kotlinx.multik.ndarray.operations.contains
import org.jetbrains.kotlinx.multik.ndarray.operations.count
import org.jetbrains.kotlinx.multik.ndarray.operations.distinct
import org.jetbrains.kotlinx.multik.ndarray.operations.distinctBy
import org.jetbrains.kotlinx.multik.ndarray.operations.drop
import org.jetbrains.kotlinx.multik.ndarray.operations.dropWhile
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.operations.filter
import org.jetbrains.kotlinx.multik.ndarray.operations.filterIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.filterMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.filterNot
import org.jetbrains.kotlinx.multik.ndarray.operations.find
import org.jetbrains.kotlinx.multik.ndarray.operations.findLast
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.jetbrains.kotlinx.multik.ndarray.operations.firstOrNull
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.operations.flatMap
import org.jetbrains.kotlinx.multik.ndarray.operations.flatMapIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.flatMapMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.fold
import org.jetbrains.kotlinx.multik.ndarray.operations.foldIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.groupNDArrayBy
import org.jetbrains.kotlinx.multik.ndarray.operations.intersect
import org.jetbrains.kotlinx.multik.ndarray.operations.last
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.mapIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.mapMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.maxBy
import org.jetbrains.kotlinx.multik.ndarray.operations.maximum
import org.jetbrains.kotlinx.multik.ndarray.operations.min
import org.jetbrains.kotlinx.multik.ndarray.operations.minBy
import org.jetbrains.kotlinx.multik.ndarray.operations.minimum
import org.jetbrains.kotlinx.multik.ndarray.operations.partition
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import org.jetbrains.kotlinx.multik.ndarray.operations.reversed
import org.jetbrains.kotlinx.multik.ndarray.operations.scan
import org.jetbrains.kotlinx.multik.ndarray.operations.sorted
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IterableNDArrayTest {

    @Test
    fun `test_of_function_associate`() {
        val charCodesNDArray = mk.ndarray(mk[72, 69, 76, 76, 79])

        val actual = mutableMapOf<Int, Char>()
        charCodesNDArray.associateTo(actual) { it to it.toChar() }
        val expected = mapOf(72 to 'H', 69 to 'E', 76 to 'L', 79 to 'O')
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_associateBy`() {
        val charCodesNDArray = mk.ndarray(mk[72, 69, 76, 76, 79])

        val actual = mutableMapOf<Char, Int>()
        charCodesNDArray.associateByTo(actual) { it.toChar() }
        val expected = mapOf('H' to 72, 'E' to 69, 'L' to 76, 'O' to 79)
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_associateBy_with_transform`() {
        val charCodesNDArray = mk.ndarray(mk[65, 65, 66, 67, 68, 69])

        val actual = mutableMapOf<Char, Char>()
        charCodesNDArray.associateByTo(actual, { it.toChar() }, { (it + 32).toChar() })
        val expected = mapOf('A' to 'a', 'B' to 'b', 'C' to 'c', 'D' to 'd', 'E' to 'e')
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_associateWith`() {
        val numbers = mk.ndarray(mk[1, 2, 3, 4])

        val actual = mutableMapOf<Int, Int>()
        numbers.associateWithTo(actual) { it * it }
        val expected = mapOf(1 to 1, 2 to 4, 3 to 9, 4 to 16)
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_average`() {
        val array = intArrayOf(12, 49, 23, 4, 35, 60, 33)

        val ndarray = mk.ndarray(array)

        val actual = ndarray.average()
        val expected = array.average()
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_chunked`() {
        val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        val actual = a.chunked(3)
        val expected = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6], mk[7, 8, 9], mk[10, 0, 0]])
        assertEquals(expected, actual)
    }

    @Test
    fun `test_of_function_contains`() {
        val ndarray = mk.d2array(5, 5) { it - 3f }
        assertTrue(-1f in ndarray)
        assertFalse(25f in ndarray)
    }

    @Test
    fun `test_of_function_count`() {
        val ndarray = mk.ndarray(mk[1, 1, 2, 3, 4, 5, 2, 10])
        assertEquals(1, ndarray.count { it == 3 })
        assertEquals(4, ndarray.count { it % 2 == 0 })
    }

    @Test
    fun `test_distinct`() {
        val data = mk.ndarrayOf(1, 2, 3, 1, 2, 3)
        assertEquals(mk.ndarrayOf(1, 2, 3), data.distinct())
    }

    @Test
    fun `test_distinctBy`() {
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
    fun `test_drop`() {
        val data = mk.arange<Float>(10)
        assertEquals(mk.arange(start = 5, stop = 10), data.drop(5))
        assertEquals(mk.arange(start = 0, 8), data.drop(-2))
    }

    @Test
    fun `test_dropWhile`() {
        val data = mk.arange<Long>(50)
        assertEquals(mk.arange(45, 50, 1), data.dropWhile { it < 45 })
    }

    @Test
    fun `test_dropWhile_empty_result`() {
        val data = mk.arange<Long>(10)
        val actual = data.dropWhile { it < 100 }
        assertEquals(0, actual.size)
        assertEquals(DataType.LongDataType, actual.dtype)

        val emptyData = mk.zeros<Double>(0)
        val actualEmpty = emptyData.dropWhile { it < 10.0 }
        assertEquals(0, actualEmpty.size)
        assertEquals(DataType.DoubleDataType, actualEmpty.dtype)
    }

    @Test
    fun `test_filter`() {
        val data = mk.arange<Int>(10, 30, 1)
        val actual = data.filter { it in 23..27 }
        assertEquals(mk.arange(23, 28, 1), actual)
    }

    @Test
    fun `test_filterIndexed`() {
        val data = mk.arange<Float>(10)
        data[0] = 10f
        assertEquals(mk.arange(6, 10, 1), data.filterIndexed { index, fl -> (index != 0) && (fl > 5) })
    }

    @Test
    fun `test_filterNot`() {
        val list = listOf(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
        val data = mk.ndarray(list)
        val actual = data.filterNot { it in 23..27 }
        val expectedList = list.filterNot { it in 23..27 }
        assertEquals(mk.ndarray(expectedList), actual)
    }

    @Test
    fun `test_filter_empty_result`() {
        val data = mk.arange<Int>(10, 30, 1)
        val actual = data.filter { it > 100 }
        assertEquals(0, actual.size)
        assertEquals(DataType.IntDataType, actual.dtype)

        val actualIndexed = data.filterIndexed { _, v -> v > 100 }
        assertEquals(0, actualIndexed.size)
        assertEquals(DataType.IntDataType, actualIndexed.dtype)

        val actualMultiIndexed = data.filterMultiIndexed { _, v -> v > 100 }
        assertEquals(0, actualMultiIndexed.size)
        assertEquals(DataType.IntDataType, actualMultiIndexed.dtype)

        val actualNot = data.filterNot { it < 100 }
        assertEquals(0, actualNot.size)
        assertEquals(DataType.IntDataType, actualNot.dtype)

        val emptyData = mk.zeros<Int>(0)
        val actualDistinctBy = emptyData.distinctBy { it }
        assertEquals(0, actualDistinctBy.size)
        assertEquals(DataType.IntDataType, actualDistinctBy.dtype)

        val (firstEmpty, secondAll) = data.partition { it > 100 }
        assertEquals(0, firstEmpty.size)
        assertEquals(data.size, secondAll.size)
        assertEquals(DataType.IntDataType, firstEmpty.dtype)
        assertEquals(DataType.IntDataType, secondAll.dtype)
    }

    @Test
    fun `test_filter_empty_across_dtypes`() {
        val dbl = mk.ndarray(listOf(1.0, 2.0, 3.0))
        val emptyDbl = dbl.filter { it > 10.0 }
        assertEquals(0, emptyDbl.size)
        assertEquals(DataType.DoubleDataType, emptyDbl.dtype)

        val flt = mk.ndarray(listOf(1.0f, 2.0f, 3.0f))
        val emptyFlt = flt.filterNot { it < 10.0f }
        assertEquals(0, emptyFlt.size)
        assertEquals(DataType.FloatDataType, emptyFlt.dtype)

        val lng = mk.ndarray(listOf(1L, 2L, 3L))
        val (emptyLng, allLng) = lng.partition { it > 10L }
        assertEquals(0, emptyLng.size)
        assertEquals(3, allLng.size)
        assertEquals(DataType.LongDataType, emptyLng.dtype)
        assertEquals(DataType.LongDataType, allLng.dtype)
    }

    @Test
    fun `test_toNDArray_empty_iterable`() {
        val emptyInt = emptyList<Int>().toNDArray()
        assertEquals(0, emptyInt.size)
        assertEquals(DataType.IntDataType, emptyInt.dtype)

        val emptyDouble = emptyList<Double>().toNDArray()
        assertEquals(0, emptyDouble.size)
        assertEquals(DataType.DoubleDataType, emptyDouble.dtype)

        val emptyFloat = emptyList<Float>().toNDArray()
        assertEquals(0, emptyFloat.size)
        assertEquals(DataType.FloatDataType, emptyFloat.dtype)

        val emptyByte = emptyList<Byte>().toNDArray()
        assertEquals(0, emptyByte.size)
        assertEquals(DataType.ByteDataType, emptyByte.dtype)

        val emptyShort = emptyList<Short>().toNDArray()
        assertEquals(0, emptyShort.size)
        assertEquals(DataType.ShortDataType, emptyShort.dtype)

        val emptyLong = emptyList<Long>().toNDArray()
        assertEquals(0, emptyLong.size)
        assertEquals(DataType.LongDataType, emptyLong.dtype)

        val emptyComplexFloat = emptyList<ComplexFloat>().toNDArray()
        assertEquals(0, emptyComplexFloat.size)
        assertEquals(DataType.ComplexFloatDataType, emptyComplexFloat.dtype)

        val emptyComplexDouble = emptyList<ComplexDouble>().toNDArray()
        assertEquals(0, emptyComplexDouble.size)
        assertEquals(DataType.ComplexDoubleDataType, emptyComplexDouble.dtype)

        val emptySet = emptySet<Double>().toNDArray()
        assertEquals(0, emptySet.size)
        assertEquals(DataType.DoubleDataType, emptySet.dtype)

        val empty2D = emptyList<List<Int>>().toNDArray()
        assertEquals(0, empty2D.size)
        assertEquals(DataType.IntDataType, empty2D.dtype)
        assertEquals(2, empty2D.dim.d)

        val empty3D = emptyList<List<List<Double>>>().toNDArray()
        assertEquals(0, empty3D.size)
        assertEquals(DataType.DoubleDataType, empty3D.dtype)
        assertEquals(3, empty3D.dim.d)

        val empty4D = emptyList<List<List<List<Float>>>>().toNDArray()
        assertEquals(0, empty4D.size)
        assertEquals(DataType.FloatDataType, empty4D.dtype)
        assertEquals(4, empty4D.dim.d)

        val emptyArrayInt = emptyArray<IntArray>().toNDArray()
        assertEquals(0, emptyArrayInt.size)
        assertEquals(DataType.IntDataType, emptyArrayInt.dtype)

        val emptyArrayDouble = emptyArray<DoubleArray>().toNDArray()
        assertEquals(0, emptyArrayDouble.size)
        assertEquals(DataType.DoubleDataType, emptyArrayDouble.dtype)

        val emptyArrayFloat = emptyArray<FloatArray>().toNDArray()
        assertEquals(0, emptyArrayFloat.size)
        assertEquals(DataType.FloatDataType, emptyArrayFloat.dtype)

        val emptyArrayLong = emptyArray<LongArray>().toNDArray()
        assertEquals(0, emptyArrayLong.size)
        assertEquals(DataType.LongDataType, emptyArrayLong.dtype)

        val emptyArrayShort = emptyArray<ShortArray>().toNDArray()
        assertEquals(0, emptyArrayShort.size)
        assertEquals(DataType.ShortDataType, emptyArrayShort.dtype)

        val emptyArrayByte = emptyArray<ByteArray>().toNDArray()
        assertEquals(0, emptyArrayByte.size)
        assertEquals(DataType.ByteDataType, emptyArrayByte.dtype)
    }

    @Test
    fun `test_find`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        assertEquals(list.find { it % 2 != 0 }, ndarray.find { it % 2 != 0 })
        assertEquals(list.findLast { it % 2 == 0 }, ndarray.findLast { it % 2 == 0 })
    }

    @Test
    fun `test_first_and_firstOrNull_with_predicate`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        println(list.first { it % 2 != 0 })
        assertEquals(list.first { it % 2 != 0 }, ndarray.first { it % 2 != 0 })
        assertEquals(list.firstOrNull { it % 10 == 0 }, ndarray.firstOrNull { it % 10 == 0 })
    }

    @Test
    fun `test_flatMap`() {
        val list = listOf(0, 1, 2, 3)
        val ndarray = mk.ndarray(list, 2, 2)
        assertEquals(
            mk.ndarray(list.flatMap { listOf(it, it + 1, it + 2) }),
            ndarray.flatMap { listOf(it, it + 1, it + 2) })
    }

    @Test
    fun `test_flatMapIndexed`() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(list)
        assertEquals(
            mk.ndarray(list.flatMapIndexed { i, e -> listOf(e, i) }),
            ndarray.flatMapIndexed { i: Int, e -> listOf(e, i) })
    }

    @Test
    fun `test_flatMap_empty_result`() {
        val data = mk.arange<Int>(5)
        val actual = data.flatMap { emptyList<Double>() }
        assertEquals(0, actual.size)
        assertEquals(DataType.DoubleDataType, actual.dtype)

        val actualIndexed = data.flatMapIndexed { _, _ -> emptyList<Float>() }
        assertEquals(0, actualIndexed.size)
        assertEquals(DataType.FloatDataType, actualIndexed.dtype)

        val actualMultiIndexed = data.flatMapMultiIndexed { _, _ -> emptyList<Byte>() }
        assertEquals(0, actualMultiIndexed.size)
        assertEquals(DataType.ByteDataType, actualMultiIndexed.dtype)

        val emptyData = mk.zeros<Long>(0)
        val actualFromEmpty = emptyData.flatMap { listOf(it.toInt()) }
        assertEquals(0, actualFromEmpty.size)
        assertEquals(DataType.IntDataType, actualFromEmpty.dtype)

        val actualFromEmptyIndexed = emptyData.flatMapIndexed { i, v -> listOf(i, v.toInt()) }
        assertEquals(0, actualFromEmptyIndexed.size)
        assertEquals(DataType.IntDataType, actualFromEmptyIndexed.dtype)

        val actualFromEmptyMultiIndexed = emptyData.flatMapMultiIndexed { i, v -> listOf(v.toDouble()) }
        assertEquals(0, actualFromEmptyMultiIndexed.size)
        assertEquals(DataType.DoubleDataType, actualFromEmptyMultiIndexed.dtype)
    }

    @Test
    fun `test_flatMapMultiIndexed`() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(list, 2, 2)
        val actual = ndarray.flatMapMultiIndexed { index, element -> listOf(element, index[0] + index[1]) }
        val expected = mk.ndarray(listOf(1, 0, 2, 1, 3, 1, 4, 2))
        assertEquals(expected, actual)
        assertEquals(DataType.IntDataType, actual.dtype)
    }

    @Test
    fun `test_fold`() {
        val list = listOf(1, 2, 3, 4)
        val ndarray = mk.ndarray(list)
        assertEquals(list.fold(3, Int::times), ndarray.fold(3, Int::times))
    }

    @Test
    fun `test_foldIndexed`() {
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
    fun `test_groupNDArrayBy`() {
        val data = mk.d3array(2, 2, 2) { it }
        val expected1 = mapOf(0 to mk.ndarrayOf(0, 2, 4, 6), 1 to mk.ndarrayOf(1, 3, 5, 7))
        assertEquals(expected1, data.groupNDArrayBy { it % 2 })

        val expected2 = mapOf(0 to mk.ndarrayOf(0f, 2f, 4f, 6f), 1 to mk.ndarrayOf(1f, 3f, 5f, 7f))
        assertEquals(expected2, data.groupNDArrayBy({ it % 2 }, { it.toFloat() }))
    }

    @Test
    fun `test_intersect`() {
        val list = listOf(1, 3, 4, 5, 6, 10)
        val ndarray = mk.ndarray(list)
        val list2 = listOf(2, 3, 5, 7, 6, 11)
        val expected = list intersect list2
        val actual = ndarray intersect list2
        assertEquals(expected, actual)
    }

    @Test
    fun `test_last`() {
        val ndarray = mk.ndarray(mk[mk[2, 3, -17], mk[10, 23, 33]])
        assertEquals(33, ndarray.last())
    }

    @Test
    fun `test_last_with_predicate`() {
        val list = listOf(1, 2, 3, -12, 42, 33, 89)
        val ndarray = mk.ndarray(list)
        println(list.last { it % 2 == 0 })
        println(ndarray.last { it % 2 == 0 })
    }

    @Test
    fun `test_map_for_scalar_ndarray`() {
        val a = mk.ndarray(mk[mk[mk[3.2]]])
        assertEquals(mk.ndarray(mk[mk[mk[3]]]), a.map { it.toInt() })
    }

    @Test
    fun `test_map`() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(1, 4, 9, 16), data.map { it * it })
    }

    @Test
    fun `test_mapIndexed`() {
        val data = mk.ndarrayOf(1, 2, 3, 4)
        assertEquals(mk.ndarrayOf(0, 2, 6, 12), data.mapIndexed { idx: Int, value -> value * idx })
        val ndarray = mk.ndarrayOf(1, 2, 3, 4).reshape(2, 2)
        ndarray.mapMultiIndexed { idx: IntArray, value -> value * (idx[0] xor idx[1]) }
    }

    @Test
    fun `test_max`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.maxOrNull(), ndarray.max())
    }

    @Test
    fun `test_maxBy`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.maxByOrNull { -it }, ndarray.maxBy { -it })
    }

    @Test
    fun `test_min`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.minOrNull(), ndarray.min())

    }

    @Test
    fun `test_minBy`() {
        val array = intArrayOf(1, -2, 10, 23, 3, 10, 32, -1, 17)
        val ndarray = mk.ndarray(array)
        assertEquals(array.minByOrNull { -it }, ndarray.minBy { -it })
    }

    @Test
    fun `test_partition`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        val (h, t) = ndarray.partition { it % 2 == 0 }
        val (lH, lT) = list.partition { it % 2 == 0 }
        assertEquals(mk.ndarray(lH), h)
        assertEquals(mk.ndarray(lT), t)
    }

    @Test
    fun `test_sort`() {
        //TODO(assert)
        val intArray = intArrayOf(42, 42, 23, 1, 23, 4, 10, 14, 3, 7, 25, 16, 2, 1, 37)
        val ndarray = mk.ndarray(intArray, 3, 5)
        val sortedNDArray = ndarray.sorted()
        sortedNDArray[2, 2] = 1000

    }

    @Test
    fun `test_reduce`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7)
        val ndarray = mk.ndarray(list)
        val expected = list.reduce { acc, i -> acc + i / 2 }
        val actual = ndarray.reduce { acc, i -> acc + i / 2 }
        assertEquals(expected, actual)
    }

    @Test
    fun `test_reversed`() {
        val list = listOf(1, 2, 3, 4, 5, 6, 7, 8)
        val ndarray = mk.ndarray(list, 2, 4)
        val expected = mk.ndarray(list.reversed(), 2, 4)
        assertEquals(expected, ndarray.reversed())
    }

    @Test
    fun `test_scan`() {
        val ndarray = mk.ndarray(mk[1, 2, 3, 4, 5, 6])
        println(ndarray.scan(10) { acc: Int, i: Int -> acc + i })
    }

    @Test
    fun `test_minimum`() {
        val ndarray1 = mk.ndarray(mk[2, 3, 4])
        val ndarray2 = mk.ndarray(mk[1, 5, 2])
        assertEquals(mk.ndarray(mk[1, 3, 2]), ndarray1.minimum(ndarray2))
    }

    @Test
    fun `test_maximum`() {
        val ndarray1 = mk.ndarray(mk[2, 3, 4])
        val ndarray2 = mk.ndarray(mk[1, 5, 2])
        assertEquals(mk.ndarray(mk[2, 5, 4]), ndarray1.maximum(ndarray2))
    }
}
