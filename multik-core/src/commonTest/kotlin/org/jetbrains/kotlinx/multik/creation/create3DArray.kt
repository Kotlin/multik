package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.api.createAlignedNDArray
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.d3arrayIndices
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.complexFloatArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.minus
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD3
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

abstract class Create3DArrayTestBase<T : Any> {
    protected val dim1 = 5
    protected val dim2 = 7
    protected val dim3 = 3

    abstract val zero: T
    abstract val one: T

    abstract fun createZeros(): D3Array<T>
    abstract fun createOnes(): D3Array<T>
    abstract fun makeFromNestedList(): Pair<D3Array<T>, List<List<List<T>>>>
    abstract fun makeFromSet(): Pair<D3Array<T>, Set<T>>
    abstract fun makeFromPrimitiveArray(): Pair<D3Array<T>, Any>
    abstract fun createWithInit(): Pair<D3Array<T>, Any>
    abstract fun createWithIndices(): Pair<D3Array<T>, Any>
    abstract fun assertDataEquals(a: D3Array<T>, expected: Any)

    @Test
    fun createZeroFilledArray() {
        val a = createZeros()
        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == zero } }
    }

    @Test
    fun createArrayFilledWithOnes() {
        val a = createOnes()
        assertEquals(dim1 * dim2 * dim3, a.size)
        assertEquals(dim1 * dim2 * dim3, a.data.size)
        assertTrue { a.all { it == one } }
    }

    @Test
    fun createFromNestedList() {
        val (a, list) = makeFromNestedList()
        assertEquals(list, a.toListD3())
    }

    @Test
    fun createFromSet() {
        val (a, set) = makeFromSet()
        assertEquals(set.size, a.size)
        assertEquals(set, a.toSet())
    }

    @Test
    fun createFromPrimitiveArray() {
        val (a, expected) = makeFromPrimitiveArray()
        assertEquals(12, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitFunction() {
        val (a, expected) = createWithInit()
        assertEquals(12, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitAndIndices() {
        val (a, expected) = createWithIndices()
        assertEquals(12, a.size)
        assertDataEquals(a, expected)
    }
}

class Create3DByteArrayTests : Create3DArrayTestBase<Byte>() {
    override val zero: Byte = 0.toByte()
    override val one: Byte = 1.toByte()
    override fun createZeros(): D3Array<Byte> = mk.zeros<Byte>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Byte> = mk.ones<Byte>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Byte>, List<List<List<Byte>>>> {
        val list = listOf(listOf(listOf<Byte>(1, 3), listOf<Byte>(7, 8)), listOf(listOf<Byte>(4, 7), listOf<Byte>(9, 2)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Byte>, Set<Byte>> {
        val set = setOf<Byte>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        return mk.ndarray<Byte, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Byte>, Any> {
        val array = byteArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Byte>, Any> =
        mk.d3array<Byte>(2, 3, 2) { (it + 3).toByte() } to byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithIndices(): Pair<D3Array<Byte>, Any> =
        mk.d3arrayIndices(2, 3, 2) { i, j, k -> (i * j + k).toByte() } to byteArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

    override fun assertDataEquals(a: D3Array<Byte>, expected: Any) {
        a.data.getByteArray() shouldBe (expected as ByteArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf<Byte>(1, 3), listOf<Byte>(8)), listOf(listOf<Byte>(4, 7)))
        val expected = listOf(listOf(listOf<Byte>(1, 3), listOf<Byte>(8, 0)), listOf(listOf<Byte>(4, 7), listOf<Byte>(0, 0)))
        val a: D3Array<Byte> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DShortArrayTests : Create3DArrayTestBase<Short>() {
    override val zero: Short = 0.toShort()
    override val one: Short = 1.toShort()
    override fun createZeros(): D3Array<Short> = mk.zeros<Short>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Short> = mk.ones<Short>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Short>, List<List<List<Short>>>> {
        val list = listOf(listOf(listOf<Short>(1, 3), listOf<Short>(7, 8)), listOf(listOf<Short>(4, 7), listOf<Short>(9, 2)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Short>, Set<Short>> {
        val set = setOf<Short>(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        return mk.ndarray<Short, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Short>, Any> {
        val array = shortArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Short>, Any> =
        mk.d3array<Short>(2, 3, 2) { (it + 3).toShort() } to shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithIndices(): Pair<D3Array<Short>, Any> =
        mk.d3arrayIndices(2, 3, 2) { i, j, k -> (i * j + k).toShort() } to shortArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

    override fun assertDataEquals(a: D3Array<Short>, expected: Any) {
        a.data.getShortArray() shouldBe (expected as ShortArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf<Short>(1, 3), listOf<Short>(8)), listOf(listOf<Short>(4, 7)))
        val expected = listOf(listOf(listOf<Short>(1, 3), listOf<Short>(8, 0)), listOf(listOf<Short>(4, 7), listOf<Short>(0, 0)))
        val a: D3Array<Short> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DIntArrayTests : Create3DArrayTestBase<Int>() {
    override val zero: Int = 0
    override val one: Int = 1
    override fun createZeros(): D3Array<Int> = mk.zeros<Int>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Int> = mk.ones<Int>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Int>, List<List<List<Int>>>> {
        val list = listOf(listOf(listOf(1, 3), listOf(7, 8)), listOf(listOf(4, 7), listOf(9, 2)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Int>, Set<Int>> {
        val set = setOf(1, 3, 8, 4, 9, 2, 7, 5, 11, 13, -8, -1)
        return mk.ndarray<Int, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Int>, Any> {
        val array = intArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Int>, Any> =
        mk.d3array<Int>(2, 3, 2) { it + 3 } to intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithIndices(): Pair<D3Array<Int>, Any> =
        mk.d3arrayIndices(2, 3, 2) { i, j, k -> i * j + k } to intArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

    override fun assertDataEquals(a: D3Array<Int>, expected: Any) {
        a.data.getIntArray() shouldBe (expected as IntArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf(1, 3), listOf(8)), listOf(listOf(4, 7)))
        val expected = listOf(listOf(listOf(1, 3), listOf(8, 0)), listOf(listOf(4, 7), listOf(0, 0)))
        val a: D3Array<Int> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DLongArrayTests : Create3DArrayTestBase<Long>() {
    override val zero: Long = 0L
    override val one: Long = 1L
    override fun createZeros(): D3Array<Long> = mk.zeros<Long>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Long> = mk.ones<Long>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Long>, List<List<List<Long>>>> {
        val list = listOf(listOf(listOf(1L, 3L), listOf(7L, 8L)), listOf(listOf(4L, 7L), listOf(9L, 2L)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Long>, Set<Long>> {
        val set = setOf(1L, 3L, 8L, 4L, 9L, 2L, 7L, 5L, 11L, 13L, -8L, -1L)
        return mk.ndarray<Long, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Long>, Any> {
        val array = longArrayOf(1, 3, 8, 4, 9, 2, 7, 3, 4, 3, 8, 1)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Long>, Any> =
        mk.d3array<Long>(2, 3, 2) { it + 3L } to longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithIndices(): Pair<D3Array<Long>, Any> =
        mk.d3arrayIndices<Long>(2, 3, 2) { i, j, k -> i * j + k.toLong() } to longArrayOf(0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 3)

    override fun assertDataEquals(a: D3Array<Long>, expected: Any) {
        a.data.getLongArray() shouldBe (expected as LongArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf(1L, 3L), listOf(8L)), listOf(listOf(4L, 7L)))
        val expected = listOf(listOf(listOf(1L, 3L), listOf(8L, 0L)), listOf(listOf(4L, 7L), listOf(0L, 0L)))
        val a: D3Array<Long> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DFloatArrayTests : Create3DArrayTestBase<Float>() {
    override val zero: Float = 0f
    override val one: Float = 1f
    override fun createZeros(): D3Array<Float> = mk.zeros<Float>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Float> = mk.ones<Float>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Float>, List<List<List<Float>>>> {
        val list = listOf(listOf(listOf(1f, 3f), listOf(7f, 8f)), listOf(listOf(4f, 7f), listOf(9f, 2f)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Float>, Set<Float>> {
        val set = setOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 5f, 11f, 13f, -8f, -1f)
        return mk.ndarray<Float, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Float>, Any> {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f, 2f, 7f, 3f, 4f, 3f, 8f, 1f)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Float>, Any> =
        mk.d3array<Float>(2, 3, 2) { it + 3f } to floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f)

    override fun createWithIndices(): Pair<D3Array<Float>, Any> =
        mk.d3arrayIndices<Float>(2, 3, 2) { i, j, k -> i * j + k.toFloat() } to floatArrayOf(0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f, 1f, 2f, 2f, 3f)

    override fun assertDataEquals(a: D3Array<Float>, expected: Any) {
        a.data.getFloatArray() shouldBe (expected as FloatArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf(1f, 3f), listOf(8f)), listOf(listOf(4f, 7f)))
        val expected = listOf(listOf(listOf(1f, 3f), listOf(8f, 0f)), listOf(listOf(4f, 7f), listOf(0f, 0f)))
        val a: D3Array<Float> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DDoubleArrayTests : Create3DArrayTestBase<Double>() {
    override val zero: Double = 0.0
    override val one: Double = 1.0
    override fun createZeros(): D3Array<Double> = mk.zeros<Double>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<Double> = mk.ones<Double>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<Double>, List<List<List<Double>>>> {
        val list = listOf(listOf(listOf(1.0, 3.0), listOf(7.0, 8.0)), listOf(listOf(4.0, 7.0), listOf(9.0, 2.0)))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<Double>, Set<Double>> {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 5.0, 11.0, 13.0, -8.0, -1.0)
        return mk.ndarray<Double, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<Double>, Any> {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0, 7.0, 3.0, 4.0, 3.0, 8.0, 1.0)
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<Double>, Any> =
        mk.d3array<Double>(2, 3, 2) { it + 3.0 } to doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

    override fun createWithIndices(): Pair<D3Array<Double>, Any> =
        mk.d3arrayIndices<Double>(2, 3, 2) { i, j, k -> i * j + k.toDouble() } to doubleArrayOf(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0)

    override fun assertDataEquals(a: D3Array<Double>, expected: Any) {
        a.data.getDoubleArray() shouldBe (expected as DoubleArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned3DArray() {
        val list = listOf(listOf(listOf(1.0, 3.0), listOf(8.0)), listOf(listOf(4.0, 7.0)))
        val expected = listOf(listOf(listOf(1.0, 3.0), listOf(8.0, 0.0)), listOf(listOf(4.0, 7.0), listOf(0.0, 0.0)))
        val a: D3Array<Double> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD3())
    }
}

class Create3DComplexFloatArrayTests : Create3DArrayTestBase<ComplexFloat>() {
    override val zero: ComplexFloat = ComplexFloat.zero
    override val one: ComplexFloat = ComplexFloat.one
    override fun createZeros(): D3Array<ComplexFloat> = mk.zeros<ComplexFloat>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<ComplexFloat> = mk.ones<ComplexFloat>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<ComplexFloat>, List<List<List<ComplexFloat>>>> {
        val list = listOf(
            listOf(listOf(1f + 0f.i, 3f + 0f.i), listOf(7f + 0f.i, 8f + 0f.i)),
            listOf(listOf(4f + 0f.i, 7f + 0f.i), listOf(9f + 0f.i, 2f + 0f.i))
        )
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<ComplexFloat>, Set<ComplexFloat>> {
        val set = setOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 5f + 0f.i, 11f + 0f.i, 13f + 0f.i, -8f + 0f.i, -1f + 0f.i
        )
        return mk.ndarray<ComplexFloat, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<ComplexFloat>, Any> {
        val array = complexFloatArrayOf(
            1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i,
            7f + 0f.i, 3f + 0f.i, 4f + 0f.i, 3f + 0f.i, 8f + 0f.i, 1f + 0f.i
        )
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<ComplexFloat>, Any> =
        mk.d3array<ComplexFloat>(2, 3, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) } to complexFloatArrayOf(
            3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i,
            6.21f + 2.168f.i, 7.21f + 3.168f.i, 8.21f + 4.168f.i,
            9.21f + 5.168f.i, 10.21f + 6.168f.i, 11.21f + 7.168f.i,
            12.21f + 8.168f.i, 13.21f + 9.168f.i, 14.21f + 10.168f.i
        )

    override fun createWithIndices(): Pair<D3Array<ComplexFloat>, Any> =
        mk.d3arrayIndices<ComplexFloat>(2, 3, 2) { i, j, k -> i * j + k + ComplexFloat(7) } to complexFloatArrayOf(
            7f + 0f.i, 8f + 0f.i, 7f + 0f.i, 8f + 0f.i, 7f + 0f.i, 8f + 0f.i,
            7f + 0f.i, 8f + 0f.i, 8f + 0f.i, 9f + 0f.i, 9f + 0f.i, 10f + 0f.i
        )

    override fun assertDataEquals(a: D3Array<ComplexFloat>, expected: Any) {
        a.data.getComplexFloatArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray)
    }
}

class Create3DComplexDoubleArrayTests : Create3DArrayTestBase<ComplexDouble>() {
    override val zero: ComplexDouble = ComplexDouble.zero
    override val one: ComplexDouble = ComplexDouble.one
    override fun createZeros(): D3Array<ComplexDouble> = mk.zeros<ComplexDouble>(dim1, dim2, dim3)
    override fun createOnes(): D3Array<ComplexDouble> = mk.ones<ComplexDouble>(dim1, dim2, dim3)

    override fun makeFromNestedList(): Pair<D3Array<ComplexDouble>, List<List<List<ComplexDouble>>>> {
        val list = listOf(
            listOf(listOf(1.0 + .0.i, 3.0 + .0.i), listOf(7.0 + .0.i, 8 + .0.i)),
            listOf(listOf(4.0 + .0.i, 7.0 + .0.i), listOf(9.0 + .0.i, 2.0 + .0.i))
        )
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D3Array<ComplexDouble>, Set<ComplexDouble>> {
        val set = setOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 5.0 + .0.i, 11.0 + .0.i, 13.0 + .0.i, -8.0 + .0.i, -1.0 + .0.i
        )
        return mk.ndarray<ComplexDouble, D3>(set, shape = intArrayOf(2, 3, 2)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D3Array<ComplexDouble>, Any> {
        val array = complexDoubleArrayOf(
            1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i,
            7.0 + .0.i, 3.0 + .0.i, 4.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 1.0 + .0.i
        )
        return mk.ndarray(array, 2, 3, 2) to array
    }

    override fun createWithInit(): Pair<D3Array<ComplexDouble>, Any> =
        mk.d3array<ComplexDouble>(2, 3, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) } to complexDoubleArrayOf(
            3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i,
            6.21 + 2.168.i, 7.21 + 3.168.i, 8.21 + 4.168.i,
            9.21 + 5.168.i, 10.21 + 6.168.i, 11.21 + 7.168.i,
            12.21 + 8.168.i, 13.21 + 9.168.i, 14.21 + 10.168.i
        )

    override fun createWithIndices(): Pair<D3Array<ComplexDouble>, Any> =
        mk.d3arrayIndices<ComplexDouble>(2, 3, 2) { i, j, k -> i * j + k + ComplexDouble(7) } to complexDoubleArrayOf(
            7.0 + .0.i, 8.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i,
            7.0 + .0.i, 8.0 + .0.i, 8.0 + .0.i, 9.0 + .0.i, 9.0 + .0.i, 10.0 + .0.i
        )

    override fun assertDataEquals(a: D3Array<ComplexDouble>, expected: Any) {
        a.data.getComplexDoubleArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray)
    }
}
