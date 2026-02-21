package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.api.createAlignedNDArray
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.diagonal
import org.jetbrains.kotlinx.multik.api.identity
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
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

abstract class Create2DArrayTestBase<T : Any> {
    protected val dim1 = 5
    protected val dim2 = 7
    protected val n = 7

    abstract val zero: T
    abstract val one: T

    abstract fun createZeros(): D2Array<T>
    abstract fun createOnes(): D2Array<T>
    abstract fun createIdentity(): D2Array<T>
    abstract fun createDiagonal(): D2Array<T>
    abstract fun expectedDiagonalElement(i: Int): T
    abstract fun makeFromNestedList(): Pair<D2Array<T>, List<List<T>>>
    abstract fun makeFromSet(): Pair<D2Array<T>, Set<T>>
    abstract fun makeFromPrimitiveArray(): Pair<D2Array<T>, Any>
    abstract fun createWithInit(): Pair<D2Array<T>, Any>
    abstract fun createWithIndices(): Pair<D2Array<T>, Any>
    abstract fun assertDataEquals(a: D2Array<T>, expected: Any)

    @Test
    fun createZeroFilledArray() {
        val a = createZeros()
        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == zero } }
    }

    @Test
    fun createArrayFilledWithOnes() {
        val a = createOnes()
        assertEquals(dim1 * dim2, a.size)
        assertEquals(dim1 * dim2, a.data.size)
        assertTrue { a.all { it == one } }
    }

    @Test
    fun createIdentityMatrix() {
        val a = createIdentity()
        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(one, a[i, j], "Expected diagonal elements to be $one")
                else
                    assertEquals(zero, a[i, j], "Expected non-diagonal elements to be $zero")
            }
        }
    }

    @Test
    fun createDiagonalMatrix() {
        val a = createDiagonal()
        assertEquals(n * n, a.size)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j)
                    assertEquals(expectedDiagonalElement(i), a[i, j], "Expected element at [$i:$j] to be ${expectedDiagonalElement(i)}")
                else
                    assertEquals(zero, a[i, j], "Expected non-diagonal elements to be $zero")
            }
        }
    }

    @Test
    fun createFromNestedList() {
        val (a, list) = makeFromNestedList()
        assertEquals(list, a.toListD2())
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
        assertEquals(6, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitFunction() {
        val (a, expected) = createWithInit()
        assertEquals(6, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitAndIndices() {
        val (a, expected) = createWithIndices()
        assertEquals(6, a.size)
        assertDataEquals(a, expected)
    }
}

class Create2DByteArrayTests : Create2DArrayTestBase<Byte>() {
    override val zero: Byte = 0.toByte()
    override val one: Byte = 1.toByte()
    override fun createZeros(): D2Array<Byte> = mk.zeros<Byte>(dim1, dim2)
    override fun createOnes(): D2Array<Byte> = mk.ones<Byte>(dim1, dim2)
    override fun createIdentity(): D2Array<Byte> = mk.identity<Byte>(n)
    override fun createDiagonal(): D2Array<Byte> = mk.diagonal(List(n) { (it + 1).toByte() })
    override fun expectedDiagonalElement(i: Int): Byte = (i + 1).toByte()

    override fun makeFromNestedList(): Pair<D2Array<Byte>, List<List<Byte>>> {
        val list = listOf(listOf<Byte>(1, 3, 8), listOf<Byte>(4, 7, 2))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Byte>, Set<Byte>> {
        val set = setOf<Byte>(1, 3, 8, 4, 9, 2)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Byte>, Any> {
        val array = byteArrayOf(1, 3, 8, 4, 9, 2)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Byte>, Any> =
        mk.d2array<Byte>(2, 3) { (it + 3).toByte() } to byteArrayOf(3, 4, 5, 6, 7, 8)

    override fun createWithIndices(): Pair<D2Array<Byte>, Any> =
        mk.d2arrayIndices(2, 3) { i, j -> (i * j + 7).toByte() } to byteArrayOf(7, 7, 7, 7, 8, 9)

    override fun assertDataEquals(a: D2Array<Byte>, expected: Any) {
        a.data.getByteArray() shouldBe (expected as ByteArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf<Byte>(1, 3, 8), listOf<Byte>(4, 7), listOf<Byte>(2, 5, 9, 10))
        val expected = listOf(listOf<Byte>(1, 3, 8, 0), listOf<Byte>(4, 7, 0, 0), listOf<Byte>(2, 5, 9, 10))
        val a: D2Array<Byte> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DShortArrayTests : Create2DArrayTestBase<Short>() {
    override val zero: Short = 0.toShort()
    override val one: Short = 1.toShort()
    override fun createZeros(): D2Array<Short> = mk.zeros<Short>(dim1, dim2)
    override fun createOnes(): D2Array<Short> = mk.ones<Short>(dim1, dim2)
    override fun createIdentity(): D2Array<Short> = mk.identity<Short>(n)
    override fun createDiagonal(): D2Array<Short> = mk.diagonal(List(n) { (it + 1).toShort() })
    override fun expectedDiagonalElement(i: Int): Short = (i + 1).toShort()

    override fun makeFromNestedList(): Pair<D2Array<Short>, List<List<Short>>> {
        val list = listOf(listOf<Short>(1, 3, 8), listOf<Short>(4, 7, 2))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Short>, Set<Short>> {
        val set = setOf<Short>(1, 3, 8, 4, 9, 2)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Short>, Any> {
        val array = shortArrayOf(1, 3, 8, 4, 9, 2)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Short>, Any> =
        mk.d2array<Short>(2, 3) { (it + 3).toShort() } to shortArrayOf(3, 4, 5, 6, 7, 8)

    override fun createWithIndices(): Pair<D2Array<Short>, Any> =
        mk.d2arrayIndices(2, 3) { i, j -> (i * j + 7).toShort() } to shortArrayOf(7, 7, 7, 7, 8, 9)

    override fun assertDataEquals(a: D2Array<Short>, expected: Any) {
        a.data.getShortArray() shouldBe (expected as ShortArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf<Short>(1, 3, 8), listOf<Short>(4, 7), listOf<Short>(2, 5, 9, 10))
        val expected = listOf(listOf<Short>(1, 3, 8, 0), listOf<Short>(4, 7, 0, 0), listOf<Short>(2, 5, 9, 10))
        val a: D2Array<Short> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DIntArrayTests : Create2DArrayTestBase<Int>() {
    override val zero: Int = 0
    override val one: Int = 1
    override fun createZeros(): D2Array<Int> = mk.zeros<Int>(dim1, dim2)
    override fun createOnes(): D2Array<Int> = mk.ones<Int>(dim1, dim2)
    override fun createIdentity(): D2Array<Int> = mk.identity<Int>(n)
    override fun createDiagonal(): D2Array<Int> = mk.diagonal(List(n) { it + 1 })
    override fun expectedDiagonalElement(i: Int): Int = i + 1

    override fun makeFromNestedList(): Pair<D2Array<Int>, List<List<Int>>> {
        val list = listOf(listOf(1, 3, 8), listOf(4, 7, 2))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Int>, Set<Int>> {
        val set = setOf(1, 3, 8, 4, 9, 2)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Int>, Any> {
        val array = intArrayOf(1, 3, 8, 4, 9, 2)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Int>, Any> =
        mk.d2array<Int>(2, 3) { it + 3 } to intArrayOf(3, 4, 5, 6, 7, 8)

    override fun createWithIndices(): Pair<D2Array<Int>, Any> =
        mk.d2arrayIndices(2, 3) { i, j -> i * j + 7 } to intArrayOf(7, 7, 7, 7, 8, 9)

    override fun assertDataEquals(a: D2Array<Int>, expected: Any) {
        a.data.getIntArray() shouldBe (expected as IntArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf(1, 3, 8), listOf(4, 7), listOf(2, 5, 9, 10))
        val expected = listOf(listOf(1, 3, 8, 0), listOf(4, 7, 0, 0), listOf(2, 5, 9, 10))
        val a: D2Array<Int> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DLongArrayTests : Create2DArrayTestBase<Long>() {
    override val zero: Long = 0L
    override val one: Long = 1L
    override fun createZeros(): D2Array<Long> = mk.zeros<Long>(dim1, dim2)
    override fun createOnes(): D2Array<Long> = mk.ones<Long>(dim1, dim2)
    override fun createIdentity(): D2Array<Long> = mk.identity<Long>(n)
    override fun createDiagonal(): D2Array<Long> = mk.diagonal(List(n) { it + 1L })
    override fun expectedDiagonalElement(i: Int): Long = i + 1L

    override fun makeFromNestedList(): Pair<D2Array<Long>, List<List<Long>>> {
        val list = listOf(listOf(1L, 3L, 8L), listOf(4L, 7L, 2L))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Long>, Set<Long>> {
        val set = setOf(1L, 3L, 8L, 4L, 9L, 2L)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Long>, Any> {
        val array = longArrayOf(1, 3, 8, 4, 9, 2)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Long>, Any> =
        mk.d2array<Long>(2, 3) { it + 3L } to longArrayOf(3, 4, 5, 6, 7, 8)

    override fun createWithIndices(): Pair<D2Array<Long>, Any> =
        mk.d2arrayIndices<Long>(2, 3) { i, j -> i * j + 7L } to longArrayOf(7, 7, 7, 7, 8, 9)

    override fun assertDataEquals(a: D2Array<Long>, expected: Any) {
        a.data.getLongArray() shouldBe (expected as LongArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf<Long>(1, 3, 8), listOf<Long>(4, 7), listOf<Long>(2, 5, 9, 10))
        val expected = listOf(listOf<Long>(1, 3, 8, 0), listOf<Long>(4, 7, 0, 0), listOf<Long>(2, 5, 9, 10))
        val a: D2Array<Long> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DFloatArrayTests : Create2DArrayTestBase<Float>() {
    override val zero: Float = 0f
    override val one: Float = 1f
    override fun createZeros(): D2Array<Float> = mk.zeros<Float>(dim1, dim2)
    override fun createOnes(): D2Array<Float> = mk.ones<Float>(dim1, dim2)
    override fun createIdentity(): D2Array<Float> = mk.identity<Float>(n)
    override fun createDiagonal(): D2Array<Float> = mk.diagonal(List(n) { it + 1f })
    override fun expectedDiagonalElement(i: Int): Float = i + 1f

    override fun makeFromNestedList(): Pair<D2Array<Float>, List<List<Float>>> {
        val list = listOf(listOf(1f, 3f, 8f), listOf(4f, 7f, 2f))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Float>, Set<Float>> {
        val set = setOf(1f, 3f, 8f, 4f, 9f, 2f)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Float>, Any> {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f, 2f)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Float>, Any> =
        mk.d2array<Float>(2, 3) { it + 3f } to floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f)

    override fun createWithIndices(): Pair<D2Array<Float>, Any> =
        mk.d2arrayIndices<Float>(2, 3) { i, j -> i * j + 7f } to floatArrayOf(7f, 7f, 7f, 7f, 8f, 9f)

    override fun assertDataEquals(a: D2Array<Float>, expected: Any) {
        a.data.getFloatArray() shouldBe (expected as FloatArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf(1f, 3f, 8f), listOf(4f, 7f), listOf(2f, 5f, 9f, 10f))
        val expected = listOf(listOf(1f, 3f, 8f, 0f), listOf(4f, 7f, 0f, 0f), listOf(2f, 5f, 9f, 10f))
        val a: D2Array<Float> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DDoubleArrayTests : Create2DArrayTestBase<Double>() {
    override val zero: Double = 0.0
    override val one: Double = 1.0
    override fun createZeros(): D2Array<Double> = mk.zeros<Double>(dim1, dim2)
    override fun createOnes(): D2Array<Double> = mk.ones<Double>(dim1, dim2)
    override fun createIdentity(): D2Array<Double> = mk.identity<Double>(n)
    override fun createDiagonal(): D2Array<Double> = mk.diagonal(List(n) { it + 1.0 })
    override fun expectedDiagonalElement(i: Int): Double = i + 1.0

    override fun makeFromNestedList(): Pair<D2Array<Double>, List<List<Double>>> {
        val list = listOf(listOf(1.0, 3.0, 8.0), listOf(4.0, 7.0, 2.0))
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<Double>, Set<Double>> {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<Double>, Any> {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0, 2.0)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<Double>, Any> =
        mk.d2array<Double>(2, 3) { it + 3.0 } to doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    override fun createWithIndices(): Pair<D2Array<Double>, Any> =
        mk.d2arrayIndices<Double>(2, 3) { i, j -> i * j + 7.0 } to doubleArrayOf(7.0, 7.0, 7.0, 7.0, 8.0, 9.0)

    override fun assertDataEquals(a: D2Array<Double>, expected: Any) {
        a.data.getDoubleArray() shouldBe (expected as DoubleArray)
    }

    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun createAligned2DArray() {
        val list = listOf(listOf(1.0, 3.0, 8.0), listOf(4.0, 7.0), listOf(2.0, 5.0, 9.0, 10.0))
        val expected = listOf(listOf(1.0, 3.0, 8.0, 0.0), listOf(4.0, 7.0, 0.0, 0.0), listOf(2.0, 5.0, 9.0, 10.0))
        val a: D2Array<Double> = mk.createAlignedNDArray(list, filling = 0.0)
        assertEquals(expected, a.toListD2())
    }
}

class Create2DComplexFloatArrayTests : Create2DArrayTestBase<ComplexFloat>() {
    override val zero: ComplexFloat = ComplexFloat.zero
    override val one: ComplexFloat = ComplexFloat.one
    override fun createZeros(): D2Array<ComplexFloat> = mk.zeros<ComplexFloat>(dim1, dim2)
    override fun createOnes(): D2Array<ComplexFloat> = mk.ones<ComplexFloat>(dim1, dim2)
    override fun createIdentity(): D2Array<ComplexFloat> = mk.identity<ComplexFloat>(n)
    override fun createDiagonal(): D2Array<ComplexFloat> = mk.diagonal(List(n) { ComplexFloat(it + 1f, it + 1f) })
    override fun expectedDiagonalElement(i: Int): ComplexFloat = ComplexFloat(i + 1f, i + 1f)

    override fun makeFromNestedList(): Pair<D2Array<ComplexFloat>, List<List<ComplexFloat>>> {
        val list = listOf(
            listOf(ComplexFloat.one, 3f + 0f.i, 8f + 0f.i),
            listOf(4f + 0f.i, 7f + 0f.i, 2f + 0f.i)
        )
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<ComplexFloat>, Set<ComplexFloat>> {
        val set = setOf(1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<ComplexFloat>, Any> {
        val array = complexFloatArrayOf(1f + 0f.i, 3f + 0f.i, 8f + 0f.i, 4f + 0f.i, 9f + 0f.i, 2f + 0f.i)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<ComplexFloat>, Any> =
        mk.d2array<ComplexFloat>(2, 3) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) } to complexFloatArrayOf(
            3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i,
            6.21f + 2.168f.i, 7.21f + 3.168f.i, 8.21f + 4.168f.i
        )

    override fun createWithIndices(): Pair<D2Array<ComplexFloat>, Any> =
        mk.d2arrayIndices<ComplexFloat>(2, 3) { i, j -> i * j + ComplexFloat(7) } to complexFloatArrayOf(
            7f + 0f.i, 7f + 0f.i, 7f + 0f.i, 7f + 0f.i, 8f + 0f.i, 9f + 0f.i
        )

    override fun assertDataEquals(a: D2Array<ComplexFloat>, expected: Any) {
        a.data.getComplexFloatArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray)
    }
}

class Create2DComplexDoubleArrayTests : Create2DArrayTestBase<ComplexDouble>() {
    override val zero: ComplexDouble = ComplexDouble.zero
    override val one: ComplexDouble = ComplexDouble.one
    override fun createZeros(): D2Array<ComplexDouble> = mk.zeros<ComplexDouble>(dim1, dim2)
    override fun createOnes(): D2Array<ComplexDouble> = mk.ones<ComplexDouble>(dim1, dim2)
    override fun createIdentity(): D2Array<ComplexDouble> = mk.identity<ComplexDouble>(n)
    override fun createDiagonal(): D2Array<ComplexDouble> = mk.diagonal(List(n) { ComplexDouble(it + 1.0, it + 1.0) })
    override fun expectedDiagonalElement(i: Int): ComplexDouble = ComplexDouble(i + 1.0, i + 1.0)

    override fun makeFromNestedList(): Pair<D2Array<ComplexDouble>, List<List<ComplexDouble>>> {
        val list = listOf(
            listOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i),
            listOf(4.0 + .0.i, 7.0 + .0.i, 2.0 + .0.i)
        )
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D2Array<ComplexDouble>, Set<ComplexDouble>> {
        val set = setOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i)
        return mk.ndarray(set, 2, 3) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D2Array<ComplexDouble>, Any> {
        val array = complexDoubleArrayOf(1.0 + .0.i, 3.0 + .0.i, 8.0 + .0.i, 4.0 + .0.i, 9.0 + .0.i, 2.0 + .0.i)
        return mk.ndarray(array, 2, 3) to array
    }

    override fun createWithInit(): Pair<D2Array<ComplexDouble>, Any> =
        mk.d2array<ComplexDouble>(2, 3) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) } to complexDoubleArrayOf(
            3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i,
            6.21 + 2.168.i, 7.21 + 3.168.i, 8.21 + 4.168.i
        )

    override fun createWithIndices(): Pair<D2Array<ComplexDouble>, Any> =
        mk.d2arrayIndices<ComplexDouble>(2, 3) { i, j -> i * j + ComplexDouble(7) } to complexDoubleArrayOf(
            7.0 + .0.i, 7.0 + .0.i, 7.0 + .0.i, 7.0 + .0.i, 8.0 + .0.i, 9.0 + .0.i
        )

    override fun assertDataEquals(a: D2Array<ComplexDouble>, expected: Any) {
        a.data.getComplexDoubleArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray)
    }
}
