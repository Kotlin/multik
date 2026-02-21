package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.dnarray
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.complexFloatArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.minus
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

abstract class CreateNDArrayTestBase<T : Any> {
    protected val dim1 = 5
    protected val dim2 = 7
    protected val dim3 = 3
    protected val dim4 = 2
    protected val dim5 = 3
    protected val random = Random(42)

    abstract val zero: T
    abstract val one: T

    abstract fun createZeros(): NDArray<T, DN>
    abstract fun createOnes(): NDArray<T, DN>
    abstract fun makeFromSet(): Pair<NDArray<T, DN>, Set<T>>
    abstract fun makeFromPrimitiveArray(): Pair<NDArray<T, DN>, Any>
    abstract fun createWithInitFunction4D(): Pair<NDArray<T, DN>, Any>
    abstract fun createWithInitFunction5D(): Pair<NDArray<T, DN>, Any>
    abstract fun assertDataEquals(a: NDArray<T, DN>, expected: Any)

    @Test
    fun createZeroFilledArray() {
        val a = createZeros()
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == zero } }
    }

    @Test
    fun createArrayFilledWithOnes() {
        val a = createOnes()
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.size)
        assertEquals(dim1 * dim2 * dim3 * dim4 * dim5, a.data.size)
        assertTrue { a.all { it == one } }
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
        assertEquals(36, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitFunctionWith4D() {
        val (a, expected) = createWithInitFunction4D()
        assertEquals(12, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitFunctionWith5D() {
        val (a, expected) = createWithInitFunction5D()
        assertEquals(16, a.size)
        assertDataEquals(a, expected)
    }
}

class CreateNDByteArrayTests : CreateNDArrayTestBase<Byte>() {
    override val zero: Byte = 0.toByte()
    override val one: Byte = 1.toByte()
    override fun createZeros(): NDArray<Byte, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Byte, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Byte, DN>, Set<Byte>> {
        val set = (1..36).map { it.toByte() }.toSet()
        return mk.ndarray<Byte, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Byte, DN>, Any> {
        val array = ByteArray(36) { random.nextInt(101).toByte() }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Byte, DN>, Any> =
        mk.dnarray<Byte>(2, 3, 1, 2) { (it + 3).toByte() } to byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithInitFunction5D(): Pair<NDArray<Byte, DN>, Any> =
        mk.dnarray<Byte>(2, 2, 1, 2, 2) { (it + 3).toByte() } to byteArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    override fun assertDataEquals(a: NDArray<Byte, DN>, expected: Any) {
        a.data.getByteArray() shouldBe (expected as ByteArray)
    }
}

class CreateNDShortArrayTests : CreateNDArrayTestBase<Short>() {
    override val zero: Short = 0.toShort()
    override val one: Short = 1.toShort()
    override fun createZeros(): NDArray<Short, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Short, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Short, DN>, Set<Short>> {
        val set = (1..36).map { it.toShort() }.toSet()
        return mk.ndarray<Short, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Short, DN>, Any> {
        val array = ShortArray(36) { random.nextInt(101).toShort() }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Short, DN>, Any> =
        mk.dnarray<Short>(2, 3, 1, 2) { (it + 3).toShort() } to shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithInitFunction5D(): Pair<NDArray<Short, DN>, Any> =
        mk.dnarray<Short>(2, 2, 1, 2, 2) { (it + 3).toShort() } to shortArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    override fun assertDataEquals(a: NDArray<Short, DN>, expected: Any) {
        a.data.getShortArray() shouldBe (expected as ShortArray)
    }
}

class CreateNDIntArrayTests : CreateNDArrayTestBase<Int>() {
    override val zero: Int = 0
    override val one: Int = 1
    override fun createZeros(): NDArray<Int, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Int, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Int, DN>, Set<Int>> {
        val set = (1..36).toSet()
        return mk.ndarray<Int, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Int, DN>, Any> {
        val array = IntArray(36) { random.nextInt(101) }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Int, DN>, Any> =
        mk.dnarray<Int>(2, 3, 1, 2) { it + 3 } to intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithInitFunction5D(): Pair<NDArray<Int, DN>, Any> =
        mk.dnarray<Int>(2, 2, 1, 2, 2) { it + 3 } to intArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    override fun assertDataEquals(a: NDArray<Int, DN>, expected: Any) {
        a.data.getIntArray() shouldBe (expected as IntArray)
    }
}

class CreateNDLongArrayTests : CreateNDArrayTestBase<Long>() {
    override val zero: Long = 0L
    override val one: Long = 1L
    override fun createZeros(): NDArray<Long, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Long, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Long, DN>, Set<Long>> {
        val set = (1..36).map { it.toLong() }.toSet()
        return mk.ndarray<Long, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Long, DN>, Any> {
        val array = LongArray(36) { random.nextLong(101) }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Long, DN>, Any> =
        mk.dnarray<Long>(2, 3, 1, 2) { it + 3L } to longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    override fun createWithInitFunction5D(): Pair<NDArray<Long, DN>, Any> =
        mk.dnarray<Long>(2, 2, 1, 2, 2) { it + 3L } to longArrayOf(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    override fun assertDataEquals(a: NDArray<Long, DN>, expected: Any) {
        a.data.getLongArray() shouldBe (expected as LongArray)
    }
}

class CreateNDFloatArrayTests : CreateNDArrayTestBase<Float>() {
    override val zero: Float = 0f
    override val one: Float = 1f
    override fun createZeros(): NDArray<Float, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Float, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Float, DN>, Set<Float>> {
        val set = (1..36).map { it.toFloat() }.toSet()
        return mk.ndarray<Float, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Float, DN>, Any> {
        val array = FloatArray(36) { random.nextFloat() }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Float, DN>, Any> =
        mk.dnarray<Float>(2, 3, 1, 2) { it + 3f } to floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f)

    override fun createWithInitFunction5D(): Pair<NDArray<Float, DN>, Any> =
        mk.dnarray<Float>(2, 2, 1, 2, 2) { it + 3f } to floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f)

    override fun assertDataEquals(a: NDArray<Float, DN>, expected: Any) {
        a.data.getFloatArray() shouldBe (expected as FloatArray)
    }
}

class CreateNDDoubleArrayTests : CreateNDArrayTestBase<Double>() {
    override val zero: Double = 0.0
    override val one: Double = 1.0
    override fun createZeros(): NDArray<Double, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<Double, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<Double, DN>, Set<Double>> {
        val set = (1..36).map { it.toDouble() }.toSet()
        return mk.ndarray<Double, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<Double, DN>, Any> {
        val array = DoubleArray(36) { random.nextDouble(101.0) }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<Double, DN>, Any> =
        mk.dnarray<Double>(2, 3, 1, 2) { it + 3.0 } to doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)

    override fun createWithInitFunction5D(): Pair<NDArray<Double, DN>, Any> =
        mk.dnarray<Double>(2, 2, 1, 2, 2) { it + 3.0 } to doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)

    override fun assertDataEquals(a: NDArray<Double, DN>, expected: Any) {
        a.data.getDoubleArray() shouldBe (expected as DoubleArray)
    }
}

class CreateNDComplexFloatArrayTests : CreateNDArrayTestBase<ComplexFloat>() {
    override val zero: ComplexFloat = ComplexFloat.zero
    override val one: ComplexFloat = ComplexFloat.one
    override fun createZeros(): NDArray<ComplexFloat, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<ComplexFloat, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<ComplexFloat, DN>, Set<ComplexFloat>> {
        val set = (1..36).map { ComplexFloat(it, it + 1) }.toSet()
        return mk.ndarray<ComplexFloat, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<ComplexFloat, DN>, Any> {
        val array = ComplexFloatArray(36) { ComplexFloat(random.nextFloat(), random.nextFloat()) }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<ComplexFloat, DN>, Any> =
        mk.dnarray<ComplexFloat>(2, 3, 1, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) } to complexFloatArrayOf(
            3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i,
            6.21f + 2.168f.i, 7.21f + 3.168f.i, 8.21f + 4.168f.i,
            9.21f + 5.168f.i, 10.21f + 6.168f.i, 11.21f + 7.168f.i,
            12.21f + 8.168f.i, 13.21f + 9.168f.i, 14.21f + 10.168f.i
        )

    override fun createWithInitFunction5D(): Pair<NDArray<ComplexFloat, DN>, Any> =
        mk.dnarray<ComplexFloat>(2, 2, 1, 2, 2) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) } to complexFloatArrayOf(
            3.21f - 0.832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i, 6.21f + 2.168f.i,
            7.21f + 3.168f.i, 8.21f + 4.168f.i, 9.21f + 5.168f.i, 10.21f + 6.168f.i,
            11.21f + 7.168f.i, 12.21f + 8.168f.i, 13.21f + 9.168f.i, 14.21f + 10.168f.i,
            15.21f + 11.168f.i, 16.21f + 12.168f.i, 17.21f + 13.168f.i, 18.21f + 14.168f.i
        )

    override fun assertDataEquals(a: NDArray<ComplexFloat, DN>, expected: Any) {
        a.data.getComplexFloatArray() shouldBe (expected as ComplexFloatArray)
    }
}

class CreateNDComplexDoubleArrayTests : CreateNDArrayTestBase<ComplexDouble>() {
    override val zero: ComplexDouble = ComplexDouble.zero
    override val one: ComplexDouble = ComplexDouble.one
    override fun createZeros(): NDArray<ComplexDouble, DN> = mk.zeros(dim1, dim2, dim3, dim4, dim5)
    override fun createOnes(): NDArray<ComplexDouble, DN> = mk.ones(dim1, dim2, dim3, dim4, dim5)

    override fun makeFromSet(): Pair<NDArray<ComplexDouble, DN>, Set<ComplexDouble>> {
        val set = (1..36).map { ComplexDouble(it, it + 1) }.toSet()
        return mk.ndarray<ComplexDouble, DN>(set, shape = intArrayOf(2, 3, 1, 2, 3)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<NDArray<ComplexDouble, DN>, Any> {
        val array = ComplexDoubleArray(36) { ComplexDouble(random.nextDouble(101.0), random.nextDouble(10.0)) }
        return mk.ndarray(array, 2, 3, 1, 2, 3) to array
    }

    override fun createWithInitFunction4D(): Pair<NDArray<ComplexDouble, DN>, Any> =
        mk.dnarray<ComplexDouble>(2, 3, 1, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) } to complexDoubleArrayOf(
            3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i,
            6.21 + 2.168.i, 7.21 + 3.168.i, 8.21 + 4.168.i,
            9.21 + 5.168.i, 10.21 + 6.168.i, 11.21 + 7.168.i,
            12.21 + 8.168.i, 13.21 + 9.168.i, 14.21 + 10.168.i
        )

    override fun createWithInitFunction5D(): Pair<NDArray<ComplexDouble, DN>, Any> =
        mk.dnarray<ComplexDouble>(2, 2, 1, 2, 2) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) } to complexDoubleArrayOf(
            3.21 - 0.832.i, 4.21 + 0.168.i, 5.21 + 1.168.i, 6.21 + 2.168.i,
            7.21 + 3.168.i, 8.21 + 4.168.i, 9.21 + 5.168.i, 10.21 + 6.168.i,
            11.21 + 7.168.i, 12.21 + 8.168.i, 13.21 + 9.168.i, 14.21 + 10.168.i,
            15.21 + 11.168.i, 16.21 + 12.168.i, 17.21 + 13.168.i, 18.21 + 14.168.i
        )

    override fun assertDataEquals(a: NDArray<ComplexDouble, DN>, expected: Any) {
        a.data.getComplexDoubleArray() shouldBe (expected as ComplexDoubleArray)
    }
}
