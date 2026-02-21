package org.jetbrains.kotlinx.multik.creation

import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.linspace
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ndarrayOf
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.toNDArray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.complexFloatArrayOf
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.minus
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.shouldBe
import kotlin.math.round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

abstract class Create1DArrayTestBase<T : Any> {
    protected val size = 10

    abstract val zero: T
    abstract val one: T

    abstract fun createZeros(): D1Array<T>
    abstract fun createOnes(): D1Array<T>
    abstract fun makeFromList(): Pair<D1Array<T>, List<T>>
    abstract fun makeFromSet(): Pair<D1Array<T>, Set<T>>
    abstract fun makeFromPrimitiveArray(): Pair<D1Array<T>, Any>
    abstract fun createWithInit(): Pair<D1Array<T>, Any>
    abstract fun makeWithNdarrayOf(): Pair<D1Array<T>, Any>
    abstract fun createToNDArray(): Pair<D1Array<T>, List<T>>
    abstract fun assertDataEquals(a: D1Array<T>, expected: Any)

    @Test
    fun createZeroFilledArray() {
        val a = createZeros()
        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == zero } }
    }

    @Test
    fun createArrayFilledWithOnes() {
        val a = createOnes()
        assertEquals(size, a.size)
        assertEquals(size, a.data.size)
        assertTrue { a.all { it == one } }
    }

    @Test
    fun createFromList() {
        val (a, list) = makeFromList()
        assertEquals(list, a.toList())
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
        assertEquals(5, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithInitFunction() {
        val (a, expected) = createWithInit()
        assertEquals(5, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun createWithNdarrayOf() {
        val (a, expected) = makeWithNdarrayOf()
        assertEquals(5, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun convertIterableToNDArray() {
        val (a, expected) = createToNDArray()
        assertEquals(expected.size, a.size)
        assertEquals(expected, a.toList())
    }
}

abstract class Create1DIntegerArrayTestBase<T : Any> : Create1DArrayTestBase<T>() {
    abstract fun arangeWithStep(): Pair<D1Array<T>, Any>
    abstract fun arangeWithNonIntegerStep(): Pair<D1Array<T>, Any>
    abstract fun arangeWithDefaultStart(): Pair<D1Array<T>, Any>
    abstract fun arangeWithDefaultStartAndNonIntegerStep(): Pair<D1Array<T>, Any>
    abstract fun linspaceArray(): Pair<D1Array<T>, Any>
    abstract fun linspaceWithNonIntegerBounds(): Pair<D1Array<T>, Any>

    @Test
    fun generateArangeArrayWithStep() {
        val (a, expected) = arangeWithStep()
        assertEquals(4, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun generateArangeArrayWithNonIntegerStep() {
        val (a, expected) = arangeWithNonIntegerStep()
        assertEquals(3, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun generateArangeArrayWithDefaultStart() {
        val (a, expected) = arangeWithDefaultStart()
        assertEquals(5, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun generateArangeArrayWithDefaultStartAndNonIntegerStep() {
        val (a, expected) = arangeWithDefaultStartAndNonIntegerStep()
        assertEquals(5, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun generateLinspaceArray() {
        val (a, expected) = linspaceArray()
        assertEquals(15, a.size)
        assertDataEquals(a, expected)
    }

    @Test
    fun generateLinspaceArrayWithNonIntegerBounds() {
        val (a, expected) = linspaceWithNonIntegerBounds()
        assertEquals(15, a.size)
        assertDataEquals(a, expected)
    }
}

class Create1DByteArrayTests : Create1DIntegerArrayTestBase<Byte>() {
    override val zero: Byte = 0.toByte()
    override val one: Byte = 1.toByte()
    override fun createZeros(): D1Array<Byte> = mk.zeros(size)
    override fun createOnes(): D1Array<Byte> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Byte>, List<Byte>> {
        val list = listOf<Byte>(1, 3, 8, 4, 9)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Byte>, Set<Byte>> {
        val set = setOf<Byte>(1, 3, 8, 4, 9)
        return mk.ndarray<Byte, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Byte>, Any> {
        val array = byteArrayOf(1, 3, 8, 4, 9)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Byte>, Any> =
        mk.d1array<Byte>(5) { (it + 3).toByte() } to byteArrayOf(3, 4, 5, 6, 7)

    override fun makeWithNdarrayOf(): Pair<D1Array<Byte>, Any> =
        mk.ndarrayOf(1.toByte(), 3.toByte(), 8.toByte(), 4.toByte(), 9.toByte()) to byteArrayOf(1, 3, 8, 4, 9)

    override fun createToNDArray(): Pair<D1Array<Byte>, List<Byte>> {
        val expected = listOf<Byte>(3, 8, 13, 2, 0)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Byte>, expected: Any) {
        a.data.getByteArray() shouldBe (expected as ByteArray)
    }

    override fun arangeWithStep(): Pair<D1Array<Byte>, Any> =
        mk.arange<Byte>(3, 10, step = 2) to byteArrayOf(3, 5, 7, 9)

    override fun arangeWithNonIntegerStep(): Pair<D1Array<Byte>, Any> =
        mk.arange<Byte>(3, 10, step = 2.5) to byteArrayOf(3, 5, 8)

    override fun arangeWithDefaultStart(): Pair<D1Array<Byte>, Any> =
        mk.arange<Byte>(10, step = 2) to byteArrayOf(0, 2, 4, 6, 8)

    override fun arangeWithDefaultStartAndNonIntegerStep(): Pair<D1Array<Byte>, Any> =
        mk.arange<Byte>(10, step = 2.3) to byteArrayOf(0, 2, 4, 6, 9)

    override fun linspaceArray(): Pair<D1Array<Byte>, Any> =
        mk.linspace<Byte>(3, 10, num = 15) to byteArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

    override fun linspaceWithNonIntegerBounds(): Pair<D1Array<Byte>, Any> =
        mk.linspace<Byte>(1.7, 13.8, num = 15) to byteArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)
}

class Create1DShortArrayTests : Create1DIntegerArrayTestBase<Short>() {
    override val zero: Short = 0.toShort()
    override val one: Short = 1.toShort()
    override fun createZeros(): D1Array<Short> = mk.zeros(size)
    override fun createOnes(): D1Array<Short> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Short>, List<Short>> {
        val list = listOf<Short>(1, 3, 8, 4, 9)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Short>, Set<Short>> {
        val set = setOf<Short>(1, 3, 8, 4, 9)
        return mk.ndarray<Short, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Short>, Any> {
        val array = shortArrayOf(1, 3, 8, 4, 9)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Short>, Any> =
        mk.d1array<Short>(5) { (it + 3).toShort() } to shortArrayOf(3, 4, 5, 6, 7)

    override fun makeWithNdarrayOf(): Pair<D1Array<Short>, Any> =
        mk.ndarrayOf(1.toShort(), 3.toShort(), 8.toShort(), 4.toShort(), 9.toShort()) to shortArrayOf(1, 3, 8, 4, 9)

    override fun createToNDArray(): Pair<D1Array<Short>, List<Short>> {
        val expected = listOf<Short>(3, 8, 13, 2, 0)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Short>, expected: Any) {
        a.data.getShortArray() shouldBe (expected as ShortArray)
    }

    override fun arangeWithStep(): Pair<D1Array<Short>, Any> =
        mk.arange<Short>(3, 10, step = 2) to shortArrayOf(3, 5, 7, 9)

    override fun arangeWithNonIntegerStep(): Pair<D1Array<Short>, Any> =
        mk.arange<Short>(3, 10, step = 2.5) to shortArrayOf(3, 5, 8)

    override fun arangeWithDefaultStart(): Pair<D1Array<Short>, Any> =
        mk.arange<Short>(10, step = 2) to shortArrayOf(0, 2, 4, 6, 8)

    override fun arangeWithDefaultStartAndNonIntegerStep(): Pair<D1Array<Short>, Any> =
        mk.arange<Short>(10, step = 2.3) to shortArrayOf(0, 2, 4, 6, 9)

    override fun linspaceArray(): Pair<D1Array<Short>, Any> =
        mk.linspace<Short>(3, 10, num = 15) to shortArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

    override fun linspaceWithNonIntegerBounds(): Pair<D1Array<Short>, Any> =
        mk.linspace<Short>(1.7, 13.8, num = 15) to shortArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)
}

class Create1DIntArrayTests : Create1DIntegerArrayTestBase<Int>() {
    override val zero: Int = 0
    override val one: Int = 1
    override fun createZeros(): D1Array<Int> = mk.zeros(size)
    override fun createOnes(): D1Array<Int> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Int>, List<Int>> {
        val list = listOf(1, 3, 8, 4, 9)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Int>, Set<Int>> {
        val set = setOf(1, 3, 8, 4, 9)
        return mk.ndarray<Int, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Int>, Any> {
        val array = intArrayOf(1, 3, 8, 4, 9)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Int>, Any> =
        mk.d1array<Int>(5) { it + 3 } to intArrayOf(3, 4, 5, 6, 7)

    override fun makeWithNdarrayOf(): Pair<D1Array<Int>, Any> =
        mk.ndarrayOf(1, 3, 8, 4, 9) to intArrayOf(1, 3, 8, 4, 9)

    override fun createToNDArray(): Pair<D1Array<Int>, List<Int>> {
        val expected = listOf(3, 8, 13, 2, 0)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Int>, expected: Any) {
        a.data.getIntArray() shouldBe (expected as IntArray)
    }

    override fun arangeWithStep(): Pair<D1Array<Int>, Any> =
        mk.arange<Int>(3, 10, step = 2) to intArrayOf(3, 5, 7, 9)

    override fun arangeWithNonIntegerStep(): Pair<D1Array<Int>, Any> =
        mk.arange<Int>(3, 10, step = 2.5) to intArrayOf(3, 5, 8)

    override fun arangeWithDefaultStart(): Pair<D1Array<Int>, Any> =
        mk.arange<Int>(10, step = 2) to intArrayOf(0, 2, 4, 6, 8)

    override fun arangeWithDefaultStartAndNonIntegerStep(): Pair<D1Array<Int>, Any> =
        mk.arange<Int>(10, step = 2.3) to intArrayOf(0, 2, 4, 6, 9)

    override fun linspaceArray(): Pair<D1Array<Int>, Any> =
        mk.linspace<Int>(3, 10, num = 15) to intArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

    override fun linspaceWithNonIntegerBounds(): Pair<D1Array<Int>, Any> =
        mk.linspace<Int>(1.7, 13.8, num = 15) to intArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)
}

class Create1DLongArrayTests : Create1DIntegerArrayTestBase<Long>() {
    override val zero: Long = 0L
    override val one: Long = 1L
    override fun createZeros(): D1Array<Long> = mk.zeros(size)
    override fun createOnes(): D1Array<Long> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Long>, List<Long>> {
        val list = listOf<Long>(1, 3, 8, 4, 9)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Long>, Set<Long>> {
        val set = setOf<Long>(1, 3, 8, 4, 9)
        return mk.ndarray<Long, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Long>, Any> {
        val array = longArrayOf(1, 3, 8, 4, 9)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Long>, Any> =
        mk.d1array<Long>(5) { it + 3L } to longArrayOf(3, 4, 5, 6, 7)

    override fun makeWithNdarrayOf(): Pair<D1Array<Long>, Any> =
        mk.ndarrayOf(1L, 3L, 8L, 4L, 9L) to longArrayOf(1, 3, 8, 4, 9)

    override fun createToNDArray(): Pair<D1Array<Long>, List<Long>> {
        val expected = listOf<Long>(3, 8, 13, 2, 0)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Long>, expected: Any) {
        a.data.getLongArray() shouldBe (expected as LongArray)
    }

    override fun arangeWithStep(): Pair<D1Array<Long>, Any> =
        mk.arange<Long>(3, 10, step = 2) to longArrayOf(3, 5, 7, 9)

    override fun arangeWithNonIntegerStep(): Pair<D1Array<Long>, Any> =
        mk.arange<Long>(3, 10, step = 2.5) to longArrayOf(3, 5, 8)

    override fun arangeWithDefaultStart(): Pair<D1Array<Long>, Any> =
        mk.arange<Long>(10, step = 2) to longArrayOf(0, 2, 4, 6, 8)

    override fun arangeWithDefaultStartAndNonIntegerStep(): Pair<D1Array<Long>, Any> =
        mk.arange<Long>(10, step = 2.3) to longArrayOf(0, 2, 4, 6, 9)

    override fun linspaceArray(): Pair<D1Array<Long>, Any> =
        mk.linspace<Long>(3, 10, num = 15) to longArrayOf(3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10)

    override fun linspaceWithNonIntegerBounds(): Pair<D1Array<Long>, Any> =
        mk.linspace<Long>(1.7, 13.8, num = 15) to longArrayOf(1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13)
}

class Create1DFloatArrayTests : Create1DArrayTestBase<Float>() {
    override val zero: Float = 0f
    override val one: Float = 1f
    override fun createZeros(): D1Array<Float> = mk.zeros(size)
    override fun createOnes(): D1Array<Float> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Float>, List<Float>> {
        val list = listOf(1f, 3f, 8f, 4f, 9f)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Float>, Set<Float>> {
        val set = setOf(1f, 3f, 8f, 4f, 9f)
        return mk.ndarray<Float, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Float>, Any> {
        val array = floatArrayOf(1f, 3f, 8f, 4f, 9f)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Float>, Any> =
        mk.d1array<Float>(5) { it + 3f } to floatArrayOf(3f, 4f, 5f, 6f, 7f)

    override fun makeWithNdarrayOf(): Pair<D1Array<Float>, Any> =
        mk.ndarrayOf(1f, 3f, 8f, 4f, 9f) to floatArrayOf(1f, 3f, 8f, 4f, 9f)

    override fun createToNDArray(): Pair<D1Array<Float>, List<Float>> {
        val expected = listOf(3f, 8f, 13f, 2f, 0f)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Float>, expected: Any) {
        a.data.getFloatArray() shouldBe (expected as FloatArray)
    }

    @Test
    fun generateArangeArrayWithStep() {
        val a = mk.arange<Float>(3, 10, step = 2)
        val expected = floatArrayOf(3f, 5f, 7f, 9f)
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithNonIntegerStep() {
        val a = mk.arange<Float>(3, 10, step = 2.5)
        val expected = floatArrayOf(3f, 5.5f, 8f)
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithDefaultStart() {
        val a = mk.arange<Float>(10, step = 2)
        val expected = floatArrayOf(0f, 2f, 4f, 6f, 8f)
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Float>(10, step = 2.3)
        val expected = floatArrayOf(0f, 2.3f, 4.6f, 6.9f, 9.2f)
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    @Test
    fun generateLinspaceArray() {
        val a = mk.linspace<Float>(3, 10, num = 15)
        val expected = floatArrayOf(3f, 3.5f, 4f, 4.5f, 5f, 5.5f, 6f, 6.5f, 7f, 7.5f, 8f, 8.5f, 9f, 9.5f, 10f)
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }

    @Test
    fun generateLinspaceArrayWithNonIntegerBounds() {
        val a = mk.linspace<Float>(1.7, 13.8, num = 15).map { round(it * 1e3f) / 1e3f }
        val expected = floatArrayOf(
            1.7f, 2.564f, 3.429f, 4.293f, 5.157f, 6.021f, 6.886f, 7.75f,
            8.614f, 9.479f, 10.343f, 11.207f, 12.071f, 12.936f, 13.8f
        )
        assertEquals(expected.size, a.size)
        a.data.getFloatArray() shouldBe expected
    }
}

class Create1DDoubleArrayTests : Create1DArrayTestBase<Double>() {
    override val zero: Double = 0.0
    override val one: Double = 1.0
    override fun createZeros(): D1Array<Double> = mk.zeros(size)
    override fun createOnes(): D1Array<Double> = mk.ones(size)

    override fun makeFromList(): Pair<D1Array<Double>, List<Double>> {
        val list = listOf(1.0, 3.0, 8.0, 4.0, 9.0)
        return mk.ndarray(list) to list
    }

    override fun makeFromSet(): Pair<D1Array<Double>, Set<Double>> {
        val set = setOf(1.0, 3.0, 8.0, 4.0, 9.0)
        return mk.ndarray<Double, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<Double>, Any> {
        val array = doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0)
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<Double>, Any> =
        mk.d1array<Double>(5) { it + 3.0 } to doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0)

    override fun makeWithNdarrayOf(): Pair<D1Array<Double>, Any> =
        mk.ndarrayOf(1.0, 3.0, 8.0, 4.0, 9.0) to doubleArrayOf(1.0, 3.0, 8.0, 4.0, 9.0)

    override fun createToNDArray(): Pair<D1Array<Double>, List<Double>> {
        val expected = listOf(3.0, 8.0, 13.0, 2.0, 0.0)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<Double>, expected: Any) {
        a.data.getDoubleArray() shouldBe (expected as DoubleArray)
    }

    @Test
    fun generateArangeArrayWithStep() {
        val a = mk.arange<Double>(3, 10, step = 2)
        val expected = doubleArrayOf(3.0, 5.0, 7.0, 9.0)
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithNonIntegerStep() {
        val a = mk.arange<Double>(3, 10, step = 2.5)
        val expected = doubleArrayOf(3.0, 5.5, 8.0)
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithDefaultStart() {
        val a = mk.arange<Double>(10, step = 2)
        val expected = doubleArrayOf(0.0, 2.0, 4.0, 6.0, 8.0)
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    @Test
    fun generateArangeArrayWithDefaultStartAndNonIntegerStep() {
        val a = mk.arange<Double>(10, step = 2.3).map { round(it * 1e3) / 1e3 }
        val expected = doubleArrayOf(0.0, 2.3, 4.6, 6.9, 9.2)
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    @Test
    fun generateLinspaceArray() {
        val a = mk.linspace<Double>(3, 10, num = 15)
        val expected = doubleArrayOf(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }

    @Test
    fun generateLinspaceArrayWithNonIntegerBounds() {
        val a = mk.linspace<Double>(1.7, 13.8, num = 15).map { round(it * 1e5) / 1e5 }
        val expected = doubleArrayOf(
            1.7, 2.56429, 3.42857, 4.29286, 5.15714, 6.02143, 6.88571, 7.75,
            8.61429, 9.47857, 10.34286, 11.20714, 12.07143, 12.93571, 13.8
        )
        assertEquals(expected.size, a.size)
        a.data.getDoubleArray() shouldBe expected
    }
}

class Create1DComplexFloatArrayTests : Create1DArrayTestBase<ComplexFloat>() {
    override val zero: ComplexFloat = ComplexFloat.zero
    override val one: ComplexFloat = ComplexFloat.one
    override fun createZeros(): D1Array<ComplexFloat> = mk.zeros(size)
    override fun createOnes(): D1Array<ComplexFloat> = mk.ones(size)

    private val complexFloatList: List<ComplexFloat> = listOf(
        ComplexFloat(1.032f, 21.4214f),
        ComplexFloat(3.2323f, 4.35903f),
        ComplexFloat(.3498f, 5.49230f),
        ComplexFloat(4.43285f, 8.5382f),
        ComplexFloat(.34829f, 3.2389f)
    )

    override fun makeFromList(): Pair<D1Array<ComplexFloat>, List<ComplexFloat>> =
        mk.ndarray(complexFloatList) to complexFloatList

    override fun makeFromSet(): Pair<D1Array<ComplexFloat>, Set<ComplexFloat>> {
        val set = complexFloatList.toSet()
        return mk.ndarray<ComplexFloat, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<ComplexFloat>, Any> {
        val array = complexFloatList.toComplexFloatArray()
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<ComplexFloat>, Any> =
        mk.d1array<ComplexFloat>(5) { ComplexFloat(it + 3.21f, round((it - .832f) * 1e5f) / 1e5f) } to
            complexFloatArrayOf(3.21f - .832f.i, 4.21f + 0.168f.i, 5.21f + 1.168f.i, 6.21f + 2.168f.i, 7.21f + 3.168f.i)

    override fun makeWithNdarrayOf(): Pair<D1Array<ComplexFloat>, Any> =
        mk.ndarrayOf(
            1.32f + 2.328f.i, 4.231f + 5.83f.i, 6.02f + 7.437f.i,
            8.372f + 9.139f.i, 10.0f + 11.7453f.i
        ) to complexFloatArrayOf(
            1.32f + 2.328f.i, 4.231f + 5.83f.i, 6.02f + 7.437f.i,
            8.372f + 9.139f.i, 10.0f + 11.7453f.i
        )

    override fun createToNDArray(): Pair<D1Array<ComplexFloat>, List<ComplexFloat>> {
        val expected = listOf(
            ComplexFloat(1.32f, 2.328f),
            ComplexFloat(4.231f, 5.83f),
            ComplexFloat(6.02f, 7.437f),
            ComplexFloat(8.372f, 9.139f),
            ComplexFloat(10f, 11.7453f)
        )
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<ComplexFloat>, expected: Any) {
        a.data.getComplexFloatArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray)
    }
}

class Create1DComplexDoubleArrayTests : Create1DArrayTestBase<ComplexDouble>() {
    override val zero: ComplexDouble = ComplexDouble.zero
    override val one: ComplexDouble = ComplexDouble.one
    override fun createZeros(): D1Array<ComplexDouble> = mk.zeros(size)
    override fun createOnes(): D1Array<ComplexDouble> = mk.ones(size)

    private val complexDoubleList: List<ComplexDouble> = listOf(
        ComplexDouble(1.032, 21.4214),
        ComplexDouble(3.2323, 4.35903),
        ComplexDouble(.3498, 5.49230),
        ComplexDouble(4.43285, 8.5382),
        ComplexDouble(.34829, 3.2389)
    )

    override fun makeFromList(): Pair<D1Array<ComplexDouble>, List<ComplexDouble>> =
        mk.ndarray(complexDoubleList) to complexDoubleList

    override fun makeFromSet(): Pair<D1Array<ComplexDouble>, Set<ComplexDouble>> {
        val set = complexDoubleList.toSet()
        return mk.ndarray<ComplexDouble, D1>(set, shape = intArrayOf(5)) to set
    }

    override fun makeFromPrimitiveArray(): Pair<D1Array<ComplexDouble>, Any> {
        val array = complexDoubleList.toComplexDoubleArray()
        return mk.ndarray(array) to array
    }

    override fun createWithInit(): Pair<D1Array<ComplexDouble>, Any> =
        mk.d1array<ComplexDouble>(5) { ComplexDouble(it + 3.21, round((it - .832) * 1e5) / 1e5) } to
            complexDoubleArrayOf(3.21 - .832.i, 4.21 + 0.168.i, 5.21 + 1.168.i, 6.21 + 2.168.i, 7.21 + 3.168.i)

    override fun makeWithNdarrayOf(): Pair<D1Array<ComplexDouble>, Any> =
        mk.ndarrayOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i) to
            complexDoubleArrayOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i)

    override fun createToNDArray(): Pair<D1Array<ComplexDouble>, List<ComplexDouble>> {
        val expected = listOf(1.32 + 2.328.i, 4.231 + 5.83.i, 6.02 + 7.437.i, 8.372 + 9.139.i, 10.0 + 11.7453.i)
        return expected.toNDArray() to expected
    }

    override fun assertDataEquals(a: D1Array<ComplexDouble>, expected: Any) {
        a.data.getComplexDoubleArray() shouldBe (expected as org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray)
    }
}
