package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.complex.*
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.toType
import kotlin.test.Test

class ComplexNumbers {
    @Test
    fun creating_complex_values() {
        // SampleStart
        val z1 = ComplexFloat(1f, -2f)
        val z2 = ComplexDouble(3.0, 4.0)

        val i1 = 1f.i      // 0 + 1i (ComplexFloat)
        val i2 = 2.0.i     // 0 + 2i (ComplexDouble)

        val real = 5.0.toComplexDouble() // 5 + 0i
        // SampleEnd
    }

    @Test
    fun complex_helpers() {
        // SampleStart
        val z = ComplexDouble(1.0, -2.0)
        val conj = z.conjugate()
        val magnitude = z.abs()
        val angle = z.angle()
        // SampleEnd
    }

    @Test
    fun creating_complex_ndarrays() {
        // SampleStart
        val a = mk.ndarray(
            mk[
                ComplexDouble(1.0, 2.0),
                ComplexDouble(3.0, -1.0)
            ]
        )

        val b = mk.ones<ComplexFloat>(2, 2)
        // SampleEnd
    }

    @Test
    fun from_complex_primitive_arrays() {
        // SampleStart
        val data = complexDoubleArrayOf(1.0 + 2.0.i, 3.0 + 4.0.i, 5.0 + 6.0.i)
        val c = mk.ndarray(data)
        // SampleEnd
    }

    @Test
    fun complex_grid_factory() {
        // SampleStart
        val grid = mk.d2arrayIndices(2, 3) { i, j -> ComplexFloat(i.toFloat(), j.toFloat()) }
        // SampleEnd
    }

    @Test
    fun complex_arithmetic() {
        // SampleStart
        val x = mk.ndarray(mk[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, -1.0)])
        val y = mk.ndarray(mk[ComplexDouble(0.5, 0.0), ComplexDouble(1.0, 2.0)])

        val sum = x + y
        val product = x * y
        // SampleEnd
    }

    @Test
    fun complex_scalar_operations() {
        val x = mk.ndarray(mk[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, -1.0)])
        // SampleStart
        val scaled = x + 2.0.toComplexDouble()
        // SampleEnd
    }

    @Test
    fun real_to_complex_conversion() {
        // SampleStart
        val real = mk.ndarray(mk[1.0, 2.0, 3.0])
        val complex = real.toType<Double, ComplexDouble, D1>()
        // SampleEnd
    }
}