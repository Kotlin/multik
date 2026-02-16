package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.toType
import kotlin.test.Test

class TypeCasting {
    @Test
    fun asType_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val b = a.asType<Double>()
        // dtype is Double, values are [1.0, 2.0, 3.0]
        // SampleEnd
    }

    @Test
    fun dataType_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val c = a.asType<Float>(DataType.FloatDataType)
        // SampleEnd
    }

    @Test
    fun toType_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val d = m.toType<Int, Double, D2>()
        // SampleEnd
    }

    @Test
    fun view_copy_example() {
        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        // SampleStart
        val view = m[0] // view of the first row

        val fullCopy = view.toType<Int, Double, D1>(CopyStrategy.FULL)
        val meaningfulCopy = view.toType<Int, Double, D1>(CopyStrategy.MEANINGFUL)
        // SampleEnd
    }

    @Test
    fun casting_to_complex_example() {
        // SampleStart
        val real = mk.ndarray(mk[1.0, 2.0])
        val complex = real.toType<Double, ComplexDouble, D1>()
        // SampleEnd
    }
}