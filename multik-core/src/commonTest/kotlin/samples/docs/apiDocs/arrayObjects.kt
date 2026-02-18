package samples.docs.apiDocs

import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.test.Test

class ArrayObjects {
    @Test
    fun creating_complex_values_example() {
        // SampleStart
        val c1 = ComplexFloat(1.0f, 2.0f)   // 1.0 + 2.0i
        val c2 = ComplexDouble(3.0, -1.0)   // 3.0 - 1.0i

        // Factory helpers on Complex companion
        val real = Complex.r(5.0)           // 5.0 + 0.0i
        val imag = Complex.i(3.0)           // 0.0 + 3.0i
        // SampleEnd
    }

    @Test
    fun creating_complex_array_example() {
        // SampleStart
        val arr = ComplexFloatArray(3) { ComplexFloat(it.toFloat(), 0f) }
        arr[0] // ComplexFloat(0.0, 0.0)
        // SampleEnd
    }

    @Test
    fun type_example() {
        // SampleStart
        val a = mk.ndarray(mk[1.0, 2.0, 3.0])

        a.dtype                          // DataType.DoubleDataType
        a.dtype.nativeCode               // 6
        a.dtype.itemSize                 // 8
        a.dtype.isNumber()               // true
        a.dtype.isComplex()              // false

        DataType.of(3)                   // DataType.IntDataType
        DataType.ofKClass(Float::class)  // DataType.FloatDataType
        // SampleEnd
    }

    @Test
    fun dim_example() {
        // SampleStart
        val v: D1Array<Int> = mk.ndarray(mk[1, 2, 3])
        v.dim   // D1
        v.dim.d // 1

        val m: D2Array<Double> = mk.zeros<Double>(3, 4)
        m.dim   // D2
        m.dim.d // 2
        // SampleEnd
    }

    @Test
    fun change_dim_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])                             // D2
        val flat = m.flatten()                                                       // D1
        val reshaped = m.reshape(1, 2, 2)                       // D3
        val squeezed = m.reshape(1, 2, 2).squeeze(0)   // DN (d=2)
        // SampleEnd
    }

    @Test
    fun slice_rInt_example() {
        // SampleStart
        val s = 0.r..4.r          // Slice(0, 4, 1)
        val s2 = 0.r until 4.r    // Slice(0, 3, 1)
        // SampleEnd
    }

    @Test
    fun slice_sl_example() {
        // SampleStart
        val a = mk.ndarray(mk[10, 20, 30, 40, 50])
        a[sl.first..2]  // [10, 20, 30]
        a[2..sl.last]   // [30, 40, 50]
        a[sl.bounds]    // [10, 20, 30, 40, 50]
        // SampleEnd
    }

    @Test
    fun view_v_property_example() {
        // SampleStart
        val tensor = mk.d3array(2, 3, 4) { it }
        val sub = tensor.asDNArray().V[0]  // DN array of shape (3, 4)
        // SampleEnd
    }

    @Test
    fun view_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
        val row: MultiArray<Int, D1> = m.view(1)          // [4, 5, 6]
        // SampleEnd
    }

    @Test
    fun slice_by_range_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
        val cols = a.slice<Int, D2, D2>(0..1, axis = 1) // columns 0 and 1
        // SampleEnd
    }

    @Test
    fun slice_by_map_example() {
        // SampleStart
        val t = mk.d3array(2, 3, 4) { it }
        val s = t.slice<Int, D3, D2>(mapOf(0 to 0.r, 2 to (0..2).toSlice()))
        // axis 0 fixed to index 0 (removed), axis 2 sliced to 0..2 (kept)
        // SampleEnd
    }

    @Test
    fun creating_progression_example() {
        // SampleStart
        val start = intArrayOf(0, 0)
        val end = intArrayOf(1, 2)

        val prog = start..end                    // forward, step 1
        val stepped = (start..end).step(2)       // forward, step 2
        val rev = end downTo start               // backward, step 1
        // SampleEnd
    }

    @Test
    fun forEach_progression_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
        m.multiIndices.forEach { idx ->
            // idx is IntArray, e.g. [0, 0], [0, 1], ...
        }
        // SampleEnd
    }
}