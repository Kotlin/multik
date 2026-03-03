package org.jetbrains.kotlinx.multik.openblas.stat

import org.jetbrains.kotlinx.multik.api.stat.Statistics
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.D4
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.openblas.math.NativeMath

internal object NativeStatistics : Statistics {

    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? {
        val size = a.size
        return when {
            size == 1 -> a.first().toDouble()
            size > 1 -> JniStat.median(a.deepCopy().data.data, size, a.dtype.nativeCode)
            else -> null
        }
    }

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double {
        if (weights == null) return mean(a)
        return NativeMath.sum(a * weights).toDouble() / NativeMath.sum(weights).toDouble()
    }

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double {
        val ret = NativeMath.sum(a)
        return ret.toDouble() / a.size
    }

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): NDArray<Double, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, DN> {
        TODO("Not yet implemented")
    }
}
