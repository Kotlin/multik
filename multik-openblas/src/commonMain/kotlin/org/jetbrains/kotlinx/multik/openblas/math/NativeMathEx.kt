/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.math

import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.toType

object NativeMathEx : MathEx {

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperationD(a, JniMath::exp)

    override fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperationF(a, JniMath::exp)

    override fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperationCF(a, JniMath::expC)

    override fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperationCD(a, JniMath::expC)

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperationD(a, JniMath::log)

    override fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperationF(a, JniMath::log)

    override fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperationCF(a, JniMath::logC)

    override fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperationCD(a, JniMath::logC)

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperationD(a, JniMath::sin)

    override fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperationF(a, JniMath::sin)

    override fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperationCF(a, JniMath::sinC)

    override fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperationCD(a, JniMath::sinC)

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperationD(a, JniMath::cos)

    override fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperationF(a, JniMath::cos)

    override fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperationCF(a, JniMath::cosC)

    override fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperationCD(a, JniMath::cosC)

    private fun <T : Number, D : Dimension> mathOperationD(
        a: MultiArray<T, D>,
        function: (arr: DoubleArray, size: Int) -> Boolean
    ): NDArray<Double, D> {
        val ret = a.toType<T, Double, D>(CopyStrategy.MEANINGFUL)
        val data = ret.data.getDoubleArray()

        function(data, ret.size)
        return ret
    }

    private fun <D : Dimension> mathOperationF(
        a: MultiArray<Float, D>,
        function: (arr: FloatArray, size: Int) -> Boolean
    ): NDArray<Float, D> {
        val ret = a.deepCopy() as NDArray<Float, D>
        val data = ret.data.getFloatArray()

        function(data, ret.size)
        return ret
    }

    private fun <D : Dimension> mathOperationCF(
        a: MultiArray<ComplexFloat, D>,
        function: (arr: FloatArray, size: Int) -> Boolean
    ): NDArray<ComplexFloat, D> {
        val ret = a.deepCopy() as NDArray<ComplexFloat, D>
        val data = ret.data.getFloatArray()

        function(data, data.size)
        return ret
    }

    private fun <D : Dimension> mathOperationCD(
        a: MultiArray<ComplexDouble, D>,
        function: (arr: DoubleArray, size: Int) -> Boolean
    ): NDArray<ComplexDouble, D> {
        val ret = a.deepCopy() as NDArray<ComplexDouble, D>
        val data = ret.data.getDoubleArray()

        function(data, data.size)
        return ret
    }
}