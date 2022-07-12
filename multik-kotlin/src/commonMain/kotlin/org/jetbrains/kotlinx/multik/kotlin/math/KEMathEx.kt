/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.kotlin.math

import org.jetbrains.kotlinx.multik.api.math.MathEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName
import kotlin.math.*

object KEMathEx : MathEx {

    override fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperation(a) { exp(it) }

    override fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperation(a) { it: Float -> exp(it) }

    override fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperation(a) {
            val expReal = exp(it.re)
            ComplexFloat(expReal * cos(it.im), expReal * sin(it.im))
        }

    override fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperation(a) {
            val expReal = exp(it.re)
            ComplexDouble(expReal * cos(it.im), expReal * sin(it.im))
        }

    override fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperation(a) { ln(it) }

    override fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperation(a) { it: Float -> ln(it) }

    override fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperation(a) { ComplexFloat(it.abs(), it.angle()) }

    override fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperation(a) { ComplexDouble(it.abs(), it.angle()) }

    override fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperation(a) { sin(it) }

    override fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperation(a) { it: Float -> sin(it) }

    override fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperation(a) { ComplexFloat(sin(it.re) * cosh(it.im), cos(it.re) * sinh(it.im)) }

    override fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperation(a) { ComplexDouble(sin(it.re) * cosh(it.im), cos(it.re) * sinh(it.im)) }

    override fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D> =
        mathOperation(a) { cos(it) }

    override fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D> =
        mathOperation(a) { it: Float -> cos(it) }

    override fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D> =
        mathOperation(a) { ComplexFloat(cos(it.re) * cosh(it.im), sin(it.re) * sinh(it.im)) }

    override fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D> =
        mathOperation(a) { ComplexDouble(cos(it.re) * cosh(it.im), sin(it.re) * sinh(it.im)) }

    private fun <T : Number, D : Dimension> mathOperation(
        a: MultiArray<T, D>, function: (Double) -> Double
    ): NDArray<Double, D> {
        val iter = a.iterator()
        val data = initMemoryView(a.size, DataType.DoubleDataType) {
            if (iter.hasNext())
                function(iter.next().toDouble())
            else
                0.0
        }
        return NDArray(data, 0, a.shape, dim = a.dim)
    }

    @JvmName("mathOperationFloat")
    private fun <D : Dimension> mathOperation(
        a: MultiArray<Float, D>, function: (Float) -> Float
    ): NDArray<Float, D> {
        val iter = a.iterator()
        val data = initMemoryView(a.size, DataType.FloatDataType) {
            if (iter.hasNext())
                function(iter.next())
            else
                0f
        }
        return NDArray(data, 0, a.shape, dim = a.dim)
    }

    @JvmName("mathOperationComplex")
    private fun <T : Complex, D : Dimension> mathOperation(
        a: MultiArray<T, D>, function: (T) -> T
    ): NDArray<T, D> {
        val iter = a.iterator()
        val data = initMemoryView(a.size, a.dtype) {
            if (!iter.hasNext()) throw Exception("")
            function(iter.next())
        }
        return NDArray(data, 0, a.shape, dim = a.dim)
    }
}