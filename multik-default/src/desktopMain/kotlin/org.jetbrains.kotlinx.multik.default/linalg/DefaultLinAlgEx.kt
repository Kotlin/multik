/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.api.KEEngineType
import org.jetbrains.kotlinx.multik.api.NativeEngineType
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.linalg.Norm
import org.jetbrains.kotlinx.multik.default.DefaultEngineFactory
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultLinAlgEx : LinAlgEx {

    private val ktLinAlgEx = DefaultEngineFactory.getEngine(KEEngineType).getLinAlg().linAlgEx
    private val natLinAlgEx = DefaultEngineFactory.getEngine(NativeEngineType).getLinAlg().linAlgEx

    actual override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> = natLinAlgEx.inv(mat)

    actual override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> = natLinAlgEx.invF(mat)

    actual override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> = natLinAlgEx.invC(mat)

    actual override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> =
        natLinAlgEx.solve(a, b)

    actual override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> =
        natLinAlgEx.solveF(a, b)

    actual override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> =
        natLinAlgEx.solveC(a, b)

    actual override fun normF(mat: MultiArray<Float, D2>, norm: Norm): Float = natLinAlgEx.normF(mat, norm)

    actual override fun norm(mat: MultiArray<Double, D2>, norm: Norm): Double = natLinAlgEx.norm(mat, norm)

    actual override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> =
        natLinAlgEx.qr(mat)

    actual override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> = natLinAlgEx.qrF(mat)

    actual override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> =
        natLinAlgEx.qrC(mat)

    actual override fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> =
        natLinAlgEx.plu(mat)

    actual override fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>> =
        natLinAlgEx.pluF(mat)

    actual override fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> =
        natLinAlgEx.pluC(mat)

    @ExperimentalMultikApi
    actual override fun svdF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>> =
        ktLinAlgEx.svdF(mat)

    @ExperimentalMultikApi
    actual override fun <T : Number> svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>> =
        ktLinAlgEx.svd(mat)

    @ExperimentalMultikApi
    actual override fun <T : Complex> svdC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>> =
        ktLinAlgEx.svdC(mat)

    actual override fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
        ktLinAlgEx.eig(mat)

    actual override fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
        ktLinAlgEx.eigF(mat)

    actual override fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> =
        ktLinAlgEx.eigC(mat)

    actual override fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> =
        ktLinAlgEx.eigVals(mat)

    actual override fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> = ktLinAlgEx.eigValsF(mat)

    actual override fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T> = ktLinAlgEx.eigValsC(mat)

    actual override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> natLinAlgEx.dotMM(a, b)
            else -> ktLinAlgEx.dotMM(a, b)
        }

    actual override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        natLinAlgEx.dotMMComplex(a, b)

    actual override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> natLinAlgEx.dotMV(a, b)
            else -> ktLinAlgEx.dotMV(a, b)
        }

    actual override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        natLinAlgEx.dotMVComplex(a, b)

    actual override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> natLinAlgEx.dotVV(a, b)
            else -> ktLinAlgEx.dotVV(a, b)
        }

    actual override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        natLinAlgEx.dotVVComplex(a, b)
}