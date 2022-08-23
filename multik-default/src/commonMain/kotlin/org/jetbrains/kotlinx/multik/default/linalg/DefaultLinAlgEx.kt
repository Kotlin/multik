/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.linalg.Norm
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

public expect object DefaultLinAlgEx: LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2>

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2>

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2>

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D>
    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>
    override fun normF(mat: MultiArray<Float, D2>, norm: Norm): Float
    override fun norm(mat: MultiArray<Double, D2>, norm: Norm): Double
    override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>>
    override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>>
    override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>>
    override fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>>
    override fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>>
    override fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>>
    @ExperimentalMultikApi
    override fun svdF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>>
    @ExperimentalMultikApi
    override fun <T : Number> svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>>
    @ExperimentalMultikApi
    override fun <T : Complex> svdC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>>
    override fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>
    override fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>
    override fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>>
    override fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble>
    override fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat>
    override fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T>
    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}
