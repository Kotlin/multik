/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

public actual object DefaultLinAlgEx : LinAlgEx {
    actual override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> = JvmLinAlgEx.inv(mat)

    actual override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> = JvmLinAlgEx.invF(mat)

    actual override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> = JvmLinAlgEx.invC(mat)

    actual override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> =
        JvmLinAlgEx.solve(a, b)

    actual override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> =
        JvmLinAlgEx.solveF(a, b)

    actual override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> =
        JvmLinAlgEx.solveC(a, b)

    actual override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> =
        JvmLinAlgEx.qr(mat)

    actual override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> = JvmLinAlgEx.qrF(mat)

    actual override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> = JvmLinAlgEx.qrC(mat)

    actual override fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> =
        JvmLinAlgEx.plu(mat)

    actual override fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>> =
        JvmLinAlgEx.pluF(mat)

    actual override fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> =
        JvmLinAlgEx.pluC(mat)

    actual override fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
        JvmLinAlgEx.eig(mat)

    actual override fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
        JvmLinAlgEx.eigF(mat)

    actual override fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> = JvmLinAlgEx.eigC(mat)

    actual override fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> = JvmLinAlgEx.eigVals(mat)

    actual override fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> = JvmLinAlgEx.eigValsF(mat)

    actual override fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T> = JvmLinAlgEx.eigValsC(mat)

    actual override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        JvmLinAlg.dot(a, b)

    actual override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        JvmLinAlg.dot(a, b)

    actual override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        JvmLinAlg.dot(a, b)

    actual override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        JvmLinAlg.dot(a, b)

    actual override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = JvmLinAlg.dot(a, b)

    actual override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T = JvmLinAlg.dot(a, b)
}