package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

public interface LinAlgEx {
    public fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2>
    public fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2>
    public fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2>

    public fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>
    public fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D>
    public fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    public fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>>
    public fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>>
    public fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>>

    public fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>>
    public fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>>
    public fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>>

    public fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>
    public fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>
    public fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>>

    public fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble>
    public fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat>
    public fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T>

    public fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
    public fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
    public fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
    public fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
    public fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
    public fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}