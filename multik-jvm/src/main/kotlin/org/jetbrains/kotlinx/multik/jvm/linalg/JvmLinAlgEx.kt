/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.toType

public object JvmLinAlgEx : LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> =
        solveCommon(mat, mk.identity(mat.shape[0], mat.dtype), mat.dtype)

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> =
        solveCommon(mat, mk.identity(mat.shape[0], mat.dtype), mat.dtype)

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> =
        solveCommon(mat, mk.identity(mat.shape[0], mat.dtype), mat.dtype)

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> =
        solveCommon(a, b, DataType.DoubleDataType)

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> =
        solveCommon(a, b, DataType.FloatDataType)

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> =
        solveCommon(a, b, a.dtype)

    private fun <T, O : Any, D : Dim2> solveCommon(a: MultiArray<T, D2>, b: MultiArray<T, D>, dtype: DataType): NDArray<O, D> {
        requireSquare(a.shape)
        requireDotShape(a.shape, b.shape)

        val _a = a.toType<T, O, D2>(dtype, CopyStrategy.MEANINGFUL)
        val bTyped = if (dtype == DataType.DoubleDataType) b.toType<T, O, D>(dtype, CopyStrategy.MEANINGFUL) else b
        val _b = (if (bTyped.dim.d == 2) bTyped else bTyped.reshape(bTyped.shape[0], 1)) as MultiArray<T, D2>

        val ans = when (dtype) {
            DataType.DoubleDataType -> solveDouble(_a as D2Array<Double>, _b as D2Array<Double>)
            DataType.FloatDataType -> solveFloat(_a as D2Array<Float>, _b as D2Array<Float>)
            DataType.ComplexDoubleDataType -> solveComplexDouble(_a as D2Array<ComplexDouble>, _b as D2Array<ComplexDouble>)
            DataType.ComplexFloatDataType -> solveComplexFloat(_a as D2Array<ComplexFloat>, _b as D2Array<ComplexFloat>)
            else -> throw UnsupportedOperationException()
        }
        return (if (b.dim.d == 2) ans else ans.reshape(ans.shape[0])) as NDArray<O, D>
    }

    override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> =
        qrDouble(mat.toType())

    override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> =
        qrFloat(mat)

    override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> = when (mat.dtype) {
        DataType.ComplexFloatDataType -> qrComplexFloat(mat as MultiArray<ComplexFloat, D2>)
        DataType.ComplexDoubleDataType -> qrComplexDouble(mat as MultiArray<ComplexDouble, D2>)
        else -> throw UnsupportedOperationException("Matrix should be complex")
    } as Pair<D2Array<T>, D2Array<T>>

    override fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> =
        pluCommon(mat, DataType.DoubleDataType)

    override fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>> =
        pluCommon(mat, DataType.FloatDataType)

    override fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> =
        pluCommon(mat, mat.dtype)

//    override fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> {
//        TODO("Not yet implemented")
//    }

//    override fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> {
//        TODO("Not yet implemented")
//    }

//    override fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> {
//        TODO("Not yet implemented")
//    }

    override fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> =
        eigenValuesCommon(mat, DataType.ComplexDoubleDataType)

    override fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> =
        eigenValuesCommon(mat, DataType.ComplexFloatDataType)

    override fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T> =
        eigenValuesCommon(mat, mat.dtype)

    private fun <T, O : Any> pluCommon(mat: MultiArray<T, D2>, dtype: DataType): Triple<D2Array<O>, D2Array<O>, D2Array<O>> {
        val a = mat.toType<T, O, D2>(dtype, CopyStrategy.MEANINGFUL)
        val (perm, L, U) = pluCompressed(a)

        val P = mk.identity<O>(a.shape[0], dtype)

        for (i in perm.indices.reversed()) {
            if (perm[i] != 0) {
                for (k in 0 until P.shape[1]) {
                    P[i, k] = P[i + perm[i], k].also { P[i + perm[i], k] = P[i, k] }
                }
            }
        }

        return Triple(P, L, U)
    }

    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> = dotMatrix(a, b)

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        dotMatrixComplex(a, b)

    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        dotMatrixToVector(a, b)

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        dotMatrixToVectorComplex(a, b)

    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        dotVecToVec(a, b)

    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        dotVecToVecComplex(a, b)
}