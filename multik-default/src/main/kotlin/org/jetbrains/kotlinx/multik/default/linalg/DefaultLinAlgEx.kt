package org.jetbrains.kotlinx.multik.default.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.jni.NativeEngine
import org.jetbrains.kotlinx.multik.jni.linalg.NativeLinAlg
import org.jetbrains.kotlinx.multik.jni.linalg.NativeLinAlgEx
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object DefaultLinAlgEx : LinAlgEx {

    init {
        NativeEngine
    }

    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> =
        NativeLinAlgEx.solve(a, b)

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> =
        NativeLinAlgEx.solveF(a, b)

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> =
        NativeLinAlgEx.solveC(a, b)

    override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> =
        NativeLinAlgEx.qr(mat)

    override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> =
        NativeLinAlgEx.qrF(mat)

    override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> =
        NativeLinAlgEx.qrC(mat)

    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> JvmLinAlg.dot(a, b)
        }

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        NativeLinAlg.dot(a, b)

    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> JvmLinAlg.dot(a, b)
        }

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        NativeLinAlg.dot(a, b)

    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        when (a.dtype) {
            DataType.FloatDataType, DataType.DoubleDataType -> NativeLinAlg.dot(a, b)
            else -> JvmLinAlg.dot(a, b)
        }

    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T =
        NativeLinAlg.dot(a, b)
}