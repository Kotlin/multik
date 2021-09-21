package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toType

public object JvmLinAlgEx : LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> =
        invDouble(if (mat.dtype == DataType.DoubleDataType) mat as MultiArray<Double, D2> else mat.toType())

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> =
        invFloat(mat)

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        return when (mat.dtype) {
            DataType.ComplexDoubleDataType -> invComplexDouble(mat as MultiArray<ComplexDouble, D2>)
            DataType.ComplexFloatDataType -> invComplexFloat(mat as MultiArray<ComplexFloat, D2>)
            else -> throw UnsupportedOperationException()
        } as NDArray<T, D2>
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> {
        val aDouble = if (a.dtype == DataType.DoubleDataType) a as MultiArray<Double, D2> else a.toType()

        val bDouble = (if (b.dim.d == 2) b else b.reshape(b.shape[0], 1)) as MultiArray<Double, D2>

        val ans = solveDouble(aDouble, bDouble)
        return (if (b.dim.d == 2) ans else ans.reshape(ans.shape[0])) as NDArray<Double, D>
    }

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> {
        val bFloat = (if (b.dim.d == 2) b else b.reshape(b.shape[0], 1)) as MultiArray<Float, D2>

        val ans = solveFloat(a, bFloat)
        return (if (b.dim.d == 2) ans else ans.reshape(ans.shape[0])) as NDArray<Float, D>
    }

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        if (a.dtype != b.dtype) throw UnsupportedOperationException("Argument matrices should have same datatype")

        return when (a.dtype) {
            DataType.ComplexDoubleDataType -> {
                val bComplexDouble =
                    (if (b.dim.d == 2) b else b.reshape(b.shape[0], 1)) as MultiArray<ComplexDouble, D2>
                val ans = solveComplexDouble(a as MultiArray<ComplexDouble, D2>, bComplexDouble)
                if (b.dim.d == 2) ans else ans.reshape(ans.shape[0])
            }
            DataType.ComplexFloatDataType -> {
                val bComplexDouble = (if (b.dim.d == 2) b else b.reshape(b.shape[0], 1)) as MultiArray<ComplexFloat, D2>
                val ans = solveComplexFloat(a as MultiArray<ComplexFloat, D2>, bComplexDouble)
                if (b.dim.d == 2) ans else ans.reshape(ans.shape[0])
            }
            else -> throw UnsupportedOperationException("Matrices should be complex")
        } as NDArray<T, D>
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