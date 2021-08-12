package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.map

public object JvmLinAlgEx : LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> =
        invDouble(if (mat.dtype == DataType.DoubleDataType) (mat as MultiArray<Double, D2>) else mat.map { it.toDouble() })

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> =
        invFloat(mat)

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> {
        val aDouble = when (a.dtype) {
            DataType.DoubleDataType -> a as MultiArray<Double, D2>
            else -> a.map { it.toDouble() }
        }

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
        TODO("Not yet implemented")
    }

    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        TODO("Not yet implemented")
    }

}