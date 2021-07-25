package org.jetbrains.kotlinx.multik.cuda

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

public object CudaLinAlgEx: LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        TODO("Not yet implemented")
    }

}