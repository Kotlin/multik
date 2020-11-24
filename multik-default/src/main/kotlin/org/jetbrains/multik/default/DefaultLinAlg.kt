package org.jetbrains.multik.default

import org.jetbrains.multik.api.LinAlg
import org.jetbrains.multik.jni.NativeLinAlg
import org.jetbrains.multik.jvm.JvmLinAlg
import org.jetbrains.multik.ndarray.data.*

public object DefaultLinAlg : LinAlg {
    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): Ndarray<T, D2> {
        return JvmLinAlg.pow(mat, n)
    }

    override fun svd() {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double = JvmLinAlg.norm(mat, p)

    override fun cond() {
        TODO("Not yet implemented")
    }

    override fun det() {
        TODO("Not yet implemented")
    }

    override fun matRank() {
        TODO("Not yet implemented")
    }

    override fun solve() {
        TODO("Not yet implemented")
    }

    override fun inv() {
        TODO("Not yet implemented")
    }

    //TODO()
    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): Ndarray<T, D> {
        return NativeLinAlg.dot(a, b)
    }

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        return NativeLinAlg.dot(a, b)
    }

}