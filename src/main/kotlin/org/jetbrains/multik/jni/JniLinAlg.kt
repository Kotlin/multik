package org.jetbrains.multik.jni

import org.jetbrains.multik.api.JvmLinAlg
import org.jetbrains.multik.api.LinAlg
import org.jetbrains.multik.api.identity
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.core.*

object NativeLinAlg : LinAlg {
    //todo
    override fun <T : Number> pow(mat: Ndarray<T, D2>, n: Int): Ndarray<T, D2> {
        if (n == 0) return mk.identity<T>(mat.shape[0], mat.dtype)

        return if (n % 2 == 0) {
            val tmp = JvmLinAlg.pow(mat, n / 2)
            dot(tmp, tmp)
        } else {
            dot(mat, JvmLinAlg.pow(mat, n - 1))
        }
    }

    override fun svd() {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: Ndarray<T, D2>, p: Int): Double {
        TODO("Not yet implemented")
    }

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

    //TODO (Double and Number type)
    override fun <T : Number, D : D2> dot(a: Ndarray<T, D2>, b: Ndarray<T, D>): Ndarray<T, D> {
        if (a.shape[1] != b.shape[0])
            throw IllegalArgumentException(
                "Shapes mismatch: shapes "
                        + "${a.shape.joinToString(prefix = "(", postfix = ")")} and "
                        + "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned:"
                        + "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
            )
        return if (b.dim.d == 1) {
            val shape = intArrayOf(a.shape[0])
            val c = DoubleArray(shape[0])
            JniLinAlg.dot(
                (a.data as MemoryViewDoubleArray).data,
                a.shape[0],
                a.shape[1],
                (b.data as MemoryViewDoubleArray).data,
                c
            )
            D1Array(MemoryViewDoubleArray(c), 0, shape, dtype = a.dtype) as Ndarray<T, D>
        } else {
            val shape = intArrayOf(a.shape[0], b.shape[1])
            //            val c = initMemoryView<T>(shape[0] * shape[1], a.dtype)
            val c = DoubleArray(shape[0] * shape[1])
            JniLinAlg.dot(
                (a.data as MemoryViewDoubleArray).data,
                a.shape[0],
                a.shape[1],
                (b.data as MemoryViewDoubleArray).data,
                b.shape[1],
                c
            )
            D2Array(MemoryViewDoubleArray(c), 0, shape, dtype = a.dtype) as Ndarray<T, D>
        }
    }

    override fun <T : Number> dot(a: Ndarray<T, D1>, b: Ndarray<T, D1>): T {
        TODO("Not yet implemented")
    }
}

object JniLinAlg {
    external fun <T : Number> pow(mat: Ndarray<T, D2>, n: Int): Ndarray<T, D2>
    external fun <T : Number> norm(mat: Ndarray<T, D2>, p: Int): Double
    external fun dot(a: DoubleArray, n: Int, m: Int, b: DoubleArray, k: Int, c: DoubleArray): Unit
    external fun dot(a: DoubleArray, n: Int, m: Int, b: DoubleArray, c: DoubleArray)
}