package org.jetbrains.multik.api

import org.jetbrains.multik.core.*
import kotlin.math.pow

object JvmLinAlg : LinAlg {
    override fun <T : Number> pow(mat: Ndarray<T, D2>, n: Int): Ndarray<T, D2> {
        if (n == 0) return mk.identity<T>(mat.shape[0], mat.dtype)

        return if (n % 2 == 0) {
            val tmp = pow(mat, n / 2)
            dot(tmp, tmp)
        } else {
            dot(mat, pow(mat, n - 1))
        }
    }

    override fun svd() {
        TODO("Not yet implemented")
    }

    override fun <T : Number> norm(mat: Ndarray<T, D2>, p: Int): Double {
        var n = 0.0
        for (element in mat) {
            n += element.toDouble().pow(p)
        }
        return n.pow(1 / p.toDouble())
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

    override fun <T : Number, D : D2> dot(a: Ndarray<T, D2>, b: Ndarray<T, D>): Ndarray<T, D> {
        if (a.shape[1] != b.shape[0])
            throw IllegalArgumentException(
                "Shapes mismatch: shapes "
                        + "${a.shape.joinToString(prefix = "(", postfix = ")")} and "
                        + "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned:"
                        + "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
            )
        return if (b.dim.d == 1) {
            dotVector(a, b as Ndarray<T, D1>) as Ndarray<T, D>
        } else {
            dotMatrix(a, b as Ndarray<T, D2>) as Ndarray<T, D>
        }
    }

    private fun <T : Number> dotMatrix(a: Ndarray<T, D2>, b: Ndarray<T, D2>): Ndarray<T, D2> {
        val newShape = intArrayOf(a.shape[0], b.shape[1])
        val ret = D2Array<T>(
            initMemoryView<T>(newShape[0] * newShape[1], a.dtype),
            shape = newShape, dtype = a.dtype
        )
        for (row in 0 until newShape[0]) {
            for (col in 0 until newShape[1]) {
                var cell: Number = zeroNumber(a.dtype)
                for (i in 0 until b.shape[0]) {
                    cell += a[row, i] * b[i, col]
                }
                ret[row, col] = cell as T
            }
        }
        return ret
    }

    private fun <T : Number> dotVector(a: Ndarray<T, D2>, b: Ndarray<T, D1>): Ndarray<T, D1> {
        val newShape = intArrayOf(a.shape[0])
        val ret = D1Array<T>(initMemoryView<T>(newShape[0], a.dtype), shape = newShape, dtype = a.dtype)
        for (i in 0 until newShape[0]) {
            for (j in 0 until b.shape[0]) {
                ret[i] = ret[i] + a[i, j] * b[j]
                //todo (org.jetbrains.kotlin.codegen.CompilationException: Back-end (JVM) Internal error: wrong bytecode generated)
//                ret[i] += a[i, j] * b[j]
            }
        }
        return ret
    }

    override fun <T : Number> dot(a: Ndarray<T, D1>, b: Ndarray<T, D1>): T {
        var ret: Number = zeroNumber(a.dtype)
        for (i in a.indices) {
            ret += a[i] * b[1]
        }
        return ret as T
    }
}
