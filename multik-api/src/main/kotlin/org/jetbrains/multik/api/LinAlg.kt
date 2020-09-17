package org.jetbrains.multik.api

import org.jetbrains.multik.ndarray.data.*

public interface LinAlg {

    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): Ndarray<T, D2>

    public fun svd()

    public fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double

    public fun cond()

    public fun det()

    public fun matRank()

    public fun solve()

    public fun inv()

    public fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): Ndarray<T, D>

    public fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}