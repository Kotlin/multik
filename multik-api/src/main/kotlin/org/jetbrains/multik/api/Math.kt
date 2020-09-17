package org.jetbrains.multik.api

import org.jetbrains.multik.ndarray.data.D1Array
import org.jetbrains.multik.ndarray.data.Dimension
import org.jetbrains.multik.ndarray.data.MultiArray
import org.jetbrains.multik.ndarray.data.Ndarray

public interface Math {

    public fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int

    public fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int

    public fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T>

    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): Ndarray<T, D>
}