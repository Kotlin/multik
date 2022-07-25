/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Extension interface for [Math] for improved type support.
 */
public interface MathEx {
    /**
     * Returns a ndarray of Double from the given ndarray to each element of which an exp function has been applied.
     */
    public fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): NDArray<Double, D>
    /**
     * Returns a ndarray of Float from the given ndarray to each element of which an exp function has been applied.
     */
    public fun <D : Dimension> expF(a: MultiArray<Float, D>): NDArray<Float, D>
    /**
     * Returns a ndarray of [ComplexFloat] from the given ndarray to each element of which an exp function has been applied.
     */
    public fun <D : Dimension> expCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    /**
     * Returns a ndarray of [ComplexDouble] from the given ndarray to each element of which an exp function has been applied.
     */
    public fun <D : Dimension> expCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    /**
     * Returns a ndarray of Double from the given ndarray to each element of which a log function has been applied.
     */
    public fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): NDArray<Double, D>
    /**
     * Returns a ndarray of Float from the given ndarray to each element of which a log function has been applied.
     */
    public fun <D : Dimension> logF(a: MultiArray<Float, D>): NDArray<Float, D>
    /**
     * Returns a ndarray of [ComplexFloat] from the given ndarray to each element of which a log function has been applied.
     */
    public fun <D : Dimension> logCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    /**
     * Returns a ndarray of [ComplexDouble] from the given ndarray to each element of which a log function has been applied.
     */
    public fun <D : Dimension> logCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    /**
     * Returns an ndarray of Double from the given ndarray to each element of which a sin function has been applied.
     */
    public fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): NDArray<Double, D>
    /**
     * Returns an ndarray of Float from the given ndarray to each element of which a sin function has been applied.
     */
    public fun <D : Dimension> sinF(a: MultiArray<Float, D>): NDArray<Float, D>
    /**
     * Returns an ndarray of [ComplexFloat] from the given ndarray to each element of which a sin function has been applied.
     */
    public fun <D : Dimension> sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    /**
     * Returns an ndarray of [ComplexDouble] from the given ndarray to each element of which a sin function has been applied.
     */
    public fun <D : Dimension> sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>

    /**
     * Returns a ndarray of Double from the given ndarray to each element of which a cos function has been applied.
     */
    public fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): NDArray<Double, D>
    /**
     * Returns a ndarray of Float from the given ndarray to each element of which a cos function has been applied.
     */
    public fun <D : Dimension> cosF(a: MultiArray<Float, D>): NDArray<Float, D>
    /**
     * Returns a ndarray of [ComplexFloat] from the given ndarray to each element of which a cos function has been applied.
     */
    public fun <D : Dimension> cosCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
    /**
     * Returns a ndarray of [ComplexDouble] from the given ndarray to each element of which a cos function has been applied.
     */
    public fun <D : Dimension> cosCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>
}