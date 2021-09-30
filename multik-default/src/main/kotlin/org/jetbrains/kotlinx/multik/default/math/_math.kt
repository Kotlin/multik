package org.jetbrains.kotlinx.multik.default.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Returns flat index of maximum element in an ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMax(): Int = DefaultMath.argMax(this)

/**
 * Returns flat index of minimum element in an ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMin(): Int = DefaultMath.argMin(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which an exp function has been applied.
 */
@JvmName("expTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.exp(): NDArray<Double, D> = DefaultMathEx.exp(this)

@JvmName("expFloatD")
public fun <D : Dimension> MultiArray<Float, D>.exp(): NDArray<Float, D> = DefaultMathEx.expF(this)

@JvmName("expComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.exp(): NDArray<ComplexFloat, D> = DefaultMathEx.expCF(this)

@JvmName("expComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.exp(): NDArray<ComplexDouble, D> = DefaultMathEx.expCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a log function has been applied.
 */
@JvmName("logTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.log(): NDArray<Double, D> = DefaultMathEx.log(this)

@JvmName("logFloatD")
public fun <D : Dimension> MultiArray<Float, D>.log(): NDArray<Float, D> = DefaultMathEx.logF(this)

@JvmName("logComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.log(): NDArray<ComplexFloat, D> = DefaultMathEx.logCF(this)

@JvmName("logComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.log(): NDArray<ComplexDouble, D> = DefaultMathEx.logCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a sin function has been applied.
 */
@JvmName("sinTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sin(): NDArray<Double, D> = DefaultMathEx.sin(this)

@JvmName("sinFloatD")
public fun <D : Dimension> MultiArray<Float, D>.sin(): NDArray<Float, D> = DefaultMathEx.sinF(this)

@JvmName("sinComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.sin(): NDArray<ComplexFloat, D> = DefaultMathEx.sinCF(this)

@JvmName("sinComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.sin(): NDArray<ComplexDouble, D> = DefaultMathEx.sinCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cosTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.cos(): NDArray<Double, D> = DefaultMathEx.cos(this)

@JvmName("cosFloatD")
public fun <D : Dimension> MultiArray<Float, D>.cos(): NDArray<Float, D> = DefaultMathEx.cosF(this)

@JvmName("cosComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.cos(): NDArray<ComplexFloat, D> = DefaultMathEx.cosCF(this)

@JvmName("cosComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.cos(): NDArray<ComplexDouble, D> = DefaultMathEx.cosCD(this)

/**
 * Returns cumulative sum of all elements in the given ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.cumSum(): D1Array<T> = DefaultMath.cumSum(this)
