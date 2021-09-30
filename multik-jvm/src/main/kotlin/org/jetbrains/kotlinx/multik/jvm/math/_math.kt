package org.jetbrains.kotlinx.multik.jvm.math

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

/**
 * Returns flat index of maximum element in an ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMax(): Int = JvmMath.argMax(this)

/**
 * Returns flat index of minimum element in an ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.argMin(): Int = JvmMath.argMin(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which an exp function has been applied.
 */
@JvmName("expTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.exp(): NDArray<Double, D> = JvmMathEx.exp(this)

@JvmName("expFloatD")
public fun <D : Dimension> MultiArray<Float, D>.exp(): NDArray<Float, D> = JvmMathEx.expF(this)

@JvmName("expComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.exp(): NDArray<ComplexFloat, D> = JvmMathEx.expCF(this)

@JvmName("expComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.exp(): NDArray<ComplexDouble, D> = JvmMathEx.expCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a log function has been applied.
 */
@JvmName("logTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.log(): NDArray<Double, D> = JvmMathEx.log(this)

@JvmName("logFloatD")
public fun <D : Dimension> MultiArray<Float, D>.log(): NDArray<Float, D> = JvmMathEx.logF(this)

@JvmName("logComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.log(): NDArray<ComplexFloat, D> = JvmMathEx.logCF(this)

@JvmName("logComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.log(): NDArray<ComplexDouble, D> = JvmMathEx.logCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a sin function has been applied.
 */
@JvmName("sinTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.sin(): NDArray<Double, D> = JvmMathEx.sin(this)

@JvmName("sinFloatD")
public fun <D : Dimension> MultiArray<Float, D>.sin(): NDArray<Float, D> = JvmMathEx.sinF(this)

@JvmName("sinComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.sin(): NDArray<ComplexFloat, D> = JvmMathEx.sinCF(this)

@JvmName("sinComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.sin(): NDArray<ComplexDouble, D> = JvmMathEx.sinCD(this)

/**
 * Returns an ndarray of Double from the given ndarray to each element of which a cos function has been applied.
 */
@JvmName("cosTD")
public fun <T : Number, D : Dimension> MultiArray<T, D>.cos(): NDArray<Double, D> = JvmMathEx.cos(this)

@JvmName("cosFloatD")
public fun <D : Dimension> MultiArray<Float, D>.cos(): NDArray<Float, D> = JvmMathEx.cosF(this)

@JvmName("cosComplexFloatD")
public fun <D : Dimension> MultiArray<ComplexFloat, D>.cos(): NDArray<ComplexFloat, D> = JvmMathEx.cosCF(this)

@JvmName("cosComplexDoubleD")
public fun <D : Dimension> MultiArray<ComplexDouble, D>.cos(): NDArray<ComplexDouble, D> = JvmMathEx.cosCD(this)

/**
 * Returns cumulative sum of all elements in the given ndarray.
 */
public fun <T : Number, D : Dimension> MultiArray<T, D>.cumSum(): D1Array<T> = JvmMath.cumSum(this)
