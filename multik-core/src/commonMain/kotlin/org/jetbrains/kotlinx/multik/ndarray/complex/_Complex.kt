package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * Converts this number to a [ComplexFloat] with zero imaginary part.
 *
 * ```
 * val c = 5.toComplexFloat() // 5.0+0.0i
 * ```
 */
public fun Number.toComplexFloat(): ComplexFloat = ComplexFloat(this.toFloat(), 0f)

/**
 * Converts this number to a [ComplexDouble] with zero imaginary part.
 *
 * ```
 * val c = 5.toComplexDouble() // 5.0+0.0i
 * ```
 */
public fun Number.toComplexDouble(): ComplexDouble = ComplexDouble(this.toDouble(), 0.0)

/** Adds this [Byte] to the [other] complex number. */
public operator fun Byte.plus(other: ComplexFloat): ComplexFloat = other + this
/** Adds this [Short] to the [other] complex number. */
public operator fun Short.plus(other: ComplexFloat): ComplexFloat = other + this
/** Adds this [Int] to the [other] complex number. */
public operator fun Int.plus(other: ComplexFloat): ComplexFloat = other + this
/** Adds this [Long] to the [other] complex number. */
public operator fun Long.plus(other: ComplexFloat): ComplexFloat = other + this
/** Adds this [Float] to the [other] complex number. */
public operator fun Float.plus(other: ComplexFloat): ComplexFloat = other + this
/** Adds this [Double] to the [other] complex number, widening the result to [ComplexDouble]. */
public operator fun Double.plus(other: ComplexFloat): ComplexDouble = other + this
/** Adds this [Byte] to the [other] complex number. */
public operator fun Byte.plus(other: ComplexDouble): ComplexDouble = other + this
/** Adds this [Short] to the [other] complex number. */
public operator fun Short.plus(other: ComplexDouble): ComplexDouble = other + this
/** Adds this [Int] to the [other] complex number. */
public operator fun Int.plus(other: ComplexDouble): ComplexDouble = other + this
/** Adds this [Long] to the [other] complex number. */
public operator fun Long.plus(other: ComplexDouble): ComplexDouble = other + this
/** Adds this [Float] to the [other] complex number. */
public operator fun Float.plus(other: ComplexDouble): ComplexDouble = other + this
/** Adds this [Double] to the [other] complex number. */
public operator fun Double.plus(other: ComplexDouble): ComplexDouble = other + this


/** Subtracts the [other] complex number from this [Byte]. */
public operator fun Byte.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Short]. */
public operator fun Short.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Int]. */
public operator fun Int.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Long]. */
public operator fun Long.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Float]. */
public operator fun Float.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Double], widening the result to [ComplexDouble]. */
public operator fun Double.minus(other: ComplexFloat): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Byte]. */
public operator fun Byte.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Short]. */
public operator fun Short.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Int]. */
public operator fun Int.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Long]. */
public operator fun Long.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Float]. */
public operator fun Float.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
/** Subtracts the [other] complex number from this [Double]. */
public operator fun Double.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)


/** Multiplies this [Byte] by the [other] complex number. */
public operator fun Byte.times(other: ComplexFloat): ComplexFloat = other * this
/** Multiplies this [Short] by the [other] complex number. */
public operator fun Short.times(other: ComplexFloat): ComplexFloat = other * this
/** Multiplies this [Int] by the [other] complex number. */
public operator fun Int.times(other: ComplexFloat): ComplexFloat = other * this
/** Multiplies this [Long] by the [other] complex number. */
public operator fun Long.times(other: ComplexFloat): ComplexFloat = other * this
/** Multiplies this [Float] by the [other] complex number. */
public operator fun Float.times(other: ComplexFloat): ComplexFloat = other * this
/** Multiplies this [Double] by the [other] complex number, widening the result to [ComplexDouble]. */
public operator fun Double.times(other: ComplexFloat): ComplexDouble = other * this
/** Multiplies this [Byte] by the [other] complex number. */
public operator fun Byte.times(other: ComplexDouble): ComplexDouble = other * this
/** Multiplies this [Short] by the [other] complex number. */
public operator fun Short.times(other: ComplexDouble): ComplexDouble = other * this
/** Multiplies this [Int] by the [other] complex number. */
public operator fun Int.times(other: ComplexDouble): ComplexDouble = other * this
/** Multiplies this [Long] by the [other] complex number. */
public operator fun Long.times(other: ComplexDouble): ComplexDouble = other * this
/** Multiplies this [Float] by the [other] complex number. */
public operator fun Float.times(other: ComplexDouble): ComplexDouble = other * this
/** Multiplies this [Double] by the [other] complex number. */
public operator fun Double.times(other: ComplexDouble): ComplexDouble = other * this

/** Divides this [Byte] by the [other] complex number. */
public operator fun Byte.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
/** Divides this [Short] by the [other] complex number. */
public operator fun Short.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
/** Divides this [Int] by the [other] complex number. */
public operator fun Int.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
/** Divides this [Long] by the [other] complex number. */
public operator fun Long.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
/** Divides this [Float] by the [other] complex number. */
public operator fun Float.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this, 0f) / other
/** Divides this [Double] by the [other] complex number, widening the result to [ComplexDouble]. */
public operator fun Double.div(other: ComplexFloat): ComplexDouble = ComplexDouble(this, 0.0) / other
/** Divides this [Byte] by the [other] complex number. */
public operator fun Byte.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
/** Divides this [Short] by the [other] complex number. */
public operator fun Short.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
/** Divides this [Int] by the [other] complex number. */
public operator fun Int.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
/** Divides this [Long] by the [other] complex number. */
public operator fun Long.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
/** Divides this [Float] by the [other] complex number. */
public operator fun Float.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
/** Divides this [Double] by the [other] complex number. */
public operator fun Double.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this, 0.0) / other

/**
 * Creates a purely imaginary [ComplexFloat] from this [Float] value.
 *
 * ```
 * val c = 3.5f.i // 0.0+(3.5)i
 * ```
 */
public val Float.i: ComplexFloat get() = ComplexFloat(0f, this)

/**
 * Creates a purely imaginary [ComplexDouble] from this [Double] value.
 *
 * ```
 * val c = 3.5.i // 0.0+(3.5)i
 * ```
 */
public val Double.i: ComplexDouble get() = ComplexDouble(0.0, this)
