package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Creates a [ComplexDouble] with the given [re] and [im] parts.
 *
 * @param re the real part.
 * @param im the imaginary part.
 */
public expect fun ComplexDouble(re: Double, im: Double): ComplexDouble

/**
 * Creates a [ComplexDouble] by converting the given [Number] values to [Double].
 *
 * @param re the real part.
 * @param im the imaginary part.
 */
public expect fun ComplexDouble(re: Number, im: Number): ComplexDouble

/**
 * Creates a [ComplexDouble] with the given real part and zero imaginary part.
 *
 * @param re the real part.
 */
public fun ComplexDouble(re: Number): ComplexDouble = ComplexDouble(re.toDouble(), 0.0)

/**
 * Double-precision complex number backed by two [Double] values.
 *
 * Declared as an `expect`/`actual` interface so that platform implementations can use inline
 * classes or other efficient representations. On JVM, this is implemented as a value class
 * wrapping two doubles.
 *
 * Supports standard arithmetic via operator overloads ([plus], [minus], [times], [div]) with
 * both real scalars and other complex numbers.
 *
 * Supports destructuring:
 * ```
 * val c = ComplexDouble(3.0, 4.0)
 * val (re, im) = c // re = 3.0, im = 4.0
 * ```
 *
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 * @see ComplexFloat
 */
public interface ComplexDouble : Complex {

    /**
     * The real part of the complex number.
     */
    public val re: Double

    /**
     * The imaginary part of the complex number.
     */
    public val im: Double

    public companion object {
        /** The complex number `1 + 0i`. */
        public val one: ComplexDouble
            get() = ComplexDouble(1.0, 0.0)

        /** The complex number `0 + 0i`. */
        public val zero: ComplexDouble
            get() = ComplexDouble(0.0, 0.0)

        /** A [ComplexDouble] where both real and imaginary parts are [Double.NaN]. */
        public val NaN: ComplexDouble
            get() = ComplexDouble(Double.NaN, Double.NaN)
    }

    /**
     * Returns the complex conjugate (`re - im*i`).
     *
     * @return a new [ComplexDouble] with the same real part and negated imaginary part.
     */
    public fun conjugate(): ComplexDouble = ComplexDouble(re, -im)

    /**
     * Returns the absolute value (modulus) of this complex number: `sqrt(re² + im²)`.
     */
    public fun abs(): Double = sqrt(re * re + im * im)

    /**
     * Returns the phase angle (argument) of this complex number in radians, computed as `atan2(im, re)`.
     */
    public fun angle(): Double = atan2(im, re)

    /**
     * Adds the other byte value to this value.
     *
     * @param other the [Byte] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Byte): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other short value to this value.
     *
     * @param other the [Short] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Short): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other integer value to this value.
     *
     * @param other the [Int] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Int): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other long value to this value.
     *
     * @param other the [Long] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Long): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other float value to this value.
     *
     * @param other the [Float] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Float): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other double value to this value.
     *
     * @param other the [Double] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Double): ComplexDouble = ComplexDouble(re + other, im)

    /**
     * Adds the other ComplexFloat value to this value.
     *
     * @param other the [ComplexFloat] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: ComplexFloat): ComplexDouble = ComplexDouble(re + other.re, im + other.im)

    /**
     * Adds the other ComplexDouble value to this value.
     *
     * @param other the [ComplexDouble] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: ComplexDouble): ComplexDouble = ComplexDouble(re + other.re, im + other.im)

    /**
     * Subtracts the other byte value from this value.
     *
     * @param other the [Byte] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Byte): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other short value from this value.
     *
     * @param other the [Short] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Short): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other integer value from this value.
     *
     * @param other the [Int] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Int): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other long value from this value.
     *
     * @param other the [Long] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Long): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other float value from this value.
     *
     * @param other the [Float] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Float): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other double value from this value.
     *
     * @param other the [Double] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Double): ComplexDouble = ComplexDouble(re - other, im)

    /**
     * Subtracts the other ComplexFloat value from this value.
     *
     * @param other the [ComplexFloat] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: ComplexFloat): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /**
     * Subtracts the other ComplexDouble value from this value.
     *
     * @param other the [ComplexDouble] value to be subtracted from this value.
     * @return a new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: ComplexDouble): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /**
     * Multiplies this complex number by the given byte value.
     *
     * @param other the [Byte] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Byte): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given short value.
     *
     * @param other the [Short] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Short): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given integer value.
     *
     * @param other the [Int] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Int): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given long value.
     *
     * @param other the [Long] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Long): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given float value.
     *
     * @param other the [Float] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Float): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given double value.
     *
     * @param other the [Double] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: Double): ComplexDouble = ComplexDouble(re * other, im * other)

    /**
     * Multiplies this complex number by the given ComplexFloat value.
     *
     * @param other the [ComplexFloat] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: ComplexFloat): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /**
     * Multiplies this complex number by the given ComplexDouble value.
     *
     * @param other the [ComplexDouble] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: ComplexDouble): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /**
     * Divides this complex number by the given [Byte] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Byte): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [Short] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Short): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [Int] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Int): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [Long] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Long): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [Float] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Float): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [Double] value.
     *
     * @param other the scalar divisor.
     * @return a new [ComplexDouble] representing the quotient.
     */
    public operator fun div(other: Double): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [ComplexFloat] value using Smith's algorithm for numerical stability.
     *
     * @param other the [ComplexFloat] divisor.
     * @return a new [ComplexDouble] representing the quotient.
     * @throws ArithmeticException if [other] is zero or if the computation produces infinity/NaN.
     */
    public operator fun div(other: ComplexFloat): ComplexDouble = when {
        kotlin.math.abs(other.re) > kotlin.math.abs(other.im) -> {
            val dr = other.im / other.re
            val dd = other.re + dr * other.im

            if (dd.isNaN() || dd == 0f) throw ArithmeticException("Division by zero or infinity")

            ComplexDouble((re + im * dr) / dd, (im - re * dr) / dd)
        }

        other.im == 0f -> throw ArithmeticException("Division by zero")

        else -> {
            val dr = other.re / other.im
            val dd = other.im + dr * other.re

            if (dd.isNaN() || dd == 0f) throw ArithmeticException("Division by zero or infinity")

            ComplexDouble((re * dr + im) / dd, (im * dr - re) / dd)
        }
    }

    /**
     * Divides this complex number by the given [ComplexDouble] value using Smith's algorithm for numerical stability.
     *
     * @param other the [ComplexDouble] divisor.
     * @return a new [ComplexDouble] representing the quotient.
     * @throws ArithmeticException if [other] is zero or if the computation produces infinity/NaN.
     */
    public operator fun div(other: ComplexDouble): ComplexDouble = when {
        kotlin.math.abs(other.re) > kotlin.math.abs(other.im) -> {
            val dr = other.im / other.re
            val dd = other.re + dr * other.im

            if (dd.isNaN() || dd == 0.0) throw ArithmeticException("Division by zero or infinity")

            ComplexDouble((re + im * dr) / dd, (im - re * dr) / dd)
        }

        other.im == 0.0 -> throw ArithmeticException("Division by zero")

        else -> {
            val dr = other.re / other.im
            val dd = other.im + dr * other.re

            if (dd.isNaN() || dd == 0.0) throw ArithmeticException("Division by zero or infinity")

            ComplexDouble((re * dr + im) / dd, (im * dr - re) / dd)
        }
    }

    /** Returns this value. */
    public operator fun unaryPlus(): ComplexDouble = this

    /** Returns the negative of this value. */
    public operator fun unaryMinus(): ComplexDouble = ComplexDouble(-re, -im)

    /** Returns the real part for destructuring declarations. */
    public operator fun component1(): Double = re

    /** Returns the imaginary part for destructuring declarations. */
    public operator fun component2(): Double = im

    // TODO("https://youtrack.jetbrains.com/issue/KT-24874/Support-custom-equals-and-hashCode-for-value-classes")
    /** Structural equality check — workaround for value class `equals` limitations. */
    public fun eq(other: Any): Boolean

    /** Hash code — workaround for value class `hashCode` limitations. */
    public fun hash(): Int
}
