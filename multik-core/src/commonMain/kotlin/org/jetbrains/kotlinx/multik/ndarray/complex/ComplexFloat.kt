package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.jvm.JvmInline
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Single-precision complex number backed by two [Float] values.
 *
 * Implemented as an [inline value class][JvmInline] wrapping a [Long] — the upper 32 bits store
 * the real part and the lower 32 bits store the imaginary part, avoiding heap allocation on the JVM.
 *
 * Supports standard arithmetic via operator overloads ([plus], [minus], [times], [div]) with
 * both real scalars and other complex numbers. Mixing with [Double] or [ComplexDouble] operands
 * widens the result to [ComplexDouble].
 *
 * Supports destructuring:
 * ```
 * val c = ComplexFloat(3f, 4f)
 * val (re, im) = c // re = 3.0, im = 4.0
 * ```
 *
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 * @see ComplexDouble
 */
@JvmInline
public value class ComplexFloat private constructor(private val number: Long) : Complex {

    /**
     * The real part of the complex number.
     */
    public val re: Float
        get() = Float.fromBits((number shr 32).toInt())


    /**
     * The imaginary part of the complex number.
     */
    public val im: Float
        get() = Float.fromBits(number.toInt())

    /**
     * Creates a [ComplexFloat] with the given real and imaginary values in floating-point format.
     *
     * @param re the real value of the complex number in float format.
     * @param im the imaginary value of the complex number in float format.
     */
    public constructor(re: Float, im: Float) : this(Complex.convertComplexFloatToLong(re, im))

    /**
     * Creates a [ComplexFloat] with the given real and imaginary values in number format.
     *
     * @param re the real value of the complex number in number format.
     * @param im the imaginary value of the complex number in number format.
     */
    public constructor(re: Number, im: Number) : this(re.toFloat(), im.toFloat())

    /**
     * Creates a [ComplexFloat] with a zero imaginary value.
     * @param re the real value of the complex number in number format.
     */
    public constructor(re: Number) : this(re.toFloat(), 0f)

    public companion object {
        /** The complex number `1 + 0i`. */
        public val one: ComplexFloat
            get() = ComplexFloat(1f, 0f)

        /** The complex number `0 + 0i`. */
        public val zero: ComplexFloat
            get() = ComplexFloat(0f, 0f)

        /** A [ComplexFloat] where both real and imaginary parts are [Float.NaN]. */
        public val NaN: ComplexFloat
            get() = ComplexFloat(Float.NaN, Float.NaN)
    }

    /**
     * Returns the complex conjugate (`re - im*i`).
     *
     * @return a new [ComplexFloat] with the same real part and negated imaginary part.
     */
    public fun conjugate(): ComplexFloat = ComplexFloat(re, -im)

    /**
     * Returns the absolute value (modulus) of this complex number: `sqrt(re² + im²)`.
     */
    public fun abs(): Float = hypot(re, im)

    /**
     * Returns the phase angle (argument) of this complex number in radians, computed as `atan2(im, re)`.
     */
    public fun angle(): Float = atan2(im, re)

    /**
     * Adds the other byte value to this value.
     *
     * @param other the [Byte] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: Byte): ComplexFloat = ComplexFloat(re + other, im)

    /**
     * Adds the other short value to this value.
     *
     * @param other the [Short] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: Short): ComplexFloat = ComplexFloat(re + other, im)

    /**
     * Adds the other integer value to this value.
     *
     * @param other the [Int] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: Int): ComplexFloat = ComplexFloat(re + other, im)

    /**
     * Adds the other long value to this value.
     *
     * @param other the [Long] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: Long): ComplexFloat = ComplexFloat(re + other, im)

    /**
     * Adds the other float value to this value.
     *
     * @param other the [Float] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: Float): ComplexFloat = ComplexFloat(re + other, im)

    /**
     * Adds the other double value to this value.
     *
     * @param other the [Double] value to add to this one.
     * @return a new [ComplexDouble] with the result of the addition.
     */
    public operator fun plus(other: Double): ComplexDouble = ComplexDouble(re + other, im.toDouble())

    /**
     * Adds the other ComplexFloat value to this value.
     *
     * @param other the [ComplexFloat] value to add to this one.
     * @return a new [ComplexFloat] with the result of the addition.
     */
    public operator fun plus(other: ComplexFloat): ComplexFloat = ComplexFloat(re + other.re, im + other.im)

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
     * @return a new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Byte): ComplexFloat = ComplexFloat(re - other, im)

    /**
     * Subtracts the other short value from this value.
     *
     * @param other the [Short] value to be subtracted from this value.
     * @return a new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Short): ComplexFloat = ComplexFloat(re - other, im)

    /**
     * Subtracts the other integer value from this value.
     *
     * @param other the [Int] value to be subtracted from this value.
     * @return a new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Int): ComplexFloat = ComplexFloat(re - other, im)

    /**
     * Subtracts the other long value from this value.
     *
     * @param other the [Long] value to be subtracted from this value.
     * @return a new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Long): ComplexFloat = ComplexFloat(re - other, im)

    /**
     * Subtracts the other float value from this value.
     *
     * @param other the [Float] value to be subtracted from this value.
     * @return a new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Float): ComplexFloat = ComplexFloat(re - other, im)

    /**
     * Subtracts the other double value from this value.
     *
     * @param other The [Double] value to be subtracted from this value.
     * @return A new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: Double): ComplexDouble = ComplexDouble(re - other, im.toDouble())

    /**
     * Subtracts the other value from this value.
     *
     * @param other The value to be subtracted from this value.
     * @return A new [ComplexFloat] representing the result of the subtraction operation.
     */
    public operator fun minus(other: ComplexFloat): ComplexFloat = ComplexFloat(re - other.re, im - other.im)

    /**
     * Subtracts the other value from this value.
     *
     * @param other The value to be subtracted from this value.
     * @return A new [ComplexDouble] representing the result of the subtraction operation.
     */
    public operator fun minus(other: ComplexDouble): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /**
     * Multiplies this complex number by the given byte value.
     *
     * @param other the [Byte] value to multiply this complex number by
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: Byte): ComplexFloat = ComplexFloat(re * other, im * other)

    /**
     * Multiplies this complex number by the given short value.
     *
     * @param other the [Short] value to multiply this complex number by
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: Short): ComplexFloat = ComplexFloat(re * other, im * other)

    /**
     * Multiplies this complex number by the given integer value.
     *
     * @param other the [Int] value to multiply this complex number by
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: Int): ComplexFloat = ComplexFloat(re * other, im * other)

    /**
     * Multiplies this complex number by the given long value.
     *
     * @param other the [Long] value to multiply this complex number by
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: Long): ComplexFloat = ComplexFloat(re * other, im * other)

    /**
     * Multiplies this complex number by the given float value.
     *
     * @param other the [Float] value to multiply this complex number by
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: Float): ComplexFloat = ComplexFloat(re * other, im * other)

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
     * @return a new [ComplexFloat] representing the result of the multiplication
     */
    public operator fun times(other: ComplexFloat): ComplexFloat =
        ComplexFloat(re * other.re - im * other.im, re * other.im + other.re * im)

    /**
     * Multiplies this complex number by the given ComplexDouble value.
     *
     * @param other the [ComplexDouble] value to multiply this complex number by
     * @return a new [ComplexDouble] representing the result of the multiplication
     */
    public operator fun times(other: ComplexDouble): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /**
     * Divides this value by the given byte value.
     *
     * @param other the [Byte] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
     */
    public operator fun div(other: Byte): ComplexFloat = ComplexFloat(re / other, im / other)

    /**
     * Divides this value by the given short value.
     *
     * @param other the [Short] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
     */
    public operator fun div(other: Short): ComplexFloat = ComplexFloat(re / other, im / other)

    /**
     * Divides this value by the given integer value.
     *
     * @param other the [Int] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
     */
    public operator fun div(other: Int): ComplexFloat = ComplexFloat(re / other, im / other)

    /**
     * Divides this value by the given long value.
     *
     * @param other the [Long] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
     */
    public operator fun div(other: Long): ComplexFloat = ComplexFloat(re / other, im / other)

    /**
     * Divides this value by the given float value.
     *
     * @param other the [Float] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
     */
    public operator fun div(other: Float): ComplexFloat = ComplexFloat(re / other, im / other)

    /**
     * Divides this value by the given double value.
     *
     * @param other the [Double] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Double): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this complex number by the given [ComplexFloat] value using Smith's algorithm for numerical stability.
     *
     * @param other the [ComplexFloat] divisor.
     * @return a new [ComplexFloat] representing the quotient.
     * @throws ArithmeticException if [other] is zero or if the computation produces infinity/NaN.
     */
    public operator fun div(other: ComplexFloat): ComplexFloat = when {
        kotlin.math.abs(other.re) > kotlin.math.abs(other.im) -> {
            val dr = other.im / other.re
            val dd = other.re + dr * other.im

            if (dd.isNaN() || dd == 0f) throw ArithmeticException("Division by zero or infinity")

            ComplexFloat((re + im * dr) / dd, (im - re * dr) / dd)
        }

        other.im == 0f -> throw ArithmeticException("Division by zero")

        else -> {
            val dr = other.re / other.im
            val dd = other.im + dr * other.re

            if (dd.isNaN() || dd == 0f) throw ArithmeticException("Division by zero or infinity")

            ComplexFloat((re * dr + im) / dd, (im * dr - re) / dd)
        }
    }

    /**
     * Divides this complex number by the given [ComplexDouble] value, widening the result to [ComplexDouble].
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
    public operator fun unaryPlus(): ComplexFloat = this

    /** Returns the negative of this value. */
    public operator fun unaryMinus(): ComplexFloat = ComplexFloat(-re, -im)

    /** Returns the real part for destructuring declarations. */
    public operator fun component1(): Float = re

    /** Returns the imaginary part for destructuring declarations. */
    public operator fun component2(): Float = im

    // TODO
    // "https://youtrack.jetbrains.com/issue/KT-24874/Support-custom-equals-and-hashCode-for-value-classes"
    internal fun eq(other: ComplexFloat): Boolean = when {
        number == other.number -> true
        else -> re == other.re && im == other.im
    }

    internal fun hash(): Int = 31 * number.hashCode()

//    override fun equals(other: Any?): Boolean = when {
//        this === other -> true
//        other is ComplexFloat -> re == other.re && im == other.im
//        else -> false
//    }

//    override fun hashCode(): Int = 31 * re.toBits() + im.toBits()

    /** Returns a string in the form `re+(im)i`, for example `3.0+(4.0)i`. */
    override fun toString(): String = "$re+($im)i"
}
