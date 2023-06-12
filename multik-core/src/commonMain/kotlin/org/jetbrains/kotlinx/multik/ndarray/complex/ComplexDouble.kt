/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.jvm.JvmInline
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Represents a complex number with double precision.
 * The class is implemented as a double-precision 128-bit complex number.
 *
 * Properties:
 * - [re]: The real part of the complex number.
 * - [im]: The imaginary part of the complex number.
 *
 * Constructors:
 * - [ComplexDouble(re: Double, im: Double)]: Creates a complex number with the given real and imaginary parts.
 * - [ComplexDouble(re: Number, im: Number)]: Creates a complex number with the given real and imaginary parts.
 * - [ComplexDouble(re: Number)]: Creates a complex number with the given real part and an imaginary part of zero.
 *
 * Methods:
 * - [conjugate()]: Returns a new complex number which is the conjugate of the current complex number.
 * - [abs()]: Returns the absolute value of the current complex number.
 * - [angle()]: Returns the angle of the current complex number.
 *
 * Operators:
 * - [plus()]: Adds another value to the current complex number.
 * - [minus()]: Subtracts another value from the current complex number.
 * - [times()]: Multiplies the current complex number by another value.
 * - [div()]: Divides the current complex number by another value.
 * - [unaryPlus()]: Returns a reference to the current complex number.
 * - [unaryMinus()]: Returns the negative of the current complex number.
 * - [component1()]: Returns the real part of the current complex number.
 * - [component2()]: Returns the imaginary part of the current complex number.
 *
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 * @throws ArithmeticException if division by zero or infinity occurs during division.
 */
public sealed interface ComplexDouble : Complex {

    /**
     * The real part of the complex number.
     */
    public val re: Double

    /**
     * The imaginary part of the complex number.
     */
    public val im: Double

    public companion object {
        /**
         * Determines whether the given double value can be accurately represented as a float value or not.
         *
         * @param d is the double value to be checked.
         * @return this method returns a Boolean value,
         * true if the double value can be accurately represented as a float value, false otherwise.
         */
        private fun fitsInFloat(d: Double): Boolean = d.toFloat().toDouble() == d

        /**
         * Creates a [ComplexDouble] with the given real and imaginary values in floating-point format.
         *
         * @param re the real value of the complex number in double format.
         * @param im the imaginary value of the complex number in double format.
         */
        public operator fun invoke(re: Double, im: Double): ComplexDouble {
            return if (fitsInFloat(re) && fitsInFloat(im))
                ComplexDouble32(re, im)
            else
                ComplexDouble64(re, im)
        }

        /**
         * Creates a [ComplexDouble] with the given real and imaginary values in number format.
         *
         * @param re the real value of the complex number in number format.
         * @param im the imaginary value of the complex number in number format.
         */
        public operator fun invoke(re: Number, im: Number): ComplexDouble = invoke(re.toDouble(), im.toDouble())

        /**
         * Creates a [ComplexDouble] with a zero imaginary value.
         * @param re the real value of the complex number in number format.
         */
        public operator fun invoke(re: Number): ComplexDouble {
            return if (fitsInFloat(re.toDouble()))
                ComplexDouble32(re.toDouble(), 0.0)
            else
                ComplexDouble64(re.toDouble(), 0.0)
        }

        /**
         * Represents a [ComplexDouble] number with 1.0 real part and 0f imaginary part.
         */
        public val one: ComplexDouble
            get() = ComplexDouble32(1.0, 0.0)

        /**
         * Represents a [ComplexDouble] number with real and imaginary parts set to 0.0.
         */
        public val zero: ComplexDouble
            get() = ComplexDouble32(0.0, 0.0)

        /**
         * Represents a not-a-number (NaN) value in complex floating point arithmetic.
         */
        public val NaN: ComplexDouble
            get() = ComplexDouble64(Double.NaN, Double.NaN)
    }

    /**
     * Returns the complex conjugate value of the current complex number.
     *
     * @return a new ComplexFloat object representing the complex conjugate of the current complex number.
     * It has the same real part as the current number, but an opposite sign of its imaginary part.
     */
    public fun conjugate(): ComplexDouble = ComplexDouble(re, -im)

    /**
     * Returns the absolute value of the complex number.
     *
     * @return the absolute value of the complex number.
     */
    public fun abs(): Double = sqrt(re * re + im * im)

    /**
     * Returns the angle of the complex number.
     *
     * @return the angle of the complex number as a Double.
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
     * Divides this value by the given byte value.
     *
     * @param other the [Byte] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Byte): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given short value.
     *
     * @param other the [Short] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Short): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given integer value.
     *
     * @param other the [Int] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Int): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given long value.
     *
     * @param other the [Long] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Long): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given float value.
     *
     * @param other the [Float] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Float): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given double value.
     *
     * @param other the [Double] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
     */
    public operator fun div(other: Double): ComplexDouble = ComplexDouble(re / other, im / other)

    /**
     * Divides this value by the given ComplexFloat value.
     *
     * @param other the [ComplexFloat] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
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
     * Divides this value by the given ComplexDouble value.
     *
     * @param other the [ComplexDouble] value to divide this ComplexFloat by.
     * @return a new [ComplexDouble] value after division.
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

    /**
     * Returns the real component of a complex number.
     *
     * @return the real part of the complex number as a Double value.
     */
    public operator fun component1(): Double = re

    /**
     * Returns the imaginary component of a complex number.
     *
     * @return the imaginary part of the complex number as a Double value.
     */
    public operator fun component2(): Double = im
}

/**
 * ComplexDouble64 represents a double-precision 128-bit complex number,
 * which implements the interface `ComplexDouble`.
 *
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 * @constructor creates an instance of ComplexDouble64.
 */
public class ComplexDouble64 internal constructor(
    public override val re: Double, public override val im: Double
) : ComplexDouble {

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexDouble -> re == other.re && im == other.im
        else -> false
    }

    override fun hashCode(): Int = 31 * re.toBits().hashCode() + im.toBits().hashCode()

    override fun toString(): String = "$re+($im)i"
}

/**
 * Represents a complex number of single-precision real and imaginary parts.
 *
 *  @param number The number representing the complex number as a Long value
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 */
@JvmInline
public value class ComplexDouble32 internal constructor(private val number: Long) : ComplexDouble {
    override val re: Double
        get() = Float.fromBits((number shr 32).toInt()).toDouble()

    override val im: Double
        get() = Float.fromBits(number.toInt()).toDouble()

    internal constructor(re: Double, im: Double) : this(Complex.convertComplexFloatToLong(re.toFloat(), im.toFloat()))

    // TODO
    // "https://youtrack.jetbrains.com/issue/KT-24874/Support-custom-equals-and-hashCode-for-value-classes"
    internal fun eq(other: ComplexDouble32): Boolean = when {
        number == other.number -> true
        else -> re == other.re && im == other.im
    }

    internal fun hash(): Int = 31 * number.hashCode()

//    override fun equals(other: Any?): Boolean = when {
//        this === other -> true
//        other is ComplexDouble -> re == other.re && im == other.im
//        else -> false
//    }
//
//    override fun hashCode(): Int = 31 * re.toBits().hashCode() + im.toBits().hashCode()

    override fun toString(): String = "$re+($im)i"
}
