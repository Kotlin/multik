/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.jvm.JvmInline
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Represents a complex number with single precision.
 * The class is implemented as a single-precision 64-bit complex number.
 *
 * Properties:
 * - [re]: The real part of the complex number.
 * - [im]: The imaginary part of the complex number.
 *
 * Constructors:
 * - [ComplexFloat(re: Float, im: Float)]: Creates a complex number with the given real and imaginary parts.
 * - [ComplexFloat(re: Number, im: Number)]: Creates a complex number with the given real and imaginary parts.
 * - [ComplexFloat(re: Number)]: Creates a complex number with the given real part and an imaginary part of zero.
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
        /**
         * Represents a [ComplexFloat] number with 1f real part and 0f imaginary part.
         */
        public val one: ComplexFloat
            get() = ComplexFloat(1f, 0f)

        /**
         * Represents a [ComplexFloat] number with real and imaginary parts set to 0f.
         */
        public val zero: ComplexFloat
            get() = ComplexFloat(0f, 0f)

        /**
         * Represents a not-a-number (NaN) value in complex floating point arithmetic.
         */
        public val NaN: ComplexFloat
            get() = ComplexFloat(Float.NaN, Float.NaN)
    }

    /**
     * Returns the complex conjugate value of the current complex number.
     *
     * @return a new ComplexFloat object representing the complex conjugate of the current complex number.
     * It has the same real part as the current number, but an opposite sign of its imaginary part.
     */
    public fun conjugate(): ComplexFloat = ComplexFloat(re, -im)

    /**
     * Returns the absolute value of the complex number.
     *
     * @return the absolute value of the complex number.
     */
    public fun abs(): Float = sqrt(re * re + im * im)

    /**
     * Returns the angle of the complex number.
     *
     * @return the angle of the complex number as a Float.
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
     * Divides this value by the given ComplexFloat value.
     *
     * @param other the [ComplexFloat] value to divide this ComplexFloat by.
     * @return a new [ComplexFloat] value after division.
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
    public operator fun unaryPlus(): ComplexFloat = this

    /** Returns the negative of this value. */
    public operator fun unaryMinus(): ComplexFloat = ComplexFloat(-re, -im)

    /**
     * Returns the real component of a complex number.
     *
     * @return the real part of the complex number as a Float value.
     */
    public operator fun component1(): Float = re

    /**
     * Returns the imaginary component of a complex number.
     *
     * @return the imaginary part of the complex number as a Float value.
     */
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

    /**
     * Returns a string representation of the complex number object in the form of
     * "real_part + (imaginary_part)i"
     *
     * @return the string representation of the complex number object
     */
    override fun toString(): String = "$re+($im)i"
}