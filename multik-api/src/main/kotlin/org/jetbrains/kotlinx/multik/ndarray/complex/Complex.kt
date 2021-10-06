/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

public interface Complex {
    public companion object {

        public fun r(re: Float): ComplexFloat = ComplexFloat(re, 0f)

        public fun r(re: Double): ComplexDouble = ComplexDouble(re, 0.0)

        public fun i(im: Float): ComplexFloat = ComplexFloat(0f, im)

        public fun i(im: Double): ComplexDouble = ComplexDouble(0.0, im)
    }
}

public class ComplexFloat(public val re: Float, public val im: Float) : Complex {

    public constructor(re: Number, im: Number): this(re.toFloat(), im.toFloat())

    public constructor(re: Number): this(re.toFloat(), 0f)

    public companion object {
        public val one: ComplexFloat
            get() = ComplexFloat(1f, 0f)

        public val zero: ComplexFloat
            get() = ComplexFloat(0f, 0f)

        public val NaN: ComplexFloat
            get() = ComplexFloat(Float.NaN, Float.NaN)
    }

    /** Returns complex conjugate value. */
    public fun conjugate(): ComplexFloat = ComplexFloat(re, -im)

    /** Returns absolute value of complex number. */
    public fun abs(): Float = sqrt(re * re + im * im)

    /** Returns angle of complex number. */
    public fun angle(): Float = atan2(im, re)

    /** Adds the other value to this value. */
    public operator fun plus(other: Byte): ComplexFloat = ComplexFloat(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Short): ComplexFloat = ComplexFloat(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Int): ComplexFloat = ComplexFloat(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Long): ComplexFloat = ComplexFloat(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Float): ComplexFloat = ComplexFloat(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Double): ComplexDouble = ComplexDouble(re + other, im.toDouble())

    /** Adds the other value to this value. */
    public operator fun plus(other: ComplexFloat): ComplexFloat = ComplexFloat(re + other.re, im + other.im)

    /** Adds the other value to this value. */
    public operator fun plus(other: ComplexDouble): ComplexDouble = ComplexDouble(re + other.re, im + other.im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Byte): ComplexFloat = ComplexFloat(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Short): ComplexFloat = ComplexFloat(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Int): ComplexFloat = ComplexFloat(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Long): ComplexFloat = ComplexFloat(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Float): ComplexFloat = ComplexFloat(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Double): ComplexDouble = ComplexDouble(re - other, im.toDouble())

    /** Subtracts the other value from this value. */
    public operator fun minus(other: ComplexFloat): ComplexFloat = ComplexFloat(re - other.re, im - other.im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: ComplexDouble): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Byte): ComplexFloat = ComplexFloat(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Short): ComplexFloat = ComplexFloat(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Int): ComplexFloat = ComplexFloat(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Long): ComplexFloat = ComplexFloat(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Float): ComplexFloat = ComplexFloat(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Double): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: ComplexFloat): ComplexFloat =
        ComplexFloat(re * other.re - im * other.im, re * other.im + other.re * im)

    /** Multiplies this value by the other value. */
    public operator fun times(other: ComplexDouble): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /** Divides this value by the other value. */
    public operator fun div(other: Byte): ComplexFloat = ComplexFloat(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Short): ComplexFloat = ComplexFloat(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Int): ComplexFloat = ComplexFloat(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Long): ComplexFloat = ComplexFloat(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Float): ComplexFloat = ComplexFloat(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Double): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: ComplexFloat): ComplexFloat = when {
        abs(other.re) > abs(other.im) -> {
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

    /** Divides this value by the other value. */
    @Suppress("DuplicatedCode")
    public operator fun div(other: ComplexDouble): ComplexDouble = when {
        abs(other.re) > abs(other.im) -> {
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

    public operator fun component1(): Float = re

    public operator fun component2(): Float = im

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexFloat -> re == other.re && im == other.im
        else -> false
    }

    override fun hashCode(): Int = 31 * re.toBits() + im.toBits()

    override fun toString(): String = "$re+($im)i"
}

public class ComplexDouble(public val re: Double, public val im: Double) : Complex {

    public constructor(re: Number, im: Number): this(re.toDouble(), im.toDouble())

    public constructor(re: Number): this(re.toDouble(), 0.0)

    public companion object {
        public val one: ComplexDouble
            get() = ComplexDouble(1.0, 0.0)

        public val zero: ComplexDouble
            get() = ComplexDouble(0.0, 0.0)

        public val NaN: ComplexDouble
            get() = ComplexDouble(Double.NaN, Double.NaN)
    }

    /** Returns complex conjugate value. */
    public fun conjugate(): ComplexDouble = ComplexDouble(re, -im)

    /** Returns absolute value of complex number. */
    public fun abs(): Double = sqrt(re * re + im * im)

    /** Returns angle of complex number. */
    public fun angle(): Double = atan2(im, re)

    /** Adds the other value to this value. */
    public operator fun plus(other: Byte): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Short): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Int): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Long): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Float): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: Double): ComplexDouble = ComplexDouble(re + other, im)

    /** Adds the other value to this value. */
    public operator fun plus(other: ComplexFloat): ComplexDouble = ComplexDouble(re + other.re, im + other.im)

    /** Adds the other value to this value. */
    public operator fun plus(other: ComplexDouble): ComplexDouble = ComplexDouble(re + other.re, im + other.im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Byte): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Short): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Int): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Long): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Float): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: Double): ComplexDouble = ComplexDouble(re - other, im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: ComplexFloat): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /** Subtracts the other value from this value. */
    public operator fun minus(other: ComplexDouble): ComplexDouble = ComplexDouble(re - other.re, im - other.im)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Byte): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Short): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Int): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Long): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Float): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: Double): ComplexDouble = ComplexDouble(re * other, im * other)

    /** Multiplies this value by the other value. */
    public operator fun times(other: ComplexFloat): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /** Multiplies this value by the other value. */
    public operator fun times(other: ComplexDouble): ComplexDouble =
        ComplexDouble(re * other.re - im * other.im, re * other.im + other.re * im)

    /** Divides this value by the other value. */
    public operator fun div(other: Byte): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Short): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Int): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Long): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Float): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: Double): ComplexDouble = ComplexDouble(re / other, im / other)

    /** Divides this value by the other value. */
    public operator fun div(other: ComplexFloat): ComplexDouble = when {
        abs(other.re) > abs(other.im) -> {
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

    /** Divides this value by the other value. */
    @Suppress("DuplicatedCode")
    public operator fun div(other: ComplexDouble): ComplexDouble = when {
        abs(other.re) > abs(other.im) -> {
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

    public operator fun component1(): Double = re

    public operator fun component2(): Double = im

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexDouble -> re == other.re && im == other.im
        else -> false
    }

    override fun hashCode(): Int = 31 * re.toBits().hashCode() + im.toBits().hashCode()

    override fun toString(): String = "$re+($im)i"
}