/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

public fun Number.toComplexFloat(): ComplexFloat = ComplexFloat(this.toFloat(), 0f)

public fun Number.toComplexDouble(): ComplexDouble = ComplexDouble(this.toDouble(), 0.0)

public operator fun Byte.plus(other: ComplexFloat): ComplexFloat = other + this
public operator fun Short.plus(other: ComplexFloat): ComplexFloat = other + this
public operator fun Int.plus(other: ComplexFloat): ComplexFloat = other + this
public operator fun Long.plus(other: ComplexFloat): ComplexFloat = other + this
public operator fun Float.plus(other: ComplexFloat): ComplexFloat = other + this
public operator fun Double.plus(other: ComplexFloat): ComplexDouble = other + this
public operator fun Byte.plus(other: ComplexDouble): ComplexDouble = other + this
public operator fun Short.plus(other: ComplexDouble): ComplexDouble = other + this
public operator fun Int.plus(other: ComplexDouble): ComplexDouble = other + this
public operator fun Long.plus(other: ComplexDouble): ComplexDouble = other + this
public operator fun Float.plus(other: ComplexDouble): ComplexDouble = other + this
public operator fun Double.plus(other: ComplexDouble): ComplexDouble = other + this


public operator fun Byte.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
public operator fun Short.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
public operator fun Int.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
public operator fun Long.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
public operator fun Float.minus(other: ComplexFloat): ComplexFloat = ComplexFloat(this - other.re, -other.im)
public operator fun Double.minus(other: ComplexFloat): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Byte.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Short.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Int.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Long.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Float.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)
public operator fun Double.minus(other: ComplexDouble): ComplexDouble = ComplexDouble(this - other.re, -other.im)


public operator fun Byte.times(other: ComplexFloat): ComplexFloat = other * this
public operator fun Short.times(other: ComplexFloat): ComplexFloat = other * this
public operator fun Int.times(other: ComplexFloat): ComplexFloat = other * this
public operator fun Long.times(other: ComplexFloat): ComplexFloat = other * this
public operator fun Float.times(other: ComplexFloat): ComplexFloat = other * this
public operator fun Double.times(other: ComplexFloat): ComplexDouble = other * this
public operator fun Byte.times(other: ComplexDouble): ComplexDouble = other * this
public operator fun Short.times(other: ComplexDouble): ComplexDouble = other * this
public operator fun Int.times(other: ComplexDouble): ComplexDouble = other * this
public operator fun Long.times(other: ComplexDouble): ComplexDouble = other * this
public operator fun Float.times(other: ComplexDouble): ComplexDouble = other * this
public operator fun Double.times(other: ComplexDouble): ComplexDouble = other * this

public operator fun Byte.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
public operator fun Short.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
public operator fun Int.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
public operator fun Long.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this.toFloat(), 0f) / other
public operator fun Float.div(other: ComplexFloat): ComplexFloat = ComplexFloat(this, 0f) / other
public operator fun Double.div(other: ComplexFloat): ComplexDouble = ComplexDouble(this, 0.0) / other
public operator fun Byte.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
public operator fun Short.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
public operator fun Int.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
public operator fun Long.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
public operator fun Float.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this.toDouble(), 0.0) / other
public operator fun Double.div(other: ComplexDouble): ComplexDouble = ComplexDouble(this, 0.0) / other

public val Byte.i: ComplexDouble
    get() = ComplexDouble(0, this.toDouble())
public val Short.i: ComplexDouble
    get() = ComplexDouble(0, this.toDouble())
public val Int.i: ComplexDouble
    get() = ComplexDouble(0, this.toDouble())
public val Long.i: ComplexDouble
    get() = ComplexDouble(0, this.toDouble())
public val Float.i: ComplexFloat
    get() = ComplexFloat(0, this)
public val Double.i: ComplexDouble
    get() = ComplexDouble(0, this)