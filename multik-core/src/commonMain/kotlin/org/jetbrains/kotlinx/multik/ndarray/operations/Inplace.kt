/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.jvm.JvmName
import kotlin.math.*

private fun unsupported(): Nothing = throw UnsupportedOperationException("Not supported for local property reference.")

public fun <T : Number, D : Dimension> NDArray<T, D>.inplace(init: InplaceOperation<T, D>.() -> Unit) {
    val inplaceOperation = InplaceOperation(this)
    inplaceOperation.init()
}

@DslMarker
public annotation class InplaceDslMarker

@InplaceDslMarker
public sealed class Inplace<T : Number, D : Dimension>(public val base: MutableMultiArray<T, D>)

public open class InplaceOperation<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) : Inplace<T, D>(base)

@JvmName("mathD1")
@Suppress("DuplicatedCode")
public fun <T : Number> InplaceOperation<T, D1>.math(init: InplaceMath<T, D1>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation) {
        for (i in base.indices) {
            if (op is Arith)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
    }
}

@JvmName("mathD2")
public fun <T : Number> InplaceOperation<T, D2>.math(init: InplaceMath<T, D2>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1]) {
                if (op is Arith)
                    base[ax0, ax1] = op(base[ax0, ax1], ax0, ax1)
                else
                    base[ax0, ax1] = op(base[ax0, ax1])
            }
}

@JvmName("mathD3")
public fun <T : Number> InplaceOperation<T, D3>.math(init: InplaceMath<T, D3>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2]) {
                    if (op is Arith)
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2], ax0, ax1, ax2)
                    else
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2])
                }
}

@JvmName("mathD4")
public fun <T : Number> InplaceOperation<T, D4>.math(init: InplaceMath<T, D4>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2])
                    for (ax3 in 0 until base.shape[3]) {
                        if (op is Arith)
                            base[ax0, ax1, ax2, ax3] = op(base[ax0, ax1, ax2, ax3], ax0, ax1, ax2, ax3)
                        else
                            base[ax0, ax1, ax2, ax3] = op(base[ax0, ax1, ax2, ax3])
                    }
}

@JvmName("mathDN")
public fun <T : Number> InplaceOperation<T, DN>.math(init: InplaceMath<T, DN>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (i in base.multiIndices) {
            if (op is Arith)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
}

public class InplaceMath<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) : InplaceOperation<T, D>(base) {
    internal val batchOperation = ArrayList<Exp<T>>()

    public operator fun plus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sum(other))
        return this
    }

    public operator fun minus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sub(other))
        return this
    }

    public operator fun times(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Prod(other))
        return this
    }

    public operator fun div(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Div(other))
        return this
    }
}

@JvmName("inplaceSinFloat")
public fun <D : Dimension> InplaceMath<Float, D>.sin(): InplaceMath<Float, D> {
    batchOperation.add(Sin())
    return this
}

@JvmName("inplaceSinDouble")
public fun <D : Dimension> InplaceMath<Double, D>.sin(): InplaceMath<Double, D> {
    batchOperation.add(Sin())
    return this
}

@JvmName("inplaceCosFloat")
public fun <D : Dimension> InplaceMath<Float, D>.cos(): InplaceMath<Float, D> {
    batchOperation.add(Cos())
    return this
}

@JvmName("inplaceCosDouble")
public fun <D : Dimension> InplaceMath<Double, D>.cos(): InplaceMath<Double, D> {
    batchOperation.add(Cos())
    return this
}

@JvmName("inplaceTanFloat")
public fun <D : Dimension> InplaceMath<Float, D>.tan(): InplaceMath<Float, D> {
    batchOperation.add(Tan())
    return this
}

@JvmName("inplaceTanDouble")
public fun <D : Dimension> InplaceMath<Double, D>.tan(): InplaceMath<Double, D> {
    batchOperation.add(Tan())
    return this
}

@JvmName("inplaceLogFloat")
public fun <D : Dimension> InplaceMath<Float, D>.log(): InplaceMath<Float, D> {
    batchOperation.add(Log())
    return this
}

@JvmName("inplaceLogDouble")
public fun <D : Dimension> InplaceMath<Double, D>.log(): InplaceMath<Double, D> {
    batchOperation.add(Log())
    return this
}

@JvmName("inplaceCeilFloat")
public fun <D : Dimension> InplaceMath<Float, D>.ceil(): InplaceMath<Float, D> {
    batchOperation.add(Ceil())
    return this
}

@JvmName("inplaceCeilDouble")
public fun <D : Dimension> InplaceMath<Double, D>.ceil(): InplaceMath<Double, D> {
    batchOperation.add(Ceil())
    return this
}

@JvmName("inplaceAbsInt")
public fun <D : Dimension> InplaceMath<Int, D>.abs(): InplaceMath<Int, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsLong")
public fun <D : Dimension> InplaceMath<Long, D>.abs(): InplaceMath<Long, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsFloat")
public fun <D : Dimension> InplaceMath<Float, D>.abs(): InplaceMath<Float, D> {
    batchOperation.add(Abs())
    return this
}

@JvmName("inplaceAbsDouble")
public fun <D : Dimension> InplaceMath<Double, D>.abs(): InplaceMath<Double, D> {
    batchOperation.add(Abs())
    return this
}

public sealed class Exp<T> {
    public open operator fun invoke(left: T): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T = unsupported()
    public open operator fun invoke(left: T, index: IntArray): T = unsupported()
}

public interface Arith

public class Sum<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T =
        if (left is Number) left + (right as MultiArray<T, D1>)[ind1] else (left as Complex) + (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        if (left is Number) left + (right as MultiArray<T, D2>)[ind1, ind2] else (left as Complex) + (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        if (left is Number) left + (right as MultiArray<T, D3>)[ind1, ind2, ind3] else (left as Complex) + (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        if (left is Number) left + (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4] else (left as Complex) + (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T =
        if (left is Number) left + (right as MultiArray<T, DN>)[index] else (left as Complex) + (right as MultiArray<T, DN>)[index]
}

public class Sub<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T =
        if (left is Number) left - (right as MultiArray<T, D1>)[ind1] else (left as Complex) - (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        if (left is Number) left - (right as MultiArray<T, D2>)[ind1, ind2] else (left as Complex) - (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        if (left is Number) left - (right as MultiArray<T, D3>)[ind1, ind2, ind3] else (left as Complex) - (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        if (left is Number) left - (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4] else (left as Complex) - (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T =
        if (left is Number) left - (right as MultiArray<T, DN>)[index] else (left as Complex) - (right as MultiArray<T, DN>)[index]
}

public class Prod<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T =
        if (left is Number) left * (right as MultiArray<T, D1>)[ind1] else (left as Complex) * (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        if (left is Number) left * (right as MultiArray<T, D2>)[ind1, ind2] else (left as Complex) * (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        if (left is Number) left * (right as MultiArray<T, D3>)[ind1, ind2, ind3] else (left as Complex) * (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        if (left is Number) left * (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4] else (left as Complex) * (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T =
        if (left is Number) left * (right as MultiArray<T, DN>)[index] else (left as Complex) * (right as MultiArray<T, DN>)[index]
}

public class Div<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T =
        if (left is Number) left / (right as MultiArray<T, D1>)[ind1] else (left as Complex) / (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T =
        if (left is Number) left / (right as MultiArray<T, D2>)[ind1, ind2] else (left as Complex) / (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        if (left is Number) left / (right as MultiArray<T, D3>)[ind1, ind2, ind3] else (left as Complex) / (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        if (left is Number) left / (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4] else (left as Complex) / (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T =
        if (left is Number) left / (right as MultiArray<T, DN>)[index] else (left as Complex) / (right as MultiArray<T, DN>)[index]
}

public class Sin<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> cos(left) as T
        is Double -> cos(left) as T
        else -> throw Exception("")
    }
}

public class Cos<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> cos(left) as T
        is Double -> cos(left) as T
        else -> throw Exception("")
    }
}

public class Tan<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> tan(left) as T
        is Double -> tan(left) as T
        else -> throw Exception("")
    }
}

public class Log<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> ln(left) as T
        is Double -> ln(left) as T
        else -> throw Exception("")
    }
}

public class Ceil<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Float -> ceil(left) as T
        is Double -> ceil(left) as T
        else -> throw Exception("")
    }
}

public class Abs<T : Number> : Exp<T>() {
    override fun invoke(left: T): T = when (left) {
        is Int -> abs(left) as T
        is Long -> abs(left) as T
        is Float -> abs(left) as T
        is Double -> abs(left) as T
        else -> throw Exception("")
    }
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Number.plus(other: T): T {
    return when {
        this is Double && other is Double -> this + other
        this is Float && other is Float -> this + other
        this is Int && other is Int -> this + other
        this is Long && other is Long -> this + other
        this is Short && other is Short -> this + other
        this is Byte && other is Byte -> this + other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Complex.plus(other: T): T {
    return when {
        this is ComplexFloat && other is ComplexFloat -> this + other
        this is ComplexDouble && other is ComplexDouble -> this + other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Number.minus(other: T): T {
    return when {
        this is Double && other is Double -> this - other
        this is Float && other is Float -> this - other
        this is Int && other is Int -> this - other
        this is Long && other is Long -> this - other
        this is Short && other is Short -> this - other
        this is Byte && other is Byte -> this - other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Complex.minus(other: T): T {
    return when {
        this is ComplexFloat && other is ComplexFloat -> this - other
        this is ComplexDouble && other is ComplexDouble -> this - other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Number.times(other: T): T {
    return when {
        this is Double && other is Double -> this * other
        this is Float && other is Float -> this * other
        this is Int && other is Int -> this * other
        this is Long && other is Long -> this * other
        this is Short && other is Short -> this * other
        this is Byte && other is Byte -> this * other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Complex.times(other: T): T {
    return when {
        this is ComplexFloat && other is ComplexFloat -> this * other
        this is ComplexDouble && other is ComplexDouble -> this * other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Number.div(other: T): T {
    return when {
        this is Double && other is Double -> this / other
        this is Float && other is Float -> this / other
        this is Int && other is Int -> this / other
        this is Long && other is Long -> this / other
        this is Short && other is Short -> this / other
        this is Byte && other is Byte -> this / other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}

@Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")
private inline operator fun <T> Complex.div(other: T): T {
    return when {
        this is ComplexFloat && other is ComplexFloat -> this / other
        this is ComplexDouble && other is ComplexDouble -> this / other
        else -> Exception("Unknown type: ${this::class}")
    } as T
}