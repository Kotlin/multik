package org.jetbrains.kotlinx.multik.ndarray.operations

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.data.D4
import org.jetbrains.kotlinx.multik.ndarray.data.DN
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.MutableMultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.jvm.JvmName
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.tan

/**
 * Applies operations to this [NDArray] inplace, modifying its elements directly without creating a copy.
 *
 * ```
 * val arr = mk.ndarray(mk[1.0, 2.0, 3.0])
 * arr.inplace { math { sin() } }
 * ```
 *
 * @param init DSL block defining the operations to apply.
 */
public fun <T : Number, D : Dimension> NDArray<T, D>.inplace(init: InplaceOperation<T, D>.() -> Unit) {
    val inplaceOperation = InplaceOperation(this)
    inplaceOperation.init()
}

/** DSL marker annotation that prevents scope leaking in inplace operation builders. */
@DslMarker
public annotation class InplaceDslMarker

/**
 * Base class for all inplace operation scopes.
 *
 * @param T the element type of the array.
 * @param D the dimension type of the array.
 * @property base the mutable array being modified inplace.
 */
@InplaceDslMarker
public sealed class Inplace<T : Number, D : Dimension>(public val base: MutableMultiArray<T, D>)

/**
 * Top-level scope for inplace operations, providing access to [math] blocks.
 *
 * @param T the element type of the array.
 * @param D the dimension type of the array.
 */
public open class InplaceOperation<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) : Inplace<T, D>(base)

/**
 * Applies batched math operations inplace to a 1-dimensional array.
 *
 * @param init DSL block where math operations (e.g., [InplaceMath.sin], [InplaceMath.plus]) are defined.
 */
@JvmName("mathD1")
@Suppress("DuplicatedCode")
public fun <T : Number> InplaceOperation<T, D1>.math(init: InplaceMath<T, D1>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation) {
        for (i in base.indices) {
            if (op is InplaceArithmetic)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
    }
}

/** Applies batched math operations inplace to a 2-dimensional array. */
@JvmName("mathD2")
public fun <T : Number> InplaceOperation<T, D2>.math(init: InplaceMath<T, D2>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1]) {
                if (op is InplaceArithmetic)
                    base[ax0, ax1] = op(base[ax0, ax1], ax0, ax1)
                else
                    base[ax0, ax1] = op(base[ax0, ax1])
            }
}

/** Applies batched math operations inplace to a 3-dimensional array. */
@JvmName("mathD3")
public fun <T : Number> InplaceOperation<T, D3>.math(init: InplaceMath<T, D3>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2]) {
                    if (op is InplaceArithmetic)
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2], ax0, ax1, ax2)
                    else
                        base[ax0, ax1, ax2] = op(base[ax0, ax1, ax2])
                }
}

/** Applies batched math operations inplace to a 4-dimensional array. */
@JvmName("mathD4")
public fun <T : Number> InplaceOperation<T, D4>.math(init: InplaceMath<T, D4>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (ax0 in 0 until base.shape[0])
            for (ax1 in 0 until base.shape[1])
                for (ax2 in 0 until base.shape[2])
                    for (ax3 in 0 until base.shape[3]) {
                        if (op is InplaceArithmetic)
                            base[ax0, ax1, ax2, ax3] = op(base[ax0, ax1, ax2, ax3], ax0, ax1, ax2, ax3)
                        else
                            base[ax0, ax1, ax2, ax3] = op(base[ax0, ax1, ax2, ax3])
                    }
}

/** Applies batched math operations inplace to an N-dimensional array. */
@JvmName("mathDN")
public fun <T : Number> InplaceOperation<T, DN>.math(init: InplaceMath<T, DN>.() -> Unit) {
    val math = InplaceMath(base)
    math.init()
    for (op in math.batchOperation)
        for (i in base.multiIndices) {
            if (op is InplaceArithmetic)
                base[i] = op(base[i], i)
            else
                base[i] = op(base[i])
        }
}

/*
- Fix Sin class computing cos instead of sin (silent data corruption)
- Fix missing throw in operator else branches (exception created but not thrown)
- Extract applyFloatingPointOp helper to centralize unchecked casts
- Convert Exp from sealed class to sealed interface
- Reduce visibility of implementation classes to internal
- Rename Prod to Mul for consistency with Kotlin operator naming
- Improve error messages in type-dispatched operators
- Add KDoc to all public and internal declarations

 */

/**
 * DSL scope for defining inplace math operations.
 *
 * Supports element-wise arithmetic with another array and unary math functions:
 * ```
 * arr.inplace {
 *     math {
 *         sin()
 *         this + otherArray
 *     }
 * }
 * ```
 *
 * Operations are collected into a batch and applied sequentially when the [math] block completes.
 */
public class InplaceMath<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) : InplaceOperation<T, D>(base) {
    internal val batchOperation = mutableListOf<Exp<T>>()

    /** Adds element-wise addition with [other] to the operation batch. */
    public operator fun plus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sum(other))
        return this
    }

    /** Adds element-wise subtraction of [other] to the operation batch. */
    public operator fun minus(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Sub(other))
        return this
    }

    /** Adds element-wise multiplication by [other] to the operation batch. */
    public operator fun times(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Mul(other))
        return this
    }

    /** Adds element-wise division by [other] to the operation batch. */
    public operator fun div(other: MultiArray<T, D>): InplaceMath<T, D> {
        batchOperation.add(Div(other))
        return this
    }
}

/** Adds element-wise sine to the operation batch for [Float] arrays. */
@JvmName("inplaceSinFloat")
public fun <D : Dimension> InplaceMath<Float, D>.sin(): InplaceMath<Float, D> {
    batchOperation.add(Sin())
    return this
}

/** Adds element-wise sine to the operation batch for [Double] arrays. */
@JvmName("inplaceSinDouble")
public fun <D : Dimension> InplaceMath<Double, D>.sin(): InplaceMath<Double, D> {
    batchOperation.add(Sin())
    return this
}

/** Adds element-wise cosine to the operation batch for [Float] arrays. */
@JvmName("inplaceCosFloat")
public fun <D : Dimension> InplaceMath<Float, D>.cos(): InplaceMath<Float, D> {
    batchOperation.add(Cos())
    return this
}

/** Adds element-wise cosine to the operation batch for [Double] arrays. */
@JvmName("inplaceCosDouble")
public fun <D : Dimension> InplaceMath<Double, D>.cos(): InplaceMath<Double, D> {
    batchOperation.add(Cos())
    return this
}

/** Adds element-wise tangent to the operation batch for [Float] arrays. */
@JvmName("inplaceTanFloat")
public fun <D : Dimension> InplaceMath<Float, D>.tan(): InplaceMath<Float, D> {
    batchOperation.add(Tan())
    return this
}

/** Adds element-wise tangent to the operation batch for [Double] arrays. */
@JvmName("inplaceTanDouble")
public fun <D : Dimension> InplaceMath<Double, D>.tan(): InplaceMath<Double, D> {
    batchOperation.add(Tan())
    return this
}

/** Adds element-wise natural logarithm to the operation batch for [Float] arrays. */
@JvmName("inplaceLogFloat")
public fun <D : Dimension> InplaceMath<Float, D>.log(): InplaceMath<Float, D> {
    batchOperation.add(Log())
    return this
}

/** Adds element-wise natural logarithm to the operation batch for [Double] arrays. */
@JvmName("inplaceLogDouble")
public fun <D : Dimension> InplaceMath<Double, D>.log(): InplaceMath<Double, D> {
    batchOperation.add(Log())
    return this
}

/** Adds element-wise ceiling to the operation batch for [Float] arrays. */
@JvmName("inplaceCeilFloat")
public fun <D : Dimension> InplaceMath<Float, D>.ceil(): InplaceMath<Float, D> {
    batchOperation.add(Ceil())
    return this
}

/** Adds element-wise ceiling to the operation batch for [Double] arrays. */
@JvmName("inplaceCeilDouble")
public fun <D : Dimension> InplaceMath<Double, D>.ceil(): InplaceMath<Double, D> {
    batchOperation.add(Ceil())
    return this
}

/** Adds element-wise absolute value to the operation batch for [Int] arrays. */
@JvmName("inplaceAbsInt")
public fun <D : Dimension> InplaceMath<Int, D>.abs(): InplaceMath<Int, D> {
    batchOperation.add(Abs())
    return this
}

/** Adds element-wise absolute value to the operation batch for [Long] arrays. */
@JvmName("inplaceAbsLong")
public fun <D : Dimension> InplaceMath<Long, D>.abs(): InplaceMath<Long, D> {
    batchOperation.add(Abs())
    return this
}

/** Adds element-wise absolute value to the operation batch for [Float] arrays. */
@JvmName("inplaceAbsFloat")
public fun <D : Dimension> InplaceMath<Float, D>.abs(): InplaceMath<Float, D> {
    batchOperation.add(Abs())
    return this
}

/** Adds element-wise absolute value to the operation batch for [Double] arrays. */
@JvmName("inplaceAbsDouble")
public fun <D : Dimension> InplaceMath<Double, D>.abs(): InplaceMath<Double, D> {
    batchOperation.add(Abs())
    return this
}

/**
 * Base interface for all inplace expression operations.
 *
 * Each implementation represents a single element-wise operation (unary or binary).
 * Overloads with index parameters are used by [InplaceArithmetic] operations
 * that need to look up the corresponding element in the right-hand operand.
 */
internal sealed interface Exp<T> {
    operator fun invoke(left: T): T = unsupported()
    operator fun invoke(left: T, ind1: Int): T = unsupported()
    operator fun invoke(left: T, ind1: Int, ind2: Int): T = unsupported()
    operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T = unsupported()
    operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T = unsupported()
    operator fun invoke(left: T, index: IntArray): T = unsupported()

    private fun unsupported(): Nothing =
        throw UnsupportedOperationException("Not supported for local property reference.")
}

/** Marker interface for binary arithmetic operations that require index-based element access. */
internal interface InplaceArithmetic

/** Element-wise addition of [right] operand. */
@Suppress("UNCHECKED_CAST")
internal class Sum<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>, InplaceArithmetic {
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

/** Element-wise subtraction of [right] operand. */
@Suppress("UNCHECKED_CAST")
internal class Sub<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>, InplaceArithmetic {
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

/** Element-wise multiplication by [right] operand. */
@Suppress("UNCHECKED_CAST")
internal class Mul<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>, InplaceArithmetic {
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

/** Element-wise division by [right] operand. */
@Suppress("UNCHECKED_CAST")
internal class Div<T, D : Dimension>(private val right: MultiArray<T, D>) : Exp<T>, InplaceArithmetic {
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

/** Applies a floating-point unary operation, dispatching to [Float] or [Double] overload. */
@Suppress("UNCHECKED_CAST")
private inline fun <T : Number> applyFloatingPointOp(
    left: T, floatOp: (Float) -> Float, doubleOp: (Double) -> Double
): T = when (left) {
    is Float -> floatOp(left) as T
    is Double -> doubleOp(left) as T
    else -> throw IllegalArgumentException("Unsupported type: ${left::class}")
}

/** Element-wise sine operation. */
internal class Sin<T : Number> : Exp<T> {
    override fun invoke(left: T): T = applyFloatingPointOp(left, ::sin, ::sin)
}

/** Element-wise cosine operation. */
internal class Cos<T : Number> : Exp<T> {
    override fun invoke(left: T): T = applyFloatingPointOp(left, ::cos, ::cos)
}

/** Element-wise tangent operation. */
internal class Tan<T : Number> : Exp<T> {
    override fun invoke(left: T): T = applyFloatingPointOp(left, ::tan, ::tan)
}

/** Element-wise natural logarithm operation. */
internal class Log<T : Number> : Exp<T> {
    override fun invoke(left: T): T = applyFloatingPointOp(left, ::ln, ::ln)
}

/** Element-wise ceiling operation. */
internal class Ceil<T : Number> : Exp<T> {
    override fun invoke(left: T): T = applyFloatingPointOp(left, ::ceil, ::ceil)
}

/** Element-wise absolute value operation. Supports [Int], [Long], [Float], and [Double]. */
@Suppress("UNCHECKED_CAST")
internal class Abs<T : Number> : Exp<T> {
    override fun invoke(left: T): T = when (left) {
        is Int -> abs(left) as T
        is Long -> abs(left) as T
        is Float -> abs(left) as T
        is Double -> abs(left) as T
        else -> throw IllegalArgumentException("Unsupported type: ${left::class}")
    }
}

// -- Type-dispatched arithmetic operators used by Sum, Sub, Mul, Div --

/** Type-dispatched addition for [Number] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Number.plus(other: T): T {
    return when (this) {
        is Double if other is Double -> this + other
        is Float if other is Float -> this + other
        is Int if other is Int -> this + other
        is Long if other is Long -> this + other
        is Short if other is Short -> this + other
        is Byte if other is Byte -> this + other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched addition for [Complex] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Complex.plus(other: T): T {
    return when (this) {
        is ComplexFloat if other is ComplexFloat -> this + other
        is ComplexDouble if other is ComplexDouble -> this + other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched subtraction for [Number] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Number.minus(other: T): T {
    return when (this) {
        is Double if other is Double -> this - other
        is Float if other is Float -> this - other
        is Int if other is Int -> this - other
        is Long if other is Long -> this - other
        is Short if other is Short -> this - other
        is Byte if other is Byte -> this - other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched subtraction for [Complex] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Complex.minus(other: T): T {
    return when (this) {
        is ComplexFloat if other is ComplexFloat -> this - other
        is ComplexDouble if other is ComplexDouble -> this - other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched multiplication for [Number] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Number.times(other: T): T {
    return when (this) {
        is Double if other is Double -> this * other
        is Float if other is Float -> this * other
        is Int if other is Int -> this * other
        is Long if other is Long -> this * other
        is Short if other is Short -> this * other
        is Byte if other is Byte -> this * other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched multiplication for [Complex] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Complex.times(other: T): T {
    return when (this) {
        is ComplexFloat if other is ComplexFloat -> this * other
        is ComplexDouble if other is ComplexDouble -> this * other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched division for [Number] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Number.div(other: T): T {
    return when (this) {
        is Double if other is Double -> this / other
        is Float if other is Float -> this / other
        is Int if other is Int -> this / other
        is Long if other is Long -> this / other
        is Short if other is Short -> this / other
        is Byte if other is Byte -> this / other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}

/** Type-dispatched division for [Complex] subtypes. Both operands must have the same runtime type. */
@Suppress("UNCHECKED_CAST")
private operator fun <T> Complex.div(other: T): T {
    return when (this) {
        is ComplexFloat if other is ComplexFloat -> this / other
        is ComplexDouble if other is ComplexDouble -> this / other
        else -> throw IllegalArgumentException("Unsupported operand types: ${this::class} and ${(other as? Any)?.let { it::class }}")
    } as T
}
