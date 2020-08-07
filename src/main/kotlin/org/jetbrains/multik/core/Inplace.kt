package org.jetbrains.multik.core

import kotlin.math.*
import org.jetbrains.multik.core.Dimension as Dimension

private fun unsupported(): Nothing = throw UnsupportedOperationException("Not supported for local property reference.")

public fun <T : Number, D : Dimension> Ndarray<T, D>.inplace(init: InplaceOperation<T, D>.() -> Unit): Unit {
    val inplaceOperation = InplaceOperation(this)
    inplaceOperation.init()
}

@DslMarker
public annotation class InplaceDslMarker

@InplaceDslMarker
public sealed class Inplace<T : Number, D : Dimension>(public val base: MutableMultiArray<T, D>)

public open class InplaceOperation<T : Number, D : Dimension>(base: MutableMultiArray<T, D>) : Inplace<T, D>(base)

@JvmName("mathD1")
public fun <T : Number> InplaceOperation<T, D1>.math(init: InplaceMath<T, D1>.() -> Unit): Unit {
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
public fun <T : Number> InplaceOperation<T, D2>.math(init: InplaceMath<T, D2>.() -> Unit): Unit {
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
public fun <T : Number> InplaceOperation<T, D3>.math(init: InplaceMath<T, D3>.() -> Unit): Unit {
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
public fun <T : Number> InplaceOperation<T, D4>.math(init: InplaceMath<T, D4>.() -> Unit): Unit {
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
public fun <T : Number> InplaceOperation<T, DN>.math(init: InplaceMath<T, DN>.() -> Unit): Unit {
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

public sealed class Exp<T : Number> {
    public open operator fun invoke(left: T): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T = unsupported()
    public open operator fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T = unsupported()
    public open operator fun invoke(left: T, index: IntArray): T = unsupported()
}

public interface Arith

public class Sum<T : Number>(private val right: MultiArray<T, Dimension>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left + (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T = left + (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left + (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left + (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left + (right as MultiArray<T, DN>)[index]
}

public class Sub<T : Number>(private val right: MultiArray<T, Dimension>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left - (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T = left - (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left - (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left - (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left - (right as MultiArray<T, DN>)[index]
}

public class Prod<T : Number>(private val right: MultiArray<T, Dimension>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left * (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T = left * (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left * (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left * (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left * (right as MultiArray<T, DN>)[index]
}

public class Div<T : Number>(private val right: MultiArray<T, Dimension>) : Exp<T>(), Arith {
    override fun invoke(left: T, ind1: Int): T = left / (right as MultiArray<T, D1>)[ind1]
    override fun invoke(left: T, ind1: Int, ind2: Int): T = left / (right as MultiArray<T, D2>)[ind1, ind2]
    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int): T =
        left / (right as MultiArray<T, D3>)[ind1, ind2, ind3]

    override fun invoke(left: T, ind1: Int, ind2: Int, ind3: Int, ind4: Int): T =
        left / (right as MultiArray<T, D4>)[ind1, ind2, ind3, ind4]

    override fun invoke(left: T, index: IntArray): T = left / (right as MultiArray<T, DN>)[index]
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