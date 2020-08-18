package org.jetbrains.multik.api

import org.jetbrains.multik.core.*
import org.jetbrains.multik.jni.NativeLinAlg
import org.jetbrains.multik.jni.NativeMath

/**
 * Alternative names.
 */
public typealias mk = Multik


public interface Math {

    public fun <T : Number, D : Dimension> argMax(a: MultiArray<T, D>): Int

    public fun <T : Number, D : Dimension> argMin(a: MultiArray<T, D>): Int

    public fun <T : Number, D : Dimension> exp(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> log(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> sin(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> cos(a: MultiArray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : Dimension> max(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> min(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> sum(a: MultiArray<T, D>): T

    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>): D1Array<T>

    public fun <T : Number, D : Dimension> cumSum(a: MultiArray<T, D>, axis: Int): Ndarray<T, D>
}

public interface LinAlg {

    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): Ndarray<T, D2>

    public fun svd()

    public fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int = 2): Double

    public fun cond()

    public fun det()

    public fun matRank()

    public fun solve()

    public fun inv()

    public fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): Ndarray<T, D>

    public fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}

/**
 * The basic object through which calls all ndarray functions.
 */
public object Multik {
    private val loader: Loader = Loader("multik_jni")
    public var nativeLibraryLoaded: Boolean = loader.load()
        private set(value) {
            field = if (value) loader.load() else false
        }

    public var useNative: Boolean = false
        set(value) {
            if (value) nativeLibraryLoaded = value
            if (nativeLibraryLoaded && value) {
                field = value
            } else if (!nativeLibraryLoaded && value) {
                System.err.println("Multik: Native library not found, use JVM implementation.")
            }
        }

    public val math: Math get() = if (useNative) NativeMath else JvmMath
    public val linalg: LinAlg get() = if (useNative) NativeLinAlg else JvmLinAlg

    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
