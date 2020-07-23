package org.jetbrains.multik.api

import org.jetbrains.multik.core.*
import org.jetbrains.multik.jni.NativeLinAlg
import org.jetbrains.multik.jni.NativeMath

/**
 * Alternative names.
 */
typealias mk = Multik


interface Math {

    public fun <T : Number, D : DN> argMax(a: Ndarray<T, D>): Int

    public fun <T : Number, D : DN> argMin(a: Ndarray<T, D>): Int

    public fun <T : Number, D : DN> exp(a: Ndarray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : DN> log(a: Ndarray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : DN> sin(a: Ndarray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : DN> cos(a: Ndarray<T, D>): Ndarray<Double, D>

    public fun <T : Number, D : DN> max(a: Ndarray<T, D>): T

    public fun <T : Number, D : DN> min(a: Ndarray<T, D>): T

    public fun <T : Number, D : DN> sum(a: Ndarray<T, D>): T

    public fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>): D1Array<T>

    public fun <T : Number, D : DN> cumSum(a: Ndarray<T, D>, axis: Int): Ndarray<T, D>
}

interface LinAlg {

    public fun <T : Number> pow(mat: Ndarray<T, D2>, n: Int): Ndarray<T, D2>

    public fun svd()

    public fun <T : Number> norm(mat: Ndarray<T, D2>, p: Int = 2): Double

    public fun cond()

    public fun det()

    public fun matRank()

    public fun solve()

    public fun inv()

    public fun <T : Number, D : D2> dot(a: Ndarray<T, D2>, b: Ndarray<T, D>): Ndarray<T, D>

    public fun <T : Number> dot(a: Ndarray<T, D1>, b: Ndarray<T, D1>): T
}

/**
 * The basic object through which calls all ndarray functions.
 */
object Multik {
    private val loader: Loader = Loader("jni_multik")
    var nativeLibraryLoaded: Boolean = loader.load()
        private set(value) { field = if (value) loader.load() else false }

    var useNative: Boolean = false
        set(value) {
            if (value) nativeLibraryLoaded = value
            if (nativeLibraryLoaded && value) {
                field = value
            } else if (!nativeLibraryLoaded && value) {
                System.err.println("Multik: Native library not found, use JVM implementation.")
            }
        }

    val math get() = if (useNative) NativeMath else JvmMath
    val linalg get() = if (useNative) NativeLinAlg else JvmLinAlg

    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
