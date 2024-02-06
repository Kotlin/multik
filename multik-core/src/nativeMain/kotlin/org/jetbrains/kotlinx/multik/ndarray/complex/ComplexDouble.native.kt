@file:OptIn(ExperimentalForeignApi::class)

package org.jetbrains.kotlinx.multik.ndarray.complex

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.Vector128
import kotlinx.cinterop.vectorOf

/**
 * Creates a [ComplexDouble] with the given real and imaginary values in floating-point format.
 *
 * @param re the real value of the complex number in double format.
 * @param im the imaginary value of the complex number in double format.
 */
public actual fun ComplexDouble(re: Double, im: Double): ComplexDouble {
    val reBits = re.toBits()
    val reHigherBits = (reBits shr 32).toInt()
    val reLowerBits = (reBits and 0xFFFFFFFFL).toInt()

    val imBits = im.toBits()
    val imHigherBits = (imBits shr 32).toInt()
    val imLowerBits = (imBits and 0xFFFFFFFFL).toInt()

    return NativeComplexDouble(vectorOf(reLowerBits, reHigherBits, imLowerBits, imHigherBits))
}

/**
 * Creates a [ComplexDouble] with the given real and imaginary values in number format.
 *
 * @param re the real value of the complex number in number format.
 * @param im the imaginary value of the complex number in number format.
 */
public actual fun ComplexDouble(re: Number, im: Number): ComplexDouble = ComplexDouble(re.toDouble(), im.toDouble())

/**
 * Represents a complex number with double precision.
 *
 * @property re The real part of the complex number.
 * @property im The imaginary part of the complex number.
 */
public value class NativeComplexDouble internal constructor(private val vector: Vector128) : ComplexDouble {
    public override val re: Double
        get() = vector.getDoubleAt(0)

    public override val im: Double
        get() = vector.getDoubleAt(1)

    override fun eq(other: Any): Boolean =
        when {
            other is NativeComplexDouble && vector == other.vector -> true
            other is NativeComplexDouble -> re == other.re && im == other.im
            else -> false
        }

    override fun hash(): Int = 31 * vector.hashCode()

    override fun toString(): String = "$re+($im)i"
}
