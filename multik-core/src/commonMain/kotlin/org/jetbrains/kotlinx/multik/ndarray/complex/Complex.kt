package org.jetbrains.kotlinx.multik.ndarray.complex

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex.Companion.i
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex.Companion.r


/**
 * Common supertype for complex number representations.
 *
 * Multik provides two concrete complex types:
 * - [ComplexFloat] — single-precision complex number backed by a pair of [Float] values.
 * - [ComplexDouble] — double-precision complex number backed by a pair of [Double] values.
 *
 * Use the companion factory methods [r] and [i] to create purely real or purely imaginary values:
 * ```
 * val realOnly = Complex.r(3.0)   // 3.0 + 0.0i
 * val imagOnly = Complex.i(2.0f)  // 0.0 + 2.0i
 * ```
 *
 * @see ComplexFloat
 * @see ComplexDouble
 */
public sealed interface Complex {
    public companion object {

        /**
         * Creates a [ComplexFloat] with the given real part and zero imaginary part.
         *
         * @param re the real part of the complex number.
         * @return a [ComplexFloat] equal to `re + 0i`.
         */
        public fun r(re: Float): ComplexFloat = ComplexFloat(re, 0f)

        /**
         * Creates a [ComplexDouble] with the given real part and zero imaginary part.
         *
         * @param re the real part of the complex number.
         * @return a [ComplexDouble] equal to `re + 0i`.
         */
        public fun r(re: Double): ComplexDouble = ComplexDouble(re, 0.0)

        /**
         * Creates a [ComplexFloat] with the given imaginary part and zero real part.
         *
         * @param im the imaginary part of the complex number.
         * @return a [ComplexFloat] equal to `0 + im*i`.
         */
        public fun i(im: Float): ComplexFloat = ComplexFloat(0f, im)

        /**
         * Creates a [ComplexDouble] with the given imaginary part and zero real part.
         *
         * @param im the imaginary part of the complex number.
         * @return a [ComplexDouble] equal to `0 + im*i`.
         */
        public fun i(im: Double): ComplexDouble = ComplexDouble(0.0, im)

        /** Packs two [Float] bit patterns into a single [Long] for [ComplexFloat] storage. */
        internal fun convertComplexFloatToLong(re: Float, im: Float): Long =
            (re.toRawBits().toLong() shl 32) or (im.toRawBits().toLong() and 0xFFFFFFFFL)
    }
}
