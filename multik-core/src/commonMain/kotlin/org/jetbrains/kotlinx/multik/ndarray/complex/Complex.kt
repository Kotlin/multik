/*
 * Copyright 2020-2023 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * A sealed interface representing a superclass for complex numbers.
 */
public sealed interface Complex {
    public companion object {

        /**
         * Returns a [ComplexFloat] with the given real part.
         *
         * @param re the real part of the complex number
         * @return a [ComplexFloat] number with the given real part and 0f imaginary part
         */
        public fun r(re: Float): ComplexFloat = ComplexFloat(re, 0f)

        /**
         * Returns a [ComplexDouble] with the given real part.
         *
         * @param re the real part of the complex number
         * @return a [ComplexDouble] number with the given real part and 0.0 imaginary part
         */
        public fun r(re: Double): ComplexDouble = ComplexDouble(re, 0.0)

        /**
         * Returns the [ComplexFloat] number representation of the given imaginary part.
         *
         * @param im the imaginary part of the complex number
         * @return a [ComplexFloat] number with the 0f real part and given imaginary part
         */
        public fun i(im: Float): ComplexFloat = ComplexFloat(0f, im)

        /**
         * Returns the [ComplexDouble] number representation of the given imaginary part.
         *
         * @param im the imaginary part of the complex number.
         * @return a [ComplexDouble] number with the 0.0 real part and given imaginary part
         */
        public fun i(im: Double): ComplexDouble = ComplexDouble(0.0, im)

        /**
         * Converts a complex float to a long value.
         *
         * This method takes in a real and imaginary float value and returns a long equivalent. The real
         * value is converted to raw bits and left shifted by 32 bits. The imaginary value is also
         * converted to raw bits and ANDed with the hexadecimal value 0xFFFFFFFFL to get the last 32 bits
         * of the long. The two 32-bit values are then ORed to get the final long value.
         *
         * @param re the real value of the complex number as a float
         * @param im the imaginary value of the complex number as a float
         *
         * @return the long equivalent of the complex number given by the real and imaginary values
         */
        internal fun convertComplexFloatToLong(re: Float, im: Float): Long =
            (re.toRawBits().toLong() shl 32) or (im.toRawBits().toLong() and 0xFFFFFFFFL)
    }
}
