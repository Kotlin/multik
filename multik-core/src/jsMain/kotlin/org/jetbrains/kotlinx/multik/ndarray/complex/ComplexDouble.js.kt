package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * Creates a [ComplexDouble] with the given real and imaginary values in floating-point format.
 *
 * @param re the real value of the complex number in double format.
 * @param im the imaginary value of the complex number in double format.
 */
public actual fun ComplexDouble(re: Double, im: Double): ComplexDouble = JsComplexDouble(re, im)

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
public class JsComplexDouble internal constructor(
    public override val re: Double, public override val im: Double
) : ComplexDouble {

    override fun eq(other: Any): Boolean = equals(other)

    override fun hash(): Int = hashCode()

    override fun equals(other: Any?): Boolean = when {
        this === other -> true
        other is ComplexDouble -> re == other.re && im == other.im
        else -> false
    }

    override fun hashCode(): Int = 31 * re.toBits().hashCode() + im.toBits().hashCode()

    override fun toString(): String = "$re+($im)i"
}
