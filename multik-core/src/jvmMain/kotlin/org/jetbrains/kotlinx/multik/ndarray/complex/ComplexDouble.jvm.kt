package org.jetbrains.kotlinx.multik.ndarray.complex

/**
 * Creates a [ComplexDouble] with the given real and imaginary values in floating-point format.
 *
 * @param re the real value of the complex number in double format.
 * @param im the imaginary value of the complex number in double format.
 */
public actual fun ComplexDouble(re: Double, im: Double): ComplexDouble =
    if (fitsInFloat(re) && fitsInFloat(im))
        ComplexDouble32(re, im)
    else
        ComplexDouble64(re, im)

/**
 * Creates a [ComplexDouble] with the given real and imaginary values in number format.
 *
 * @param re the real value of the complex number in number format.
 * @param im the imaginary value of the complex number in number format.
 */
public actual fun ComplexDouble(re: Number, im: Number): ComplexDouble = ComplexDouble(re.toDouble(), im.toDouble())

/**
 * Determines whether the given double value can be accurately represented as a float value or not.
 *
 * @param d is the double value to be checked.
 * @return this method returns a Boolean value,
 * true if the double value can be accurately represented as a float value, false otherwise.
 */
private fun fitsInFloat(d: Double): Boolean = d.toFloat().toDouble() == d

/**
 * ComplexDouble64 represents a double-precision 128-bit complex number,
 * which implements the interface `ComplexDouble`.
 *
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 * @constructor creates an instance of ComplexDouble64.
 */
public class ComplexDouble64 internal constructor(
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

/**
 * Represents a complex number of single-precision real and imaginary parts.
 *
 *  @param number The number representing the complex number as a Long value
 * @property re the real part of the complex number.
 * @property im the imaginary part of the complex number.
 */
@JvmInline
public value class ComplexDouble32 internal constructor(private val number: Long) : ComplexDouble {
    override val re: Double
        get() = Float.fromBits((number shr 32).toInt()).toDouble()

    override val im: Double
        get() = Float.fromBits(number.toInt()).toDouble()

    internal constructor(re: Double, im: Double) : this(Complex.convertComplexFloatToLong(re.toFloat(), im.toFloat()))

    override fun eq(other: Any): Boolean =
        when {
            other is ComplexDouble32 && number == other.number -> true
            other is ComplexDouble32 -> re == other.re && im == other.im
            else -> false
        }

    override fun hash(): Int = 31 * number.hashCode()

    override fun toString(): String = "$re+($im)i"
}
