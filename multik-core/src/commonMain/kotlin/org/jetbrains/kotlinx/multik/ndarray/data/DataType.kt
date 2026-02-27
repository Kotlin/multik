package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.reflect.KClass

/** Resolves the [DataType] for the given [KClass]. Platform-specific implementation. */
@PublishedApi
internal expect inline fun <T : Any> dataTypeOf(type: KClass<out T>): DataType

/**
 * Describes the element type of an [NDArray].
 *
 * Each entry maps to a Kotlin primitive (or [ComplexFloat]/[ComplexDouble]) and carries metadata
 * used by the native JNI layer and memory allocation routines.
 *
 * @param nativeCode integer identifier used in JNI calls to dispatch to the correct native routine.
 * @param itemSize size of a single element in bytes.
 * @param clazz the Kotlin [KClass] of the element type.
 *
 * @property ByteDataType 8-bit signed integer ([Byte]).
 * @property ShortDataType 16-bit signed integer ([Short]).
 * @property IntDataType 32-bit signed integer ([Int]).
 * @property LongDataType 64-bit signed integer ([Long]).
 * @property FloatDataType 32-bit IEEE 754 floating point ([Float]).
 * @property DoubleDataType 64-bit IEEE 754 floating point ([Double]).
 * @property ComplexFloatDataType complex number with 32-bit real and imaginary parts ([ComplexFloat]).
 * @property ComplexDoubleDataType complex number with 64-bit real and imaginary parts ([ComplexDouble]).
 */
public enum class DataType(public val nativeCode: Int, public val itemSize: Int, public val clazz: KClass<out Any>) {
    ByteDataType(1, 1, Byte::class),
    ShortDataType(2, 2, Short::class),
    IntDataType(3, 4, Int::class),
    LongDataType(4, 8, Long::class),
    FloatDataType(5, 4, Float::class),
    DoubleDataType(6, 8, Double::class),
    ComplexFloatDataType(7, 8, ComplexFloat::class),
    ComplexDoubleDataType(8, 16, ComplexDouble::class);

    /**
     * Returns `true` if this type represents a real numeric type ([Byte], [Short], [Int], [Long], [Float], [Double]).
     *
     * @see [isComplex]
     */
    public fun isNumber(): Boolean = when (nativeCode) {
        1, 2, 3, 4, 5, 6 -> true
        else -> false
    }

    /**
     * Returns `true` if this type represents a complex numeric type ([ComplexFloat] or [ComplexDouble]).
     *
     * @see [isNumber]
     */
    public fun isComplex(): Boolean = !isNumber()

    public companion object {

        /**
         * Returns [DataType] by [nativeCode].
         */
        public fun of(i: Int): DataType {
            return when (i) {
                1 -> ByteDataType
                2 -> ShortDataType
                3 -> IntDataType
                4 -> LongDataType
                5 -> FloatDataType
                6 -> DoubleDataType
                7 -> ComplexFloatDataType
                8 -> ComplexDoubleDataType
                else -> throw IllegalStateException("One of the primitive types indexes was expected, got $i")
            }
        }

        /**
         * Returns [DataType] by class of [element].
         */
        public inline fun <T> of(element: T): DataType {
            element ?: throw IllegalStateException("Element is null cannot find type")
            return dataTypeOf(element!!::class)
        }


        /**
         * Returns [DataType] by [KClass] of [type]. [T] is `reified` type.
         */
        public inline fun <T : Any> ofKClass(type: KClass<out T>): DataType = dataTypeOf(type)
    }

    override fun toString(): String {
        return "DataType(nativeCode=$nativeCode, itemSize=$itemSize, class=$clazz)"
    }
}
