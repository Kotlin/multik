package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble32
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble64
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.reflect.KClass

@PublishedApi
@Suppress("NOTHING_TO_INLINE")
internal actual inline fun <T : Any> dataTypeOf(type: KClass<out T>): DataType =
    when (type) {
        Byte::class -> ByteDataType
        Short::class -> ShortDataType
        Int::class -> IntDataType
        Long::class -> LongDataType
        Float::class -> FloatDataType
        Double::class -> DoubleDataType
        ComplexFloat::class -> ComplexFloatDataType
        ComplexDouble::class, ComplexDouble64::class, ComplexDouble32::class -> ComplexDoubleDataType
        else -> throw IllegalStateException("One of the primitive types was expected, got ${type.simpleName}")
    }