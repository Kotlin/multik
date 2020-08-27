package org.jetbrains.multik.core

import kotlin.reflect.KClass
import kotlin.reflect.jvm.jvmName

public enum class DataType(public val nativeCode: Int, public val itemSize: Int, public val clazz: KClass<out Number>) {
    ByteDataType(1, 1, Byte::class),
    ShortDataType(2, 2, Short::class),
    IntDataType(3, 4, Int::class),
    LongDataType(4, 8, Long::class),
    FloatDataType(5, 4, Float::class),
    DoubleDataType(6, 8, Double::class);

    public companion object {
        public fun of(i: Int): DataType {
            return when (i) {
                1 -> ByteDataType
                2 -> ShortDataType
                3 -> IntDataType
                4 -> LongDataType
                5 -> FloatDataType
                6 -> DoubleDataType
                else -> throw IllegalStateException("One of the primitive types was expected")
            }
        }

        public fun <T : Number> of(element: T): DataType {
            return when (element) {
                is Byte -> ByteDataType
                is Short -> ShortDataType
                is Int -> IntDataType
                is Long -> LongDataType
                is Float -> FloatDataType
                is Double -> DoubleDataType
                else -> throw IllegalStateException("One of the primitive types was expected")
            }
        }

        public inline fun <reified T : Number> of(type: KClass<out T>): DataType = when (type) {
            Byte::class -> ByteDataType
            Short::class -> ShortDataType
            Int::class -> IntDataType
            Long::class -> LongDataType
            Float::class -> FloatDataType
            Double::class -> DoubleDataType
            else -> throw IllegalStateException("One of the primitive types was expected, got ${type.jvmName}")
        }
    }
}