/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import kotlin.reflect.KClass
import kotlin.reflect.jvm.jvmName

/**
 * Describes the type of elements stored in a [Ndarray].
 *
 * @param nativeCode an integer value of the type. Required to define the type in JNI.
 * @param itemSize size of one ndarray element in bytes.
 * @param clazz [KClass] type.
 *
 * @property ByteDataType byte.
 * @property ShortDataType short.
 * @property IntDataType int.
 * @property LongDataType long.
 * @property FloatDataType float.
 * @property DoubleDataType double.
 */
public enum class DataType(public val nativeCode: Int, public val itemSize: Int, public val clazz: KClass<out Number>) {
    ByteDataType(1, 1, Byte::class),
    ShortDataType(2, 2, Short::class),
    IntDataType(3, 4, Int::class),
    LongDataType(4, 8, Long::class),
    FloatDataType(5, 4, Float::class),
    DoubleDataType(6, 8, Double::class);

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
                else -> throw IllegalStateException("One of the primitive types was expected")
            }
        }

        /**
         * Returns [DataType] by class of [element].
         */
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

        /**
         * Returns [DataType] by [KClass] of [type]. [T] is `reified` type.
         */
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