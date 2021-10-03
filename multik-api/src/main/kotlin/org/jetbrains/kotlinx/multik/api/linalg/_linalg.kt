package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

@JvmName("dotDefMMNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMM(this, b)

@JvmName("dotDefMMComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = mk.linalg.linAlgEx.dotMMComplex(this, b)

@JvmName("dotDefMVNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMV(this, b)

@JvmName("dotDefMVComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = mk.linalg.linAlgEx.dotMVComplex(this, b)

@JvmName("dotDefVVNumber")
public infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVV(this, b)

@JvmName("dotDefVVComplex")
public infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = mk.linalg.linAlgEx.dotVVComplex(this, b)