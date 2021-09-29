package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

@JvmName("dotJvmMMNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>) = JvmLinAlgEx.dotMM(this, b)

@JvmName("dotJvmMMComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = JvmLinAlgEx.dotMMComplex(this, b)

@JvmName("dotJvmMVNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = JvmLinAlgEx.dotMV(this, b)

@JvmName("dotJvmMVComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = JvmLinAlgEx.dotMVComplex(this, b)

@JvmName("dotJvmVVNumber")
public infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = JvmLinAlgEx.dotVV(this, b)

@JvmName("dotJvmVVComplex")
public infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = JvmLinAlgEx.dotVVComplex(this, b)