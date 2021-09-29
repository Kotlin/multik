package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.jni.linalg.NativeLinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

@JvmName("dotNatMMNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>) = NativeLinAlgEx.dotMM(this, b)

@JvmName("dotNatMMComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2> = NativeLinAlgEx.dotMMComplex(this, b)

@JvmName("dotNatMVNumber")
public infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = NativeLinAlgEx.dotMV(this, b)

@JvmName("dotNatMVComplex")
public infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1> = NativeLinAlgEx.dotMVComplex(this, b)

@JvmName("dotNatVVNumber")
public infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = NativeLinAlgEx.dotVV(this, b)

@JvmName("dotNatVVComplex")
public infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T = NativeLinAlgEx.dotVVComplex(this, b)