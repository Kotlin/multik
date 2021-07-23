package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

@JvmName("invB")
public fun LinAlg.inv(mat: MultiArray<Byte, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invS")
public fun LinAlg.inv(mat: MultiArray<Short, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invI")
public fun LinAlg.inv(mat: MultiArray<Int, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invL")
public fun LinAlg.inv(mat: MultiArray<Long, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invF")
public fun LinAlg.inv(mat: MultiArray<Float, D2>): NDArray<Float, D2> = this.linAlgEx.invF(mat)

@JvmName("invD")
public fun LinAlg.inv(mat: MultiArray<Double, D2>): NDArray<Double, D2> = this.linAlgEx.inv(mat)

@JvmName("invC")
public fun <T : Complex> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<T, D2> = this.linAlgEx.invC(mat)