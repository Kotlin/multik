package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray


@JvmName("solveB")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Byte, D2>, b: MultiArray<Byte, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

@JvmName("solveS")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Short, D2>, b: MultiArray<Short, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

@JvmName("solveI")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Int, D2>, b: MultiArray<Int, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

@JvmName("solveL")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Long, D2>, b: MultiArray<Long, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

@JvmName("solveF")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> = this.linAlgEx.solveF(a, b)

@JvmName("solveD")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Double, D2>, b: MultiArray<Double, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

@JvmName("solveC")
public fun <T : Complex, D : Dim2> LinAlg.solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> = this.linAlgEx.solveC(a, b)